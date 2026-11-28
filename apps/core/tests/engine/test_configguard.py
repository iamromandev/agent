"""`ConfigGuard.react`: what the agent does when the config watch reports a change (rules, stores, MCP, marker)."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from core.engine.configguard import ConfigGuard, GlobalMarker
from core.engine.configwatch import Cause, ConfigWatch, TrustRevoked, WatchedSet
from core.engine.lsp.config import LspSetup, ServerSpec
from core.engine.lsp.hub import LspHub
from core.engine.mcp.trust import TrustStore
from core.engine.permissions.config import project_settings_path
from core.engine.permissions.setup import Rules, prepare
from core.testing.helpers import assert_all_gone
from core.testing.lsp_fakes import fake_lsp


class FakeMcp:
    """The hub's side of a revoke: `withdraw` (synchronous) and `stop` (on the loop), recorded with their threads."""

    def __init__(self) -> None:
        self.names = {"p": "project", "g": "global"}
        self.withdrawn: list[tuple[list[str], str]] = []
        self.stopped: list[tuple[list[str], str]] = []
        self.stop_threads: list[threading.Thread] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.fail = False

    def sources(self) -> Mapping[str, str]:
        return self.names

    def withdraw(self, names: Sequence[str], reason: str) -> list[str]:
        if self.fail:
            raise RuntimeError("hub broken")
        self.withdrawn.append((list(names), reason))
        return list(names)

    async def stop(self, names: Sequence[str], reason: str) -> None:
        self.stop_threads.append(threading.current_thread())
        self.stopped.append((list(names), reason))


class Env:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.workspace = tmp_path / "ws"
        (self.workspace / ".agent").mkdir(parents=True)
        self.xdg = {"XDG_CONFIG_HOME": str(tmp_path / "config"), "XDG_DATA_HOME": str(tmp_path / "data")}
        self.config = tmp_path / "config" / "agent"
        self.data = tmp_path / "data" / "agent"
        self.perm_store = TrustStore(self.data / "permission-trust.json")
        self.mcp_store = TrustStore(self.data / "mcp-trust.json")
        self.marker = GlobalMarker(self.data / "config-changed.json")
        self.mcp = FakeMcp()
        self.stopped = self.mcp.stopped
        self.sources = self.mcp.names
        self.warnings: list[str] = []

    def rules(self) -> Rules:
        self.config.mkdir(parents=True, exist_ok=True)
        project_settings_path(self.workspace).write_text(json.dumps({"permissions": {"allow": ["bash(ls)"]}}))
        rules = prepare(self.workspace, self.config / "settings.json", self.perm_store).rules
        rules.trust_project()
        return rules

    def guard(self, rules: Rules, lsp: LspHub | None = None) -> ConfigGuard:
        guard = ConfigGuard(
            self.workspace,
            rules,
            self.mcp,
            [self.perm_store, self.mcp_store],
            self.marker,
            home=self.tmp / "home",
            lsp=lsp,
        )
        guard.listeners.append(self.warnings.append)
        return guard

    def watch(self, guard: ConfigGuard) -> ConfigWatch:
        watch = ConfigWatch(WatchedSet.for_run(self.workspace, self.xdg, self.tmp / "home"), guard.react)
        guard.bind(watch)
        return watch


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def _allowed(rules: Rules) -> bool:
    return rules.allow("bash", {"command": "ls"}, None) is not None


async def test_a_project_change_revokes_trust_forgets_both_stores_and_says_so(env: Env) -> None:
    rules = env.rules()
    env.perm_store.trust(env.workspace, b"x")
    env.mcp_store.trust(env.workspace, b"y")
    env.perm_store.trust(env.tmp / "other", b"z")
    guard = env.guard(rules)
    watch = env.watch(guard)
    assert _allowed(rules)

    (env.workspace / ".agent" / "settings.json").write_text("{}")
    watch.check(Cause.tool("bash", "./setup.sh"))
    await guard.settle()

    assert not _allowed(rules)
    assert env.perm_store.is_trusted(env.workspace, b"x") is False
    assert env.mcp_store.is_trusted(env.workspace, b"y") is False
    assert env.perm_store.is_trusted(env.tmp / "other", b"z") is True
    assert env.stopped == [(["p"], "config changed")]
    [text] = env.warnings
    assert text == (
        "config changed by bash (./setup.sh): .agent/settings.json; "
        "project trust revoked — the next launch will ask"
    )
    assert guard.changes and guard.changes[0].areas == frozenset({"project"})


async def test_the_guards_own_store_writes_are_not_a_second_change(env: Env) -> None:
    rules = env.rules()
    env.perm_store.trust(env.workspace, b"x")
    guard = env.guard(rules)
    watch = env.watch(guard)

    (env.workspace / ".agent" / "settings.json").write_text("{}")
    watch.check(Cause.background())

    assert watch.check(Cause.background()) is None
    assert len(env.warnings) == 1


async def test_a_trust_store_change_alone_also_revokes(env: Env) -> None:
    rules = env.rules()
    guard = env.guard(rules)
    watch = env.watch(guard)

    env.data.mkdir(parents=True, exist_ok=True)
    (env.data / "permission-trust.json").write_text("{}")
    watch.check(Cause.background())

    assert not _allowed(rules)
    assert "permission-trust.json" in env.warnings[0]
    assert "a background process or terminal (during the turn)" in env.warnings[0]


async def test_a_global_change_writes_the_marker_disables_global_rules_and_stops_the_changed_global_servers(env: Env) -> None:
    env.config.mkdir(parents=True)
    (env.config / "settings.json").write_text(json.dumps({"permissions": {"allow": ["bash(ls)"]}}))
    rules = prepare(env.workspace, env.config / "settings.json", env.perm_store).rules
    guard = env.guard(rules)
    watch = env.watch(guard)
    assert _allowed(rules)

    (env.config / "mcp.json").write_text("{}")
    (env.config / "settings.json").write_text(json.dumps({"permissions": {"allow": ["bash(ls)", "bash(curl*)"]}}))
    watch.check(Cause.tool("bash", "x"))
    await guard.settle()

    assert not _allowed(rules)
    assert env.marker.exists()
    marker = json.loads((env.data / "config-changed.json").read_text())
    assert marker["marker"] == 1 and len(marker["files"]) == 2 and "at" in marker
    assert env.stopped == [(["g"], "config changed")]
    assert "global allow rules and hooks are off until you trust them at the next launch" in env.warnings[0]
    assert "project trust" not in env.warnings[0]
    assert watch.check(Cause.background()) is None  # the marker write was accepted


async def test_a_global_settings_change_does_not_stop_global_servers(env: Env) -> None:
    guard = env.guard(env.rules())
    watch = env.watch(guard)

    (env.config / "settings.json").write_text("{}")
    watch.check(Cause.background())
    await guard.settle()

    assert env.stopped == []


async def test_global_deny_rules_still_apply_after_a_global_change(env: Env) -> None:
    env.config.mkdir(parents=True)
    (env.config / "settings.json").write_text(json.dumps({"permissions": {"deny": ["bash(rm*)"]}}))
    rules = prepare(env.workspace, env.config / "settings.json", env.perm_store).rules
    guard = env.guard(rules)
    watch = env.watch(guard)

    (env.config / "settings.json").write_text(json.dumps({"permissions": {"deny": ["bash(rm*)"], "allow": ["bash"]}}))
    watch.check(Cause.background())

    assert rules.deny("bash", {"command": "rm x"}, None) is not None


async def test_a_project_and_a_global_change_together_do_both(env: Env) -> None:
    rules = env.rules()
    guard = env.guard(rules)
    watch = env.watch(guard)

    (env.workspace / ".agent" / "mcp.json").write_text("{}")
    (env.config / "commands").mkdir()
    (env.config / "commands" / "x.md").write_text("hi")
    watch.check(Cause.hook("PreToolUse"))

    [text] = env.warnings
    assert text.startswith("config changed by hook PreToolUse: ")
    assert "project trust revoked" in text and "global allow rules and hooks are off" in text
    assert env.marker.exists() and not _allowed(rules)


async def test_a_marker_that_is_removed_by_a_command_is_written_again(env: Env) -> None:
    env.marker.write(["old"])
    guard = env.guard(env.rules())
    watch = env.watch(guard)

    (env.data / "config-changed.json").unlink()
    watch.check(Cause.background())

    assert env.marker.exists()
    assert "marker" in env.warnings[0]


@pytest.mark.parametrize("how", ["removed", "emptied"])
async def test_a_rewritten_marker_keeps_the_original_file_list(env: Env, how: str) -> None:
    env.data.mkdir(parents=True)
    GlobalMarker(env.marker.path).write(["~/.config/agent/settings.json"])  # an earlier run's
    marker = GlobalMarker(env.marker.path)  # this launch's
    env.marker = marker
    guard = env.guard(env.rules())
    watch = env.watch(guard)

    if how == "removed":
        marker.path.unlink()
    else:
        marker.path.write_text(json.dumps({"marker": 1, "files": []}))
    watch.check(Cause.background())

    info = marker.read()
    assert info is not None and info.files[0] == "~/.config/agent/settings.json"
    assert len(info.files) == len(set(info.files))


async def test_a_failing_store_or_marker_is_named_in_the_text_and_never_raises(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    rules = env.rules()
    env.perm_store.trust(env.workspace, b"x")
    guard = env.guard(rules)
    watch = env.watch(guard)

    def broken(src: object, dst: object) -> None:
        raise OSError(28, "no space")

    monkeypatch.setattr(os, "replace", broken)
    (env.workspace / ".agent" / "settings.json").write_text("{}")
    (env.config / "settings.json").write_text("{}")
    watch.check(Cause.background())

    [text] = env.warnings
    assert "could not update permission-trust.json (ENOSPC); deleted it instead" in text
    assert not env.perm_store.path.exists()  # the fallback: no trust left, so the next launch asks
    assert "the next launch will NOT ask about the global config: could not write config-changed.json (ENOSPC)" in text
    assert not _allowed(rules)  # the switches flip even when the files cannot be written


async def test_a_store_that_can_be_neither_updated_nor_deleted_says_the_next_launch_will_not_ask(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    rules = env.rules()
    env.perm_store.trust(env.workspace, b"x")
    guard = env.guard(rules)
    watch = env.watch(guard)

    def denied(*args: object) -> bool:
        raise PermissionError(13, "denied")

    monkeypatch.setattr(env.perm_store, "forget", denied)
    monkeypatch.setattr(os, "unlink", denied)
    (env.workspace / ".agent" / "settings.json").write_text("{}")
    watch.check(Cause.background())

    [text] = env.warnings
    assert "project trust revoked for this run; the next launch will NOT ask: could not update permission-trust.json (EACCES)" in text
    assert not _allowed(rules)


async def test_each_failing_step_is_named_and_the_others_still_run(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    rules = env.rules()
    env.mcp_store.trust(env.workspace, b"y")
    guard = env.guard(rules)
    watch = env.watch(guard)
    env.mcp.fail = True

    def denied(*args: object) -> bool:
        raise PermissionError(13, "denied")

    monkeypatch.setattr(env.perm_store, "forget", denied)
    (env.workspace / ".agent" / "settings.json").write_text("{}")
    watch.check(Cause.background())

    [text] = env.warnings
    assert "could not stop the MCP servers (RuntimeError)" in text
    assert "permission-trust.json" in text
    assert not env.mcp_store.holds(env.workspace)  # the other store was still forgotten
    assert not _allowed(rules)


async def test_a_command_that_rewrites_files_while_the_guard_forgets_is_still_seen(env: Env) -> None:
    """poc2: a child writes the settings and re-trusts the folder right after the guard's own store write."""
    rules = env.rules()
    workspace = env.workspace

    class Racing(TrustStore):
        def forget(self, workspace: Path) -> bool:
            wrote = super().forget(workspace)
            project_settings_path(workspace).write_text('{"permissions": {"allow": ["bash"]}}')
            TrustStore.trust(self, workspace, b"evil")
            return wrote

    store = Racing(env.perm_store.path)
    store.trust(workspace, b"x")
    guard = ConfigGuard(workspace, rules, env.mcp, [store, env.mcp_store], env.marker, home=env.tmp / "home")
    guard.listeners.append(env.warnings.append)
    watch = env.watch(guard)

    (workspace / ".agent" / "mcp.json").write_text("{}")
    watch.check(Cause.tool("bash", "x"))

    assert len(env.warnings) == 2  # the first change, then the settings the child wrote in the gap
    assert ".agent/settings.json" in env.warnings[1]
    assert not store.holds(workspace)  # re-trusted in the gap: the store was deleted
    assert watch.check(Cause.background()) is None


async def test_the_guard_accepts_only_the_files_it_wrote(env: Env) -> None:
    rules = env.rules()
    env.perm_store.trust(env.workspace, b"x")
    config = env.config

    class Racing(TrustStore):
        def forget(self, workspace: Path) -> bool:
            (config / "settings.json").write_text("{}")  # a child, between the change and the guard's write
            return super().forget(workspace)

    store = Racing(env.perm_store.path)
    guard = ConfigGuard(env.workspace, rules, env.mcp, [store], env.marker, home=env.tmp / "home")
    guard.listeners.append(env.warnings.append)
    watch = env.watch(guard)

    (env.workspace / ".agent" / "mcp.json").write_text("{}")
    watch.check(Cause.background())

    assert len(env.warnings) == 2 and "settings.json" in env.warnings[1]
    assert "global allow rules and hooks are off" in env.warnings[1]


async def test_the_servers_are_withdrawn_before_react_returns(env: Env) -> None:
    guard = env.guard(env.rules())
    watch = env.watch(guard)

    (env.workspace / ".agent" / "settings.json").write_text("{}")
    watch.check(Cause.background())

    assert env.mcp.withdrawn == [(["p"], "config changed")]  # no await needed: tools() is right at once
    await guard.settle()
    assert env.stopped == [(["p"], "config changed")]


async def test_a_reaction_off_the_loop_stops_the_servers_and_tells_the_listeners_on_the_loop(env: Env) -> None:
    rules = env.rules()
    guard = env.guard(rules)
    heard: list[threading.Thread] = []
    guard.listeners.append(lambda text: heard.append(threading.current_thread()))
    watch = env.watch(guard)  # bound inside the running loop: the guard keeps it

    (env.workspace / ".agent" / "settings.json").write_text("{}")
    await asyncio.to_thread(watch.check, Cause.background())
    await asyncio.sleep(0)
    await guard.settle()

    main = threading.current_thread()
    assert heard == [main] and len(env.warnings) == 1
    assert env.mcp.stop_threads == [main] and env.stopped == [(["p"], "config changed")]
    assert env.mcp.withdrawn and not _allowed(rules)


async def test_only_the_global_mcp_json_itself_stops_the_global_servers(env: Env) -> None:
    guard = env.guard(env.rules())
    watch = env.watch(guard)

    (env.config / "commands").mkdir()
    (env.config / "commands" / "mcp.json").write_text("{}")  # a command file named mcp.json
    watch.check(Cause.background())
    await guard.settle()
    assert env.stopped == []

    (env.workspace / ".agent" / "mcp.json").write_text("{}")  # the project's: only the project's servers
    watch.check(Cause.background())
    await guard.settle()
    assert env.stopped == [(["p"], "config changed")]


def test_clearing_the_marker_resets_its_state(env: Env) -> None:
    env.marker.write(["a"])
    assert env.marker.active

    env.marker.clear()

    assert env.marker.active is False


async def test_many_files_are_cut_to_ten_and_names_are_escaped(env: Env) -> None:
    guard = env.guard(env.rules())
    watch = env.watch(guard)
    commands = env.workspace / ".agent" / "commands"
    commands.mkdir()
    for i in range(12):
        (commands / f"c{i:02}\x1b[31m.md").write_text("x")

    watch.check(Cause.background())

    [text] = env.warnings
    assert "and 3 more" in text  # 12 files + the folder = 13 paths
    assert "\x1b" not in text and "\\x1b" in text


async def test_a_failing_listener_does_not_stop_the_others_or_the_reaction(env: Env) -> None:
    rules = env.rules()
    guard = env.guard(rules)

    def boom(text: str) -> None:
        raise RuntimeError("ui")

    guard.listeners.insert(0, boom)
    watch = env.watch(guard)

    (env.workspace / ".agent" / "settings.json").write_text("{}")
    watch.check(Cause.background())

    assert len(env.warnings) == 1 and not _allowed(rules)


def test_the_marker_reads_back_its_time_and_files(env: Env) -> None:
    assert env.marker.read() is None

    env.marker.write(["~/.config/agent/settings.json", "~/.config/agent/mcp.json"])
    info = env.marker.read()

    assert info is not None
    assert info.files == ("~/.config/agent/settings.json", "~/.config/agent/mcp.json")
    assert info.at is not None and info.at.tzinfo is not None


def test_a_hostile_or_broken_marker_reads_as_set_with_nothing_known(env: Env) -> None:
    env.data.mkdir(parents=True)
    (env.data / "config-changed.json").write_text('{"at": 5, "files": ["ok", 3, "x\\u001b[31m"]}')

    info = env.marker.read()

    assert info is not None and info.at is None and info.files == ("ok", "x\x1b[31m")  # escaped by whoever shows it
    (env.data / "config-changed.json").write_text("not json")
    broken = env.marker.read()
    assert broken is not None and broken.at is None and broken.files == ()


def _global_rules(env: Env, allow: list[str]) -> Rules:
    env.config.mkdir(parents=True, exist_ok=True)
    (env.config / "settings.json").write_text(json.dumps({"permissions": {"allow": allow}}))
    return prepare(env.workspace, env.config / "settings.json", env.perm_store).rules


async def test_trust_global_clears_the_marker_without_a_warning_and_turns_global_rules_on(env: Env) -> None:
    env.marker.write(["~/.config/agent/settings.json"])
    rules = _global_rules(env, ["bash(ls)"])
    rules.disable_global(until_trusted=True)
    guard = env.guard(rules)
    watch = env.watch(guard)

    guard.trust_global()

    assert env.marker.exists() is False and rules.global_active and _allowed(rules)
    assert watch.check(Cause.background()) is None and env.warnings == []


async def test_trust_global_is_refused_and_keeps_the_marker_after_a_change_this_run(env: Env) -> None:
    env.marker.write(["~/.config/agent/settings.json"])
    rules = _global_rules(env, ["bash(ls)"])
    rules.disable_global(until_trusted=True)
    guard = env.guard(rules)
    watch = env.watch(guard)
    (env.workspace / ".agent" / "settings.json").write_text("{}")
    watch.check(Cause.background())  # a project revoke

    with pytest.raises(TrustRevoked):
        guard.trust_global()

    assert env.marker.exists() and not rules.global_active and not _allowed(rules)


async def test_a_global_change_seen_as_trust_global_opens_keeps_everything_off(env: Env) -> None:
    env.marker.write(["~/.config/agent/settings.json"])
    rules = _global_rules(env, ["bash(ls)"])
    rules.disable_global(until_trusted=True)
    guard = env.guard(rules)
    env.watch(guard)
    (env.config / "settings.json").write_text(json.dumps({"permissions": {"allow": ["bash(curl *)"]}}))

    with pytest.raises(TrustRevoked):
        guard.trust_global()

    assert env.marker.exists() and not rules.global_active
    assert len(env.warnings) == 1 and "settings.json" in env.warnings[0]


@pytest.mark.parametrize("how", ["folder", "unwritable"])
async def test_a_marker_that_cannot_be_cleared_gives_one_clear_line_and_keeps_everything_off(env: Env, how: str) -> None:
    rules = _global_rules(env, ["bash(ls)"])
    if how == "folder":
        env.marker.path.mkdir(parents=True)
    else:
        env.marker.write(["x"])
    rules.disable_global(until_trusted=True)
    guard = env.guard(rules)
    env.watch(guard)
    if how == "unwritable":
        env.data.chmod(0o500)
    try:
        with pytest.raises(OSError) as raised:
            guard.trust_global()
    finally:
        env.data.chmod(0o700)

    message = str(raised.value)
    assert message.startswith("could not clear the marker: E") and "\n" not in message
    assert message.endswith(f"; remove {env.marker.path} by hand")
    assert not rules.global_active and env.marker.exists()


def test_a_listener_that_comes_late_hears_what_was_said_before_it(env: Env) -> None:
    rules = env.rules()
    guard = ConfigGuard(env.workspace, rules, env.mcp, [env.perm_store, env.mcp_store], env.marker, home=env.tmp / "home")
    watch = env.watch(guard)
    (env.workspace / ".agent" / "settings.json").write_text("{}")
    watch.check(Cause.background())  # no listener yet (the app is not up)

    heard: list[str] = []
    guard.listen(heard.append)
    later: list[str] = []
    guard.listen(later.append)

    assert len(heard) == 1 and ".agent/settings.json" in heard[0]
    assert later == []  # the backlog goes to the first listener only


def test_a_warning_told_after_the_last_listener_left_is_kept_for_later(env: Env) -> None:
    """The app's transcript is gone at quit: what the quit checks see is kept and printed after the app."""
    rules = env.rules()
    guard = ConfigGuard(env.workspace, rules, env.mcp, [env.perm_store, env.mcp_store], env.marker, home=env.tmp / "home")
    watch = env.watch(guard)
    heard: list[str] = []
    guard.listen(heard.append)
    guard.unlisten(heard.append)
    (env.workspace / ".agent" / "settings.json").write_text("{}")
    watch.check(Cause.background())

    kept = guard.undelivered()

    assert heard == [] and len(kept) == 1 and ".agent/settings.json" in kept[0]
    assert guard.undelivered() == []  # handed over once


def test_a_warning_whose_every_listener_failed_is_kept(env: Env) -> None:
    rules = env.rules()
    guard = ConfigGuard(env.workspace, rules, env.mcp, [env.perm_store, env.mcp_store], env.marker, home=env.tmp / "home")
    watch = env.watch(guard)

    def gone(text: str) -> None:
        raise RuntimeError("the app is not running")

    guard.listen(gone)
    (env.workspace / ".agent" / "settings.json").write_text("{}")
    watch.check(Cause.background())

    assert len(guard.undelivered()) == 1


def test_a_delivered_warning_is_not_kept(env: Env) -> None:
    guard = env.guard(env.rules())
    watch = env.watch(guard)
    (env.workspace / ".agent" / "settings.json").write_text("{}")
    watch.check(Cause.background())

    assert len(env.warnings) == 1 and guard.undelivered() == []


def test_summary_says_no_changes_then_lists_each_change_and_the_marker(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    rules = env.rules()
    guard = env.guard(rules)
    watch = env.watch(guard)
    assert guard.summary() == ["Config watch: no changes this run"]

    (env.workspace / ".agent" / "settings.json").write_text("{}")
    change = watch.check(Cause.tool("bash", "./setup.sh"))
    env.config.mkdir(parents=True, exist_ok=True)
    (env.config / "mcp.json").write_text("{}")
    second = watch.check(Cause.hook("Stop"))

    lines = guard.summary()
    assert change is not None and second is not None
    stamp = time.strftime("%H:%M:%S", time.localtime(change.at))
    stamp2 = time.strftime("%H:%M:%S", time.localtime(second.at))
    assert lines[0] == "Config watch: 2 changes this run"
    assert lines[1] == f"  {stamp} bash (./setup.sh): .agent/settings.json"
    assert lines[2] == f"  {stamp2} hook Stop: {env.config / 'mcp.json'}"
    assert lines[3].startswith("global marker: set (since ")
    assert len(lines) == 4


def test_summary_lists_at_most_20_changes_then_a_count(env: Env) -> None:
    guard = env.guard(env.rules())
    watch = env.watch(guard)
    for i in range(25):
        (env.workspace / ".agent" / "settings.json").write_text(f'{{"n": {i}}}')
        watch.check(Cause.tool("bash", f"step {i}"))

    lines = guard.summary()

    assert lines[0] == "Config watch: 25 changes this run"
    assert len(lines) == 22 and lines[20].startswith("  ") and "step 19" in lines[20]
    assert lines[21] == "  and 5 more"


def test_the_marker_line_shows_without_any_change_this_run(env: Env) -> None:
    env.marker.write(["x"])
    guard = env.guard(env.rules())
    env.watch(guard)

    assert guard.summary()[0] == "Config watch: no changes this run"
    assert guard.summary()[1].startswith("global marker: set (since ")


def test_the_marker_round_trip_is_atomic_and_clear_is_idempotent(env: Env) -> None:
    assert env.marker.exists() is False

    env.marker.write(["a", "b"])
    assert env.marker.exists() and json.loads((env.data / "config-changed.json").read_text())["files"] == ["a", "b"]
    assert [p.name for p in env.data.iterdir()] == ["config-changed.json"]
    env.marker.clear()
    env.marker.clear()

    assert env.marker.exists() is False


def _hub(env: Env, monkeypatch: pytest.MonkeyPatch) -> tuple[LspHub, Path]:
    dump = env.tmp / "dump.jsonl"
    monkeypatch.setenv("FAKE_LSP_DUMP", str(dump))
    (env.tmp / "bin").mkdir(exist_ok=True)
    specs = tuple(
        ServerSpec(name, str(fake_lsp(env.tmp / "bin", name)), (), frozenset({".py"}), {}, source)
        for name, source in (("proj", "project"), ("glob", "global"))
    )
    hub = LspHub(LspSetup(True, False, specs, ()), env.workspace, dict(os.environ), wait=3, start_wait=5, max_lines=20)
    return hub, dump


async def _run_both(hub: LspHub, env: Env, dump: Path) -> tuple[int, int]:
    """The pids of the project server and the global one (the table order is the order they first appear)."""
    for text in ("x\n", "y\n"):
        await hub.block(env.workspace / "a.py", text, lambda e: None, 10_000)
    found = list(dict.fromkeys(json.loads(x)["pid"] for x in dump.read_text().splitlines()))
    assert len(found) == 2
    return found[0], found[1]


async def test_ac10_a_project_change_stops_the_project_lsp_servers_and_nothing_restarts(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub, dump = _hub(env, monkeypatch)
    guard = env.guard(env.rules(), lsp=hub)
    watch = env.watch(guard)
    proj_pid, glob_pid = await _run_both(hub, env, dump)

    (env.workspace / ".agent" / "settings.json").write_text("{}")
    watch.check(Cause.tool("bash", "./setup.sh"))
    await guard.settle()
    await assert_all_gone(proj_pid)  # right after the reaction, not at close()
    os.kill(glob_pid, 0)  # the global server is still alive

    assert ("proj", "project", "off", "config changed") in hub.listing()
    assert next(row for row in hub.listing() if row[0] == "glob")[2] == "running"
    assert env.warnings[0].endswith("project trust revoked — the next launch will ask; LSP servers stopped")
    await hub.block(env.workspace / "a.py", "z\n", lambda e: None, 10_000)
    assert len({json.loads(x)["pid"] for x in dump.read_text().splitlines()}) == 2  # nothing new started
    await hub.close()
    await assert_all_gone(glob_pid)


async def test_ac10_a_global_settings_change_stops_the_global_lsp_servers_only(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub, dump = _hub(env, monkeypatch)
    guard = env.guard(env.rules(), lsp=hub)
    watch = env.watch(guard)
    proj_pid, glob_pid = await _run_both(hub, env, dump)

    (env.config / "settings.json").write_text("{}")
    watch.check(Cause.background())
    await guard.settle()
    await assert_all_gone(glob_pid)  # right after the reaction, not at close()
    os.kill(proj_pid, 0)  # the project server is still alive

    states = {row[0]: row[2:] for row in hub.listing()}
    assert states["glob"] == ("off", "config changed") and states["proj"][0] == "running"
    assert env.warnings[0].endswith("; LSP servers stopped")
    await hub.close()
    await assert_all_gone(proj_pid)


async def test_a_failing_lsp_withdraw_does_not_break_the_reaction(env: Env) -> None:
    class Broken:
        def sources(self) -> dict[str, str]:
            return {"proj": "project"}

        def withdraw(self, names: Sequence[str], reason: str) -> None:
            raise RuntimeError("hub broken")

    rules = env.rules()
    guard = ConfigGuard(env.workspace, rules, env.mcp, [env.perm_store, env.mcp_store], env.marker, home=env.tmp / "home", lsp=Broken())
    guard.listeners.append(env.warnings.append)
    watch = env.watch(guard)

    (env.workspace / ".agent" / "settings.json").write_text("{}")
    watch.check(Cause.background())
    await guard.settle()

    assert not _allowed(rules)  # the revoke still happened
    assert "could not stop the LSP servers" in env.warnings[0]
