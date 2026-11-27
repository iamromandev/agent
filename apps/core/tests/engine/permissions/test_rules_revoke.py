"""`Rules.revoke_project` and `disable_global`: the switches the config watch flips for the rest of a run."""

from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
from core.engine.configwatch import TrustRevoked
from core.engine.hooks.setup import prepare as prepare_hooks
from core.engine.mcp.trust import TrustStore
from core.engine.permissions.config import project_settings_path
from core.engine.permissions.setup import PermissionSetup, Rules, prepare
from core.testing.fakes import FakeRunner


class Launch:
    def __init__(self, tmp_path: Path) -> None:
        self.workspace = tmp_path / "ws"
        self.workspace.mkdir()
        self.global_path = tmp_path / "config" / "settings.json"
        self.store = TrustStore(tmp_path / "data" / "permission-trust.json")
        self.project_path = project_settings_path(self.workspace)

    def write(self, path: Path, allow: list[str] | None = None, deny: list[str] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        permissions = {k: v for k, v in (("allow", allow), ("deny", deny)) if v is not None}
        path.write_text(json.dumps({"permissions": permissions}))

    def rules(self) -> Rules:
        return prepare(self.workspace, self.global_path, self.store).rules


@pytest.fixture
def launch(tmp_path: Path) -> Launch:
    return Launch(tmp_path)


def _allowed(rules: Rules, command: str) -> bool:
    return rules.allow("bash", {"command": command}, None) is not None


def _denied(rules: Rules, command: str) -> bool:
    return rules.deny("bash", {"command": command}, None) is not None


def test_revoke_project_stops_trusted_project_allow_rules(launch: Launch) -> None:
    launch.write(launch.project_path, allow=["bash(ls)"])
    rules = launch.rules()
    rules.trust_project()
    assert _allowed(rules, "ls")

    rules.revoke_project()

    assert not _allowed(rules, "ls")
    assert rules.project_trusted is False


def test_revoke_project_also_stops_rules_added_with_always_here_in_this_run(launch: Launch) -> None:
    rules = launch.rules()
    rules.save_project("bash(make)")
    assert _allowed(rules, "make")

    rules.revoke_project()

    assert not _allowed(rules, "make")


def test_revoke_project_clears_a_trust_given_for_the_run_only(launch: Launch) -> None:
    launch.write(launch.project_path, allow=["bash(ls)"])
    rules = launch.rules()
    rules.trust_for_run()

    rules.revoke_project()

    assert not _allowed(rules, "ls") and rules.project_trusted is False


def test_a_revoked_run_cannot_be_trusted_again_by_a_late_trust_call(launch: Launch) -> None:
    launch.write(launch.project_path, allow=["bash(ls)"])
    rules = launch.rules()
    rules.revoke_project()

    rules.trust_for_run()
    with pytest.raises(TrustRevoked):
        rules.trust_project()

    assert not _allowed(rules, "ls")
    assert not launch.store._read()


class RevokingWrite:
    """An `own_write` whose opening check sees a change (as the config watch would) and revokes; records the paths."""

    def __init__(self, launch: Launch, revoke_on: int = 1) -> None:
        self.launch = launch
        self.rules: Rules | None = None
        self.calls: list[tuple[Path, ...]] = []
        self.revoke_on = revoke_on

    @contextmanager
    def __call__(self, paths: Sequence[Path]) -> Generator[None]:
        self.calls.append(tuple(paths))
        if len(self.calls) == self.revoke_on and self.rules is not None:
            self.rules.revoke_project()
            self.launch.store.forget(self.launch.workspace)
        yield


def _with(launch: Launch, own: RevokingWrite) -> Rules:
    rules = prepare(launch.workspace, launch.global_path, launch.store, own_write=own).rules
    own.rules = rules
    return rules


def test_a_revoke_seen_when_the_model_key_save_opens_refuses_it_and_trusts_nothing(launch: Launch) -> None:
    """poc5: a background change is seen by the check that opens `/model`'s own write."""
    launch.write(launch.project_path, allow=["bash(make *)"])
    launch.store.trust(launch.workspace, launch.project_path.read_bytes())
    before = launch.project_path.read_bytes()
    rules = _with(launch, RevokingWrite(launch))
    assert rules.project_trusted

    with pytest.raises(TrustRevoked):
        rules.save_project_key("model", "m1")

    assert rules.project_trusted is False
    assert launch.store._read() == {}  # the next launch asks
    assert launch.project_path.read_bytes() == before


def test_a_revoke_seen_when_always_here_opens_refuses_the_rule(launch: Launch) -> None:
    rules = _with(launch, RevokingWrite(launch))

    with pytest.raises(TrustRevoked):
        rules.save_project("bash(make)")

    assert not _allowed(rules, "make") and launch.store._read() == {}


def test_a_revoke_seen_when_the_vouch_opens_leaves_the_store_alone(launch: Launch) -> None:
    rules = _with(launch, RevokingWrite(launch, revoke_on=2))  # the rule's write, then the trust write

    with pytest.raises(TrustRevoked):
        rules.save_project("bash(make)")

    assert not _allowed(rules, "make") and rules.project_trusted is False
    assert launch.store._read() == {}


def test_a_revoke_seen_when_trust_project_opens_writes_no_trust(launch: Launch) -> None:
    launch.write(launch.project_path, allow=["bash(ls)"])
    rules = _with(launch, RevokingWrite(launch))

    with pytest.raises(TrustRevoked):
        rules.trust_project()

    assert not _allowed(rules, "ls") and launch.store._read() == {}


def test_own_writes_name_the_files_they_write(launch: Launch) -> None:
    launch.write(launch.project_path, allow=["bash(ls)"])
    own = RevokingWrite(launch, revoke_on=0)
    rules = _with(launch, own)

    rules.trust_project()
    rules.save_project("bash(make)")
    rules.save_project_key("model", "m")

    store, settings = launch.store.path, launch.project_path
    assert own.calls == [(store,), (settings, store), (store,), (settings,), (store,)]


def test_a_revoked_run_refuses_always_here_and_the_key_save(launch: Launch) -> None:
    rules = launch.rules()
    rules.revoke_project()

    assert rules.can_save is False
    with pytest.raises(OSError):
        rules.save_project("bash(make)")
    with pytest.raises(OSError):
        rules.save_project_key("model", "x")
    assert not launch.project_path.exists()


def test_deny_rules_always_apply_even_after_both_switches(launch: Launch) -> None:
    launch.write(launch.global_path, deny=["bash(rm*)"])
    launch.write(launch.project_path, deny=["bash(curl*)"])
    rules = launch.rules()

    rules.revoke_project()
    rules.disable_global()

    assert _denied(rules, "rm -rf x") and _denied(rules, "curl x")


def test_disable_global_stops_global_allow_rules_but_not_project_ones(launch: Launch) -> None:
    launch.write(launch.global_path, allow=["bash(ls)"])
    launch.write(launch.project_path, allow=["bash(pwd)"])
    rules = launch.rules()
    rules.trust_project()
    assert rules.global_active is True

    rules.disable_global()

    assert rules.global_active is False
    assert not _allowed(rules, "ls") and _allowed(rules, "pwd")


def test_global_rules_held_at_launch_come_back_with_enable_global(launch: Launch) -> None:
    launch.write(launch.global_path, allow=["bash(ls)"])
    rules = launch.rules()
    rules.disable_global(until_trusted=True)  # the launch marker: off until the user trusts them
    assert not _allowed(rules, "ls")

    rules.enable_global()

    assert rules.global_active is True and _allowed(rules, "ls")


def test_enable_global_is_refused_after_the_watch_switched_global_off(launch: Launch) -> None:
    launch.write(launch.global_path, allow=["bash(ls)"])
    rules = launch.rules()
    rules.disable_global(until_trusted=True)
    rules.disable_global()  # a change seen during the run

    with pytest.raises(TrustRevoked):
        rules.enable_global()

    assert rules.global_active is False and not _allowed(rules, "ls")


def test_enable_global_is_refused_after_a_project_revoke(launch: Launch) -> None:
    launch.write(launch.global_path, allow=["bash(ls)"])
    rules = launch.rules()
    rules.disable_global(until_trusted=True)
    rules.revoke_project()

    with pytest.raises(TrustRevoked):
        rules.enable_global()

    assert not _allowed(rules, "ls")


def test_global_allow_lists_the_global_files_allow_rules(launch: Launch) -> None:
    launch.write(launch.global_path, allow=["bash(ls)", "read"])
    launch.write(launch.project_path, allow=["bash(pwd)"])

    assert [rule.text for rule in launch.rules().global_allow] == ["bash(ls)", "read"]


def _global_settings(launch: Launch) -> PermissionSetup:
    launch.global_path.parent.mkdir(parents=True, exist_ok=True)
    launch.global_path.write_text(
        json.dumps({"permissions": {"allow": ["bash(ls)"], "deny": ["bash(rm)"]}, "hooks": {"Stop": [{"command": "bye"}]}})
    )
    return prepare(launch.workspace, launch.global_path, launch.store)


@pytest.mark.parametrize(
    ("until_trusted", "reason"), [(False, "inactive: global config changed"), (True, "inactive: held until trusted")]
)
def test_permissions_and_hooks_mark_the_global_entries_inactive_while_global_is_off(
    launch: Launch, until_trusted: bool, reason: str
) -> None:
    setup = _global_settings(launch)
    hooks = prepare_hooks(setup, launch.workspace, FakeRunner())
    setup.rules.disable_global(until_trusted=until_trusted)

    assert f"allow bash(ls) (global) {reason}" in setup.rules.listing()
    assert "deny bash(rm) (global)" in setup.rules.listing()  # deny rules always apply
    assert hooks.listing() == [f"Stop [*] (global, 60s, {reason}) bye"]


def test_permissions_and_hooks_show_global_entries_as_active_by_default(launch: Launch) -> None:
    setup = _global_settings(launch)
    hooks = prepare_hooks(setup, launch.workspace, FakeRunner())

    assert "allow bash(ls) (global)" in setup.rules.listing()
    assert hooks.listing() == ["Stop [*] (global, 60s) bye"]
