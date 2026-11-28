"""The config watch: a stat-first fingerprint of the agent's own files and the baseline logic around it."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest
from core.engine import configwatch
from core.engine.configwatch import (
    MAX_ENTRIES,
    Cause,
    Change,
    ConfigWatch,
    WatchedSet,
    fingerprint,
)


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    return {"XDG_CONFIG_HOME": str(tmp_path / "config"), "XDG_DATA_HOME": str(tmp_path / "data")}


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    (root / ".agent").mkdir(parents=True)
    return root


def _watch(ws: Path, env: dict[str, str], tmp_path: Path) -> tuple[ConfigWatch, list[Change]]:
    seen: list[Change] = []
    watched = WatchedSet.for_run(ws, env, tmp_path / "home")
    return ConfigWatch(watched, seen.append), seen


def test_for_run_names_the_project_global_trust_and_marker_files(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    watched = WatchedSet.for_run(ws, env, tmp_path / "home")

    roots = {(p, area, walk) for p, area, walk in watched.roots}
    config, data = tmp_path / "config" / "agent", tmp_path / "data" / "agent"
    assert (ws / ".agent" / "settings.json", "project", False) in roots
    assert (ws / ".agent" / "mcp.json", "project", False) in roots
    for name in ("commands", "skills", "agents"):
        assert (ws / ".agent" / name, "project", True) in roots
        assert (config / name, "global", True) in roots
    assert (config / "settings.json", "global", False) in roots
    assert (config / "mcp.json", "global", False) in roots
    assert (data / "permission-trust.json", "trust", False) in roots
    assert (data / "mcp-trust.json", "trust", False) in roots
    assert (data / "config-changed.json", "marker", False) in roots


def test_a_relative_xdg_folder_falls_back_to_the_home_folder(ws: Path, tmp_path: Path) -> None:
    home = tmp_path / "home"
    watched = WatchedSet.for_run(ws, {"XDG_CONFIG_HOME": "rel", "XDG_DATA_HOME": "rel"}, home)

    paths = {p for p, _, _ in watched.roots}
    assert home / ".config" / "agent" / "settings.json" in paths
    assert home / ".local" / "share" / "agent" / "permission-trust.json" in paths


def test_no_change_means_no_report(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    (ws / ".agent" / "settings.json").write_text("{}")
    watch, seen = _watch(ws, env, tmp_path)

    assert watch.check(Cause.background()) is None
    assert seen == []


def test_a_rewritten_file_is_one_change_naming_the_file_and_the_cause(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    settings = ws / ".agent" / "settings.json"
    settings.write_text("{}")
    watch, seen = _watch(ws, env, tmp_path)

    settings.write_text('{"permissions": {"allow": ["bash"]}}')
    change = watch.check(Cause.tool("bash", "./setup.sh"))

    assert change is not None and seen == [change]
    assert change.paths == (settings,)
    assert change.areas == frozenset({"project"})
    assert change.cause.text == "bash (./setup.sh)"


def test_the_same_change_is_reported_once_and_a_different_one_again(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    settings = ws / ".agent" / "settings.json"
    watch, seen = _watch(ws, env, tmp_path)

    settings.write_text("a")
    watch.check(Cause.background())
    assert watch.check(Cause.background()) is None
    settings.write_text("bb")
    watch.check(Cause.background())

    assert len(seen) == 2


def test_accept_takes_a_new_baseline_without_reporting(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    settings = ws / ".agent" / "settings.json"
    watch, seen = _watch(ws, env, tmp_path)

    settings.write_text("mine")
    assert watch.accept([settings]) is None

    assert watch.check(Cause.background()) is None
    assert seen == []


def test_own_write_reports_what_came_before_and_accepts_what_it_wrote(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    settings = ws / ".agent" / "settings.json"
    mcp = ws / ".agent" / "mcp.json"
    watch, seen = _watch(ws, env, tmp_path)

    mcp.write_text("sneaky")  # not ours: reported by the check before the own write
    with watch.own_write([settings]):
        settings.write_text("ours")

    assert [c.paths for c in seen] == [(mcp,)]
    assert watch.check(Cause.background()) is None


def test_a_content_free_touch_is_not_a_change(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    settings = ws / ".agent" / "settings.json"
    settings.write_text("same")
    watch, _ = _watch(ws, env, tmp_path)

    os.utime(settings, ns=(1, 1))

    assert watch.check(Cause.background()) is None


def test_a_file_that_appears_or_disappears_is_a_change(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    mcp = ws / ".agent" / "mcp.json"
    watch, seen = _watch(ws, env, tmp_path)

    mcp.write_text("{}")
    watch.check(Cause.background())
    mcp.unlink()
    watch.check(Cause.background())

    assert len(seen) == 2


def test_a_folder_created_by_a_command_is_a_change(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    watch, seen = _watch(ws, env, tmp_path)

    (ws / ".agent" / "commands").mkdir()
    (ws / ".agent" / "commands" / "x.md").write_text("hi")
    watch.check(Cause.background())

    assert seen and ws / ".agent" / "commands" / "x.md" in seen[0].paths


def test_a_file_changed_inside_a_watched_folder_is_a_change_with_its_area(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    skill = tmp_path / "config" / "agent" / "skills" / "s" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("one")
    watch, seen = _watch(ws, env, tmp_path)

    skill.write_text("two!")
    watch.check(Cause.background())

    assert seen[0].areas == frozenset({"global"})
    assert seen[0].paths == (skill,)


def test_a_trust_store_or_marker_change_has_its_own_area(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    data = tmp_path / "data" / "agent"
    data.mkdir(parents=True)
    watch, seen = _watch(ws, env, tmp_path)

    (data / "permission-trust.json").write_text(json.dumps({str(ws): "abc"}))
    (data / "config-changed.json").write_text("{}")
    watch.check(Cause.background())

    assert seen[0].areas == frozenset({"trust", "marker"})


def _store(tmp_path: Path) -> Path:
    data = tmp_path / "data" / "agent"
    data.mkdir(parents=True, exist_ok=True)
    return data / "permission-trust.json"


def test_another_workspaces_trust_entry_is_not_a_change(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    """Two instances in different folders share the stores: only this folder's entry is fingerprinted."""
    store = _store(tmp_path)
    watch, seen = _watch(ws, env, tmp_path)

    store.write_text(json.dumps({"/elsewhere": "abc"}))  # the store appears, holding another folder only
    assert watch.check(Cause.background()) is None
    store.write_text(json.dumps({"/elsewhere": "def", "/other": "x"}))
    assert watch.check(Cause.background()) is None
    store.unlink()

    assert watch.check(Cause.background()) is None and seen == []


def test_this_workspaces_trust_entry_appearing_changing_or_going_is_a_change(
    ws: Path, env: dict[str, str], tmp_path: Path
) -> None:
    store = _store(tmp_path)
    watch, seen = _watch(ws, env, tmp_path)

    store.write_text(json.dumps({"/elsewhere": "abc", str(ws): "one"}))
    assert watch.check(Cause.background()) is not None
    store.write_text(json.dumps({str(ws): "two"}))
    assert watch.check(Cause.background()) is not None
    store.unlink()

    assert watch.check(Cause.background()) is not None
    assert [c.paths for c in seen] == [(store,)] * 3


def test_an_unreadable_trust_store_is_its_own_state(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write_text(json.dumps({"/elsewhere": "abc"}))
    watch, seen = _watch(ws, env, tmp_path)

    store.write_text("not json")

    assert watch.check(Cause.background()) is not None and seen[0].paths == (store,)


def test_a_project_symlink_is_a_link_not_its_target_and_is_never_followed(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    """The project loaders refuse links, so only the link itself counts there (global ones are followed: below)."""
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "f.md"
    secret.write_text("one")
    commands = ws / ".agent" / "commands"
    commands.mkdir()
    (commands / "link").symlink_to(outside)
    watch, seen = _watch(ws, env, tmp_path)

    secret.write_text("changed content")  # behind the link: not watched
    assert watch.check(Cause.background()) is None
    (commands / "link").unlink()
    (commands / "link").symlink_to(tmp_path)  # the link itself moved
    watch.check(Cause.background())

    assert [c.paths for c in seen] == [(commands / "link",)]


def test_a_settings_file_that_is_a_symlink_is_fingerprinted_as_a_link(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    target = tmp_path / "t.json"
    target.write_text("a")
    (ws / ".agent" / "settings.json").symlink_to(target)
    watch, seen = _watch(ws, env, tmp_path)

    target.write_text("bbbb")

    assert watch.check(Cause.background()) is None and seen == []


def test_an_unreadable_file_gets_an_error_entry_and_a_change_when_it_becomes_readable(
    ws: Path, env: dict[str, str], tmp_path: Path
) -> None:
    settings = ws / ".agent" / "settings.json"
    settings.write_text("a")
    settings.chmod(0)
    try:
        watch, seen = _watch(ws, env, tmp_path)
        entries = fingerprint(WatchedSet.for_run(ws, env, tmp_path / "home"), {})
        assert entries[settings].kind == "error"
        assert entries[settings].detail == "EACCES"
        assert watch.check(Cause.background()) is None
    finally:
        settings.chmod(0o600)
    watch.check(Cause.background())

    assert len(seen) == 1


def test_a_removed_folder_and_a_file_that_vanishes_never_raise(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    (ws / ".agent" / "commands").mkdir()
    (ws / ".agent" / "commands" / "a.md").write_text("x")
    watch, seen = _watch(ws, env, tmp_path)

    (ws / ".agent" / "commands" / "a.md").unlink()
    (ws / ".agent" / "commands").rmdir()
    (ws / ".agent").rmdir()
    change = watch.check(Cause.background())

    assert change is not None and seen == [change]


def test_a_walk_past_the_bound_records_an_overflow_whose_count_changes(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    commands = ws / ".agent" / "commands"
    commands.mkdir()
    for i in range(3000):
        (commands / f"c{i:04}.md").write_text("x")
    watch, seen = _watch(ws, env, tmp_path)
    watched = WatchedSet.for_run(ws, env, tmp_path / "home")

    entries = fingerprint(watched, {})
    kept = [p for p in entries if commands in p.parents]
    assert len(kept) <= MAX_ENTRIES + 1
    assert any(e.kind == "overflow" for e in entries.values())
    assert watch.check(Cause.background()) is None
    (commands / "extra.md").write_text("x")  # beyond the bound: still a change
    watch.check(Cause.background())

    assert len(seen) == 1


def test_git_and_node_modules_folders_inside_a_walk_are_skipped(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    """The loaders never read them, and they can be huge."""
    skills = ws / ".agent" / "skills"
    for folder in (skills / ".git", skills / "s" / "node_modules", skills / "s" / ".git"):
        folder.mkdir(parents=True)
        (folder / "f").write_text("x")
    watch, seen = _watch(ws, env, tmp_path)

    for folder in (skills / ".git", skills / "s" / "node_modules", skills / "s" / ".git"):
        (folder / "f").write_text("changed")
        (folder / "new").write_text("x")

    assert watch.check(Cause.background()) is None and seen == []
    entries = fingerprint(WatchedSet.for_run(ws, env, tmp_path / "home"), {})
    assert not any(".git" in p.parts or "node_modules" in p.parts for p in entries)


def test_a_skill_named_node_modules_is_still_watched(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    """A skill folder may be called that (the loader reads it), so only deeper ones are skipped."""
    skill = ws / ".agent" / "skills" / "node_modules"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\ndescription: d\n---\nx")
    watch, _ = _watch(ws, env, tmp_path)

    (skill / "SKILL.md").write_text("---\ndescription: d\n---\nevil")

    assert watch.check(Cause.background()) is not None


def test_a_deep_tree_is_cut_at_six_levels(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    deep = ws / ".agent" / "skills"
    for i in range(10):
        deep = deep / f"d{i}"
    deep.mkdir(parents=True)
    (deep / "f.md").write_text("x")
    watched = WatchedSet.for_run(ws, env, tmp_path / "home")

    entries = fingerprint(watched, {})

    assert deep / "f.md" not in entries
    assert any(e.kind == "overflow" for e in entries.values())


def test_a_name_with_control_characters_gives_an_escaped_cause_and_a_change(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    commands = ws / ".agent" / "commands"
    commands.mkdir()
    watch, seen = _watch(ws, env, tmp_path)

    (commands / "a\x1b[31m\nb.md").write_text("x")
    watch.check(Cause.tool("bash", "echo\x1b[2J\n" + "x" * 200))

    assert len(seen) == 1
    assert "\x1b" not in seen[0].cause.text and "\n" not in seen[0].cause.text
    assert len(seen[0].cause.text) < 120


def test_cause_texts(ws: Path) -> None:
    assert Cause.tool("term_start", None).text == "term_start"
    assert Cause.hook("PreToolUse").text == "hook PreToolUse"
    assert Cause.background().text == "a background process or terminal (during the turn)"


def test_a_failing_listener_never_breaks_a_check(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    def boom(change: Change) -> None:
        raise RuntimeError("ui gone")

    watch = ConfigWatch(WatchedSet.for_run(ws, env, tmp_path / "home"), boom)
    (ws / ".agent" / "settings.json").write_text("x")

    assert watch.check(Cause.background()) is not None


def test_a_same_size_rewrite_that_restores_the_mtime_is_a_change(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    settings = ws / ".agent" / "settings.json"
    settings.write_text('{"permissions":{"allow":["bash(ls *)"]}}')
    watch, _ = _watch(ws, env, tmp_path)
    before = os.stat(settings)

    settings.write_text('{"permissions":{"allow":["bash(** *)"]}}')
    os.utime(settings, ns=(before.st_atime_ns, before.st_mtime_ns))

    assert settings.stat().st_size == before.st_size
    assert watch.check(Cause.background()) is not None


def test_for_run_also_watches_both_workflow_folders(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    roots = set(WatchedSet.for_run(ws, env, tmp_path / "home").roots)

    assert (ws / ".agent" / "workflows", "project", True) in roots
    assert (tmp_path / "config" / "agent" / "workflows", "global", True) in roots


def test_a_global_settings_file_that_is_a_symlink_is_followed(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    dotfiles = tmp_path / "dotfiles"
    dotfiles.mkdir()
    (dotfiles / "settings.json").write_text("{}")
    config = tmp_path / "config" / "agent"
    config.mkdir(parents=True)
    (config / "settings.json").symlink_to(dotfiles / "settings.json")
    watch, seen = _watch(ws, env, tmp_path)

    (dotfiles / "settings.json").write_text('{"permissions": {"allow": ["bash"]}}')

    assert watch.check(Cause.background()) is not None
    assert seen[0].paths == (config / "settings.json",)


def test_a_trust_store_that_is_a_symlink_is_followed(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    real = tmp_path / "real-trust.json"
    real.write_text("{}")
    data = tmp_path / "data" / "agent"
    data.mkdir(parents=True)
    (data / "permission-trust.json").symlink_to(real)
    watch, _ = _watch(ws, env, tmp_path)

    real.write_text(json.dumps({str(ws): "abc"}))

    assert watch.check(Cause.background()) is not None


def test_a_global_folder_that_is_a_symlink_is_walked(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    dotfiles = tmp_path / "dotfiles"
    dotfiles.mkdir()
    config = tmp_path / "config" / "agent"
    config.mkdir(parents=True)
    (config / "skills").symlink_to(dotfiles)
    watch, seen = _watch(ws, env, tmp_path)

    (dotfiles / "evil.md").write_text("x")

    assert watch.check(Cause.background()) is not None
    assert config / "skills" / "evil.md" in seen[0].paths


def test_a_link_inside_a_global_walk_is_followed(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text("one")
    commands = tmp_path / "config" / "agent" / "commands"
    commands.mkdir(parents=True)
    (commands / "x.md").symlink_to(outside)
    watch, _ = _watch(ws, env, tmp_path)

    outside.write_text("two!")

    assert watch.check(Cause.background()) is not None


def test_a_link_loop_in_a_global_walk_is_an_error_entry_and_never_raises(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    skills = tmp_path / "config" / "agent" / "skills"
    (skills / "s").mkdir(parents=True)
    (skills / "s" / "up").symlink_to(skills)  # a folder link back to an ancestor
    (skills / "self").symlink_to(skills / "self")  # a link to itself
    watched = WatchedSet.for_run(ws, env, tmp_path / "home")

    entries = fingerprint(watched, {})

    assert entries[skills / "self"].kind == "error"
    assert entries[skills / "s" / "up"].kind == "error"
    watch, seen = _watch(ws, env, tmp_path)
    assert watch.check(Cause.background()) is None and seen == []


def test_accept_adopts_only_the_named_paths_and_reports_the_rest_as_background(
    ws: Path, env: dict[str, str], tmp_path: Path
) -> None:
    settings = ws / ".agent" / "settings.json"
    mcp = ws / ".agent" / "mcp.json"
    watch, seen = _watch(ws, env, tmp_path)

    settings.write_text("mine")
    mcp.write_text("a child's")
    change = watch.accept([settings])

    assert change is not None and seen == [change]
    assert change.paths == (mcp,)
    assert change.cause == Cause.background()
    assert watch.check(Cause.background()) is None


def test_accept_of_a_file_in_a_new_folder_also_adopts_the_folder(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    command = ws / ".agent" / "commands" / "x.md"
    watch, seen = _watch(ws, env, tmp_path)

    command.parent.mkdir()
    command.write_text("mine")

    assert watch.accept([command]) is None and seen == []


def test_accept_of_one_file_in_a_walked_folder_keeps_the_others_reported(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    commands = ws / ".agent" / "commands"
    commands.mkdir()
    watch, seen = _watch(ws, env, tmp_path)

    (commands / "mine.md").write_text("x")
    (commands / "theirs.md").write_text("y")
    watch.accept([commands / "mine.md"])

    assert [c.paths for c in seen] == [(commands / "theirs.md",)]


def test_a_foreign_write_inside_own_write_is_reported(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    settings = ws / ".agent" / "settings.json"
    global_settings = tmp_path / "config" / "agent" / "settings.json"
    global_settings.parent.mkdir(parents=True)
    watch, seen = _watch(ws, env, tmp_path)

    with watch.own_write([settings]):
        settings.write_text("ours")
        global_settings.write_text("a child's")

    assert [c.paths for c in seen] == [(global_settings,)]


def test_watches_names_watched_paths_also_through_a_resolved_parent(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    link = tmp_path / "wslink"
    link.symlink_to(ws)
    watch, _ = _watch(link, env, tmp_path)

    assert watch.watches(ws / ".agent" / "settings.json")
    assert watch.watches(link / ".agent" / "commands" / "a" / "b.md")
    assert not watch.watches(ws / "README.md")


def test_an_edit_or_a_rename_beyond_the_bound_changes_the_overflow(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    commands = ws / ".agent" / "commands"
    commands.mkdir()
    for i in range(MAX_ENTRIES + 20):
        (commands / f"c{i:05}.md").write_text("x")
    last = commands / f"c{MAX_ENTRIES + 19:05}.md"
    watch, seen = _watch(ws, env, tmp_path)

    last.write_text("longer content")
    watch.check(Cause.background())
    last.rename(commands / "zz-renamed.md")
    watch.check(Cause.background())

    assert len(seen) == 2


def test_the_overflow_scan_stops_at_its_bound_and_records_a_count(
    ws: Path, env: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(configwatch, "MAX_ENTRIES", 5)
    monkeypatch.setattr(configwatch, "OVERFLOW_SCAN", 20)
    commands = ws / ".agent" / "commands"
    commands.mkdir()
    for i in range(100):
        (commands / f"c{i:03}.md").write_text("x")

    entries = fingerprint(WatchedSet.for_run(ws, env, tmp_path / "home"), {})

    [overflow] = [e for e in entries.values() if e.kind == "overflow"]
    assert overflow.detail.startswith("20+")


def test_an_unlistable_folder_whose_files_change_behind_a_chmod_is_a_change(
    ws: Path, env: dict[str, str], tmp_path: Path
) -> None:
    agent = ws / ".agent"
    (agent / "settings.json").write_text("a")
    agent.chmod(0)
    try:
        watch, seen = _watch(ws, env, tmp_path)
        assert fingerprint(WatchedSet.for_run(ws, env, tmp_path / "home"), {})[agent / "settings.json"].kind == "error"
        time.sleep(0.01)
        agent.chmod(0o700)
        (agent / "settings.json").write_text("b")
        agent.chmod(0)
        watch.check(Cause.background())
    finally:
        agent.chmod(0o700)

    assert len(seen) == 1


def test_an_unreadable_file_rewritten_between_two_chmods_is_a_change(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    settings = ws / ".agent" / "settings.json"
    settings.write_text("a")
    settings.chmod(0)
    try:
        watch, seen = _watch(ws, env, tmp_path)
        time.sleep(0.01)
        settings.chmod(0o600)
        settings.write_text("b")
        settings.chmod(0)
        watch.check(Cause.background())
    finally:
        settings.chmod(0o600)

    assert len(seen) == 1


def test_past_the_hash_budget_a_file_whose_stat_changed_counts_as_changed(
    ws: Path, env: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = ws / ".agent" / "commands"
    commands.mkdir()
    for name in ("a.md", "b.md"):
        (commands / name).write_text("x" * 100)
    watch, _ = _watch(ws, env, tmp_path)
    monkeypatch.setattr(configwatch, "MAX_CHECK_HASH_BYTES", 150)

    for name in ("a.md", "b.md"):
        os.utime(commands / name, ns=(5, 5))  # same bytes: only the stat moved
    change = watch.check(Cause.background())

    assert change is not None and len(change.paths) == 1  # one was hashed (same bytes), the other could not be


def test_concurrent_checks_report_one_change_once(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    watch, seen = _watch(ws, env, tmp_path)
    (ws / ".agent" / "settings.json").write_text("x")

    threads = [threading.Thread(target=watch.check, args=(Cause.background(),)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(seen) == 1


def test_an_unchanged_check_of_200_files_takes_at_most_5_ms(ws: Path, env: dict[str, str], tmp_path: Path) -> None:
    commands = ws / ".agent" / "commands"
    commands.mkdir()
    for i in range(200):
        (commands / f"c{i}.md").write_text("x" * 100)
    watch, _ = _watch(ws, env, tmp_path)
    watch.check(Cause.background())

    # CPU time of this process, best of 20: other processes on a loaded machine do not count against the check
    times = []
    for _ in range(20):
        start = time.process_time()
        watch.check(Cause.background())
        times.append(time.process_time() - start)

    assert min(times) <= 0.005
