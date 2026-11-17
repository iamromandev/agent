"""CheckpointStore.snapshot with the real git: contents, exclusions, isolation from the user's repository."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
from src.engine.checkpoints.git import GitError, GitRunner
from src.engine.checkpoints.store import FILE_CAP, CheckpointStore
from src.engine.guardrails.setup import Guardrails

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root.resolve()


@pytest.fixture
def data(tmp_path: Path) -> Path:
    return tmp_path / "data" / "agent" / "checkpoints"


def make_store(workspace: Path, data: Path, env: dict[str, str] | None = None, **kw: Any) -> CheckpointStore:
    return CheckpointStore(workspace, data, Guardrails.off(workspace), env or dict(os.environ), **kw)


def shadow(store: CheckpointStore, *args: str) -> str:
    done = subprocess.run(
        ["git", "--git-dir", str(store.folder), *args], capture_output=True, text=True, check=True, env={"PATH": os.environ["PATH"]}
    )
    return done.stdout


def files(store: CheckpointStore, commit: str) -> list[str]:
    return shadow(store, "ls-tree", "-r", "--name-only", commit).split()


async def test_a_snapshot_holds_every_file_and_symlink(workspace: Path, data: Path) -> None:
    (workspace / "a.py").write_text("a")
    (workspace / "sub").mkdir()
    (workspace / "sub" / "b.py").write_text("b")
    (workspace / "link").symlink_to("a.py")
    store = make_store(workspace, data)

    snap = await store.snapshot()

    assert snap is not None
    assert files(store, snap.commit) == ["a.py", "link", "sub/b.py"]
    assert shadow(store, "cat-file", "-p", f"{snap.commit}:a.py") == "a"
    assert shadow(store, "ls-tree", snap.commit, "link").startswith("120000")


async def test_gitignore_env_and_large_files_are_left_out(workspace: Path, data: Path) -> None:
    (workspace / ".gitignore").write_text("build/\n")
    (workspace / "build").mkdir()
    (workspace / "build" / "out.o").write_text("x")
    (workspace / ".env").write_text("SECRET=1")
    (workspace / "keep.txt").write_text("k")
    with (workspace / "big.bin").open("wb") as fh:
        fh.truncate(FILE_CAP + 1)
    store = make_store(workspace, data)

    snap = await store.snapshot()

    assert snap is not None
    assert files(store, snap.commit) == [".gitignore", "keep.txt"]
    assert snap.skipped_large == ("big.bin",)


async def test_agent_config_files_are_left_out(workspace: Path, data: Path) -> None:
    agent = workspace / ".agent"
    (agent / "commands").mkdir(parents=True)
    (agent / "skills" / "s").mkdir(parents=True)
    for name in ("settings.json", "mcp.json", "commands/c.md", "skills/s/SKILL.md", "notes.txt"):
        (agent / name).write_text("x")
    store = make_store(workspace, data)

    snap = await store.snapshot()

    assert snap is not None
    assert files(store, snap.commit) == [".agent/notes.txt"]


async def test_an_unchanged_workspace_makes_no_new_commit(workspace: Path, data: Path) -> None:
    (workspace / "a.py").write_text("a")
    store = make_store(workspace, data)
    first = await store.snapshot()

    second = await store.snapshot()

    assert first is not None and second is not None
    assert second.commit == first.commit


async def test_a_change_makes_a_child_commit_and_a_deletion_leaves_the_tree(workspace: Path, data: Path) -> None:
    (workspace / "a.py").write_text("a")
    (workspace / "b.py").write_text("b")
    store = make_store(workspace, data)
    first = await store.snapshot()
    (workspace / "a.py").write_text("a2")
    (workspace / "b.py").unlink()

    second = await store.snapshot()

    assert first is not None and second is not None
    assert second.commit != first.commit
    assert files(store, second.commit) == ["a.py"]
    assert shadow(store, "rev-parse", f"{second.commit}^").strip() == first.commit


async def test_a_nested_repository_is_snapshotted_as_plain_files_without_its_git(workspace: Path, data: Path) -> None:
    nested = workspace / "vendor" / "lib"
    nested.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(nested)], check=True)
    (nested / "x.py").write_text("x")
    store = make_store(workspace, data)

    snap = await store.snapshot()

    assert snap is not None
    assert files(store, snap.commit) == ["vendor/lib/x.py"]


async def test_the_users_repository_is_untouched_and_no_hook_or_filter_runs(
    workspace: Path, data: Path, tmp_path: Path
) -> None:
    run = lambda *a: subprocess.run(["git", *a], cwd=workspace, check=True, capture_output=True, text=True)  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "u@x")
    run("config", "user.name", "u")
    (workspace / "a.py").write_text("a")
    run("add", "a.py")
    run("commit", "-qm", "one")
    (workspace / "a.py").write_text("staged")
    run("add", "a.py")
    marker, filter_marker = tmp_path / "hook-ran", tmp_path / "filter-ran"
    hook = workspace / ".git" / "hooks" / "pre-commit"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)
    (workspace / ".gitattributes").write_text("* filter=evil\n")
    run("config", "filter.evil.clean", f"touch {filter_marker}; cat")
    run("config", "core.fsmonitor", f"touch {filter_marker}; echo")
    home = tmp_path / "home"
    (home / ".config" / "git").mkdir(parents=True)
    hooks = tmp_path / "evil-hooks"
    hooks.mkdir()
    (hooks / "pre-commit").write_text(f"#!/bin/sh\ntouch {marker}\n")
    (hooks / "pre-commit").chmod(0o755)
    config = (
        f'[filter "evil"]\n\tclean = touch {filter_marker}; cat\n\tsmudge = cat\n'
        f"[core]\n\tfsmonitor = touch {filter_marker}; echo\n\thooksPath = {hooks}\n"
        "[user]\n\tsigningkey = ABC\n[commit]\n\tgpgSign = true\n"
    )
    (home / ".gitconfig").write_text(config)
    (home / ".config" / "git" / "config").write_text(config)
    head = run("rev-parse", "HEAD").stdout
    index = hashlib.sha256((workspace / ".git" / "index").read_bytes()).hexdigest()
    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "GIT_DIR": str(workspace / ".git"),
        "GIT_INDEX_FILE": "/nonexistent",
    }
    control = subprocess.run(
        ["git", "config", "--global", "filter.evil.clean"], env=env, capture_output=True, text=True, check=True
    )
    assert filter_marker.name in control.stdout  # git, run with this HOME, does read the evil config
    store = make_store(workspace, data, env)

    snap = await store.snapshot()

    assert snap is not None
    assert "a.py" in files(store, snap.commit)
    assert run("rev-parse", "HEAD").stdout == head
    assert hashlib.sha256((workspace / ".git" / "index").read_bytes()).hexdigest() == index
    assert not marker.exists()
    assert not filter_marker.exists()
    attrs = subprocess.run(
        ["git", "--git-dir", str(store.folder), "--work-tree", str(workspace), "check-attr", "filter", "--", "a.py"],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": os.environ["PATH"]},
    )
    assert attrs.stdout.strip() == "a.py: filter: unset"  # the workspace's .gitattributes cannot name a filter


async def test_the_shadow_folder_is_private_and_removed_on_close(workspace: Path, data: Path) -> None:
    (workspace / "a.py").write_text("a")
    store = make_store(workspace, data)
    await store.snapshot()

    assert stat.S_IMODE(store.folder.stat().st_mode) == 0o700
    assert str(os.getpid()) in store.folder.name
    store.close()
    assert not store.folder.exists()
    assert await store.snapshot() is None


async def test_nothing_is_created_before_the_first_snapshot(workspace: Path, data: Path) -> None:
    make_store(workspace, data)

    assert not data.exists()


async def test_a_failing_git_turns_the_store_off_with_one_warning(workspace: Path, data: Path) -> None:
    store = make_store(workspace, data, {"PATH": str(workspace)})

    assert await store.snapshot() is None

    assert not store.enabled
    assert store.off_reason is not None
    warning = store.take_warning()
    assert warning is not None and warning.startswith("checkpoints off: ")
    assert warning.endswith("/undo restores write/edit changes only")
    assert store.take_warning() is None


async def test_a_slow_snapshot_turns_the_store_off_and_kills_the_git_group(
    workspace: Path, data: Path, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "child.pid"
    script = bin_dir / "git"
    script.write_text(f"#!/bin/sh\nsleep 30 &\necho $! > {marker}\nwait\n")
    script.chmod(0o755)
    store = make_store(workspace, data, {"PATH": f"{bin_dir}:/bin:/usr/bin"}, timeout=0.5)

    assert await store.snapshot() is None

    assert store.off_reason == "snapshot took longer than 0.5 s"
    assert store.take_warning() == "checkpoints off: snapshot took longer than 0.5 s; /undo restores write/edit changes only"
    pid = int(marker.read_text())
    await asyncio.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_a_path_under_a_folder_swapped_for_a_symlink_is_left_out_and_the_store_stays_on(
    workspace: Path, data: Path, tmp_path: Path
) -> None:
    (workspace / "sub").mkdir()
    (workspace / "sub" / "a").write_text("inside")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "a").write_text("outside")
    store = make_store(workspace, data)
    assert await store.snapshot() is not None
    shutil.rmtree(workspace / "sub")
    (workspace / "sub").symlink_to("../outside")

    snap = await store.snapshot()

    assert snap is not None
    assert store.enabled
    assert files(store, snap.commit) == ["sub"]
    assert shadow(store, "ls-tree", snap.commit, "sub").startswith("120000")


async def test_a_file_that_vanishes_after_it_was_classified_does_not_turn_the_store_off(
    workspace: Path, data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "a.py").write_text("a")
    (workspace / "b.py").write_text("b")
    store = make_store(workspace, data)
    classify = store._classify

    def vanishing(rels: list[str], gone: set[str]) -> tuple[list[str], list[str], list[str]]:
        result = classify(rels, gone)
        (workspace / "a.py").unlink(missing_ok=True)
        return result

    monkeypatch.setattr(store, "_classify", vanishing)

    snap = await store.snapshot()

    assert snap is not None
    assert store.enabled
    assert files(store, snap.commit) == ["b.py"]


async def test_a_failing_update_index_is_classified_again_and_retried_once(
    workspace: Path, data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "a.py").write_text("a")
    store = make_store(workspace, data)
    real = GitRunner.run
    failures: list[str] = []

    async def flaky(self: GitRunner, *args: str, **kw: Any) -> bytes:
        if args[0] == "update-index" and "--add" in args and not failures:
            failures.append("x")
            raise GitError("a transient failure")
        return await real(self, *args, **kw)

    monkeypatch.setattr(GitRunner, "run", flaky)

    snap = await store.snapshot()

    assert snap is not None and store.enabled
    assert files(store, snap.commit) == ["a.py"]


async def test_a_persistently_failing_update_index_turns_the_store_off(
    workspace: Path, data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "a.py").write_text("a")
    store = make_store(workspace, data)
    real = GitRunner.run

    async def broken(self: GitRunner, *args: str, **kw: Any) -> bytes:
        if args[0] == "update-index" and "--add" in args:
            raise GitError("always")
        return await real(self, *args, **kw)

    monkeypatch.setattr(GitRunner, "run", broken)

    assert await store.snapshot() is None

    assert not store.enabled


async def test_ignore_rules_of_the_users_config_home_do_not_apply(workspace: Path, data: Path, tmp_path: Path) -> None:
    config = tmp_path / "xdg"
    (config / "git").mkdir(parents=True)
    (config / "git" / "ignore").write_text("*.log\n")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text("[core]\n\texcludesFile = ~/ignore-all\n")
    (home / "ignore-all").write_text("*\n")
    (workspace / "run.log").write_text("log")
    (workspace / "a.py").write_text("a")
    env = {**os.environ, "HOME": str(home), "XDG_CONFIG_HOME": str(config)}
    store = make_store(workspace, data, env)

    snap = await store.snapshot()

    assert snap is not None
    assert files(store, snap.commit) == ["a.py", "run.log"]


async def test_the_workspace_git_info_exclude_still_applies(workspace: Path, data: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    (workspace / ".git" / "info" / "exclude").write_text("*.tmp\n")
    (workspace / "a.py").write_text("a")
    (workspace / "x.tmp").write_text("x")
    store = make_store(workspace, data)

    snap = await store.snapshot()

    assert snap is not None
    assert files(store, snap.commit) == ["a.py"]


async def test_protected_files_in_a_nested_repository_are_left_out(workspace: Path, data: Path) -> None:
    nested = workspace / "vendor"
    nested.mkdir()
    subprocess.run(["git", "init", "-q", str(nested)], check=True)
    for name in (".env", "id_rsa", "x.PEM", "ok.txt"):
        (nested / name).write_text("x")
    store = make_store(workspace, data)

    snap = await store.snapshot()

    assert snap is not None
    assert files(store, snap.commit) == ["vendor/ok.txt"]


async def test_a_nested_repositorys_own_ignore_rules_apply(workspace: Path, data: Path) -> None:
    nested = workspace / "vendor"
    deeper = nested / "inner"
    deeper.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(nested)], check=True)
    subprocess.run(["git", "init", "-q", str(deeper)], check=True)
    (nested / ".gitignore").write_text("*.o\nbuild/\n")
    (nested / "build").mkdir()
    (nested / "build" / "out").write_text("x")
    (nested / "a.o").write_text("x")
    (nested / "a.c").write_text("x")
    (nested / ".git" / "info" / "exclude").write_text("*.tmp\n")
    (nested / "b.tmp").write_text("x")
    (deeper / ".gitignore").write_text("*.pyc\n")
    (deeper / "m.pyc").write_text("x")
    (deeper / "m.py").write_text("x")
    store = make_store(workspace, data)

    snap = await store.snapshot()

    assert snap is not None
    assert files(store, snap.commit) == ["vendor/.gitignore", "vendor/a.c", "vendor/inner/.gitignore", "vendor/inner/m.py"]


async def test_a_tracked_file_that_becomes_ignored_leaves_the_tree(workspace: Path, data: Path) -> None:
    (workspace / "a.py").write_text("a")
    (workspace / "gen.txt").write_text("g")
    store = make_store(workspace, data)
    assert await store.snapshot() is not None
    (workspace / ".gitignore").write_text("gen.txt\n")

    snap = await store.snapshot()

    assert snap is not None
    assert files(store, snap.commit) == [".gitignore", "a.py"]


async def test_agent_config_names_match_whatever_the_case(workspace: Path, data: Path) -> None:
    agent = workspace / ".AGENT"
    (agent / "Commands").mkdir(parents=True)
    (agent / "SETTINGS.JSON").write_text("x")
    (agent / "Commands" / "c.md").write_text("x")
    (agent / "notes.txt").write_text("x")
    store = make_store(workspace, data)

    snap = await store.snapshot()

    assert snap is not None
    assert files(store, snap.commit) == [".AGENT/notes.txt"]


async def test_close_during_the_first_snapshot_leaves_no_folder(
    workspace: Path, data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "a.py").write_text("a")
    store = make_store(workspace, data)
    real = GitRunner.run

    async def closing(self: GitRunner, *args: str, **kw: Any) -> bytes:
        if args[0] == "init":
            store.close()  # lands before the git process is registered: nothing to kill, git then makes the folder
        return await real(self, *args, **kw)

    monkeypatch.setattr(GitRunner, "run", closing)

    assert await store.snapshot() is None

    assert not store.folder.exists()


async def test_the_checkpoint_folders_are_private_when_created(workspace: Path, data: Path) -> None:
    (workspace / "a.py").write_text("a")
    store = make_store(workspace, data)
    old = os.umask(0)
    try:
        assert await store.snapshot() is not None
    finally:
        os.umask(old)

    for folder in (data, data.parent):
        assert stat.S_IMODE(folder.stat().st_mode) == 0o700
