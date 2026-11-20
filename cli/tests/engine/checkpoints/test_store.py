"""CheckpointStore.snapshot with the real git: contents, exclusions, isolation from the user's repository."""

from __future__ import annotations

import asyncio
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


async def test_a_nested_repository_is_snapshotted_as_plain_files_without_its_git(workspace: Path, data: Path) -> None:
    nested = workspace / "vendor" / "lib"
    nested.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(nested)], check=True)
    (nested / "x.py").write_text("x")
    store = make_store(workspace, data)

    snap = await store.snapshot()

    assert snap is not None
    assert files(store, snap.commit) == ["vendor/lib/x.py"]


async def test_the_shadow_folder_is_private_and_removed_on_close(workspace: Path, data: Path) -> None:
    (workspace / "a.py").write_text("a")
    store = make_store(workspace, data)
    await store.snapshot()

    assert stat.S_IMODE(store.folder.stat().st_mode) == 0o700
    assert str(os.getpid()) in store.folder.name
    store.close()
    assert not store.folder.exists()
    assert await store.snapshot() is None


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
    (agent / "Agents").mkdir()
    (agent / "Agents" / "a.md").write_text("x")
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
