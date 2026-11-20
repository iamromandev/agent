"""CheckpointStore.diff and read_blobs on a hand-made pair of commits."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from src.engine.checkpoints.git import GitError
from src.engine.checkpoints.store import CheckpointStore, DiffEntry
from src.engine.guardrails.setup import Guardrails

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root.resolve()


@pytest.fixture
def store(workspace: Path, tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(workspace, tmp_path / "data", Guardrails.off(workspace), dict(os.environ))


async def two_commits(store: CheckpointStore, workspace: Path) -> tuple[str, str]:
    (workspace / "keep.txt").write_text("same")
    (workspace / "edit.txt").write_text("old")
    (workspace / "gone.txt").write_text("bye")
    (workspace / "run.sh").write_text("#!/bin/sh\n")
    (workspace / "run.sh").chmod(0o755)
    first = await store.snapshot()
    (workspace / "edit.txt").write_text("new")
    (workspace / "gone.txt").unlink()
    (workspace / "added.txt").write_text("hi")
    (workspace / "link").symlink_to("keep.txt")
    second = await store.snapshot()
    assert first is not None and second is not None
    return first.commit, second.commit


async def test_read_blobs_leaves_out_what_would_pass_the_total_limit(store: CheckpointStore, workspace: Path) -> None:
    before, after = await two_commits(store, workspace)
    entries = {e.path: e for e in await store.diff(before, after)}
    old, gone = entries["edit.txt"].old_sha, entries["gone.txt"].old_sha
    assert old and gone

    blobs = await store.read_blobs([old, gone], limit=4)  # "old" and "bye" are 3 bytes each

    assert blobs == {old: b"old"}


def test_diff_entry_is_a_value() -> None:
    assert DiffEntry("a", 0, None, 0o100644, "x") == DiffEntry("a", 0, None, 0o100644, "x")


async def test_numstat_before_any_snapshot_raises(store: CheckpointStore) -> None:
    with pytest.raises(GitError):
        await store.numstat("a" * 40, "b" * 40)


async def commits_with(store: CheckpointStore, workspace: Path, files: dict[str, bytes]) -> tuple[str, str]:
    first = await store.snapshot()
    for name, data in files.items():
        (workspace / name).write_bytes(data)
    second = await store.snapshot()
    assert first is not None and second is not None
    return first.commit, second.commit


async def test_blob_sizes_of_an_unknown_sha_raises(store: CheckpointStore, workspace: Path) -> None:
    await two_commits(store, workspace)

    with pytest.raises(GitError):
        await store.blob_sizes(["0123456789abcdef0123456789abcdef01234567"])
