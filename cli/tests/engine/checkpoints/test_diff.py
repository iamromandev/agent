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


async def test_diff_lists_added_modified_and_deleted_files(store: CheckpointStore, workspace: Path) -> None:
    before, after = await two_commits(store, workspace)

    entries = {e.path: e for e in await store.diff(before, after)}

    assert sorted(entries) == ["added.txt", "edit.txt", "gone.txt", "link"]
    added, edited, gone, link = entries["added.txt"], entries["edit.txt"], entries["gone.txt"], entries["link"]
    assert added.old_sha is None and added.new_sha is not None and added.new_mode == 0o100644
    assert edited.old_sha is not None and edited.new_sha is not None and edited.old_sha != edited.new_sha
    assert gone.old_sha is not None and gone.new_sha is None and gone.old_mode == 0o100644
    assert link.new_mode == 0o120000


async def test_diff_of_a_commit_with_itself_is_empty(store: CheckpointStore, workspace: Path) -> None:
    before, _ = await two_commits(store, workspace)

    assert await store.diff(before, before) == []


async def test_diff_reports_a_mode_change(store: CheckpointStore, workspace: Path) -> None:
    _, before = await two_commits(store, workspace)
    (workspace / "run.sh").chmod(0o644)
    third = await store.snapshot()
    assert third is not None

    [entry] = await store.diff(before, third.commit)
    assert entry.path == "run.sh" and (entry.old_mode, entry.new_mode) == (0o100755, 0o100644)


async def test_read_blobs_returns_the_content_by_sha(store: CheckpointStore, workspace: Path) -> None:
    before, after = await two_commits(store, workspace)
    entries = {e.path: e for e in await store.diff(before, after)}
    shas = [entries["edit.txt"].old_sha, entries["gone.txt"].old_sha, entries["edit.txt"].old_sha]

    blobs = await store.read_blobs([s for s in shas if s])

    assert blobs == {entries["edit.txt"].old_sha: b"old", entries["gone.txt"].old_sha: b"bye"}


async def test_read_blobs_leaves_out_what_would_pass_the_total_limit(store: CheckpointStore, workspace: Path) -> None:
    before, after = await two_commits(store, workspace)
    entries = {e.path: e for e in await store.diff(before, after)}
    old, gone = entries["edit.txt"].old_sha, entries["gone.txt"].old_sha
    assert old and gone

    blobs = await store.read_blobs([old, gone], limit=4)  # "old" and "bye" are 3 bytes each

    assert blobs == {old: b"old"}


async def test_read_blobs_of_nothing_is_empty_and_runs_no_git(store: CheckpointStore) -> None:
    assert await store.read_blobs([]) == {}


async def test_read_blobs_of_an_unknown_sha_raises(store: CheckpointStore, workspace: Path) -> None:
    await two_commits(store, workspace)

    with pytest.raises(GitError):
        await store.read_blobs(["0123456789abcdef0123456789abcdef01234567"])


async def test_diff_before_any_snapshot_raises(store: CheckpointStore) -> None:
    with pytest.raises(GitError):
        await store.diff("a" * 40, "b" * 40)


def test_diff_entry_is_a_value() -> None:
    assert DiffEntry("a", 0, None, 0o100644, "x") == DiffEntry("a", 0, None, 0o100644, "x")
