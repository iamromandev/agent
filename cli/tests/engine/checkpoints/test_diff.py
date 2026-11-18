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


async def test_numstat_counts_added_and_removed_lines_per_path(store: CheckpointStore, workspace: Path) -> None:
    (workspace / "edit.txt").write_text("a\nb\nc\n")
    (workspace / "gone.txt").write_text("x\ny\n")
    first = await store.snapshot()
    (workspace / "edit.txt").write_text("a\nB\nc\nd\n")
    (workspace / "gone.txt").unlink()
    (workspace / "new.txt").write_text("1\n2\n3\n")
    second = await store.snapshot()
    assert first is not None and second is not None

    counts = await store.numstat(first.commit, second.commit)

    assert counts == {"edit.txt": (2, 1), "gone.txt": (0, 2), "new.txt": (3, 0)}


async def test_numstat_counts_a_binary_file_as_text_because_the_diff_attribute_is_set(store: CheckpointStore, workspace: Path) -> None:
    first = await store.snapshot()
    (workspace / "img.bin").write_bytes(b"\x89PNG\x00\x01\n\x02\n")
    second = await store.snapshot()
    assert first is not None and second is not None

    counts = await store.numstat(first.commit, second.commit)

    assert counts["img.bin"] is not None


async def test_numstat_keeps_odd_path_names_intact(store: CheckpointStore, workspace: Path) -> None:
    first = await store.snapshot()
    (workspace / "we ird\tname\n.txt").write_text("a\n")
    second = await store.snapshot()
    assert first is not None and second is not None

    assert await store.numstat(first.commit, second.commit) == {"we ird\tname\n.txt": (1, 0)}


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


async def test_numstat_leaves_out_the_skipped_paths_and_takes_them_literally(
    store: CheckpointStore, workspace: Path
) -> None:
    before, after = await commits_with(
        store, workspace, {"a.txt": b"1\n", "[ab].txt": b"1\n2\n", "keep.txt": b"1\n2\n3\n"}
    )

    counts = await store.numstat(before, after, skip=["[ab].txt"])

    assert counts == {"a.txt": (1, 0), "keep.txt": (3, 0)}


async def test_numstat_skipping_every_path_is_empty_not_everything(store: CheckpointStore, workspace: Path) -> None:
    before, after = await commits_with(store, workspace, {"a.txt": b"1\n"})

    assert await store.numstat(before, after, skip=["a.txt"]) == {}


async def test_blob_sizes_are_read_without_the_content(store: CheckpointStore, workspace: Path) -> None:
    before, after = await commits_with(store, workspace, {"a.txt": b"12345", "e.txt": b""})
    entries = {e.path: e for e in await store.diff(before, after)}
    shas = [s for s in (entries["a.txt"].new_sha, entries["e.txt"].new_sha) if s]

    sizes = await store.blob_sizes(shas)

    assert sizes == {entries["a.txt"].new_sha: 5, entries["e.txt"].new_sha: 0}


async def test_blob_sizes_of_an_unknown_sha_raises(store: CheckpointStore, workspace: Path) -> None:
    await two_commits(store, workspace)

    with pytest.raises(GitError):
        await store.blob_sizes(["0123456789abcdef0123456789abcdef01234567"])


async def test_binary_blobs_are_the_ones_with_a_nul_or_invalid_utf8_anywhere(
    store: CheckpointStore, workspace: Path
) -> None:
    chunk = 64 * 1024
    files = {
        "text.txt": "caf\u00e9\n".encode(),
        "empty.txt": b"",
        "nul.bin": b"\x89PNG\x00\x01",
        "latin1.txt": b"caf\xe9\n",
        "late-nul.bin": b"a" * (3 * chunk) + b"\x00",
        "straddle.txt": b"a" * (chunk - 1) + "\u00e9".encode() * 10,  # a 2-byte character across the read boundary
    }
    before, after = await commits_with(store, workspace, files)
    entries = {e.path: e.new_sha for e in await store.diff(before, after)}

    binary = await store.binary_blobs([sha for sha in entries.values() if sha])

    assert {name for name, sha in entries.items() if sha in binary} == {"nul.bin", "latin1.txt", "late-nul.bin"}


async def test_binary_blobs_of_nothing_runs_no_git(store: CheckpointStore) -> None:
    assert await store.binary_blobs([]) == set()
