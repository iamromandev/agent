"""The change set of a turn: from two real checkpoint snapshots, or from the undo log; and one file's diff."""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import os
import random
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
from src.engine.checkpoints.git import GitError, GitTimeout
from src.engine.checkpoints.store import CheckpointStore, DiffEntry
from src.engine.guardrails.setup import Guardrails
from src.engine.turn_diff import (
    DIFF_BYTES,
    LINE_LIMIT,
    DiffItem,
    TurnDiff,
    build_diff,
    load_file,
    render_lines,
)
from src.engine.undo import UndoLog, UndoTurn

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root.resolve()


@pytest.fixture
def store(workspace: Path, tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(workspace, tmp_path / "data", Guardrails.off(workspace), dict(os.environ))


async def turn_around(store: CheckpointStore, change, *, large: bool = False) -> UndoTurn:
    """An UndoTurn whose snapshots surround `change()`."""
    first = await store.snapshot()
    change()
    second = await store.snapshot()
    assert first is not None and second is not None
    return UndoTurn(0, "p", before=first.commit, after=second.commit, skipped_large=second.skipped_large)


async def diff_of(store: CheckpointStore, workspace: Path, turn: UndoTurn) -> TurnDiff:
    return await build_diff(turn, store, workspace)


def item(diff: TurnDiff, path: str) -> DiffItem:
    return next(i for i in diff.items if i.path == path)


async def text_of(diff: TurnDiff, path: str) -> list[str]:
    return [line.text for line in render_lines(await load_file(diff, item(diff, path)))]


@needs_git
async def test_a_png_is_binary_with_sizes_and_no_hunks(store: CheckpointStore, workspace: Path) -> None:
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + bytes(range(256))
    turn = await turn_around(store, lambda: (workspace / "i.png").write_bytes(png))
    diff = await diff_of(store, workspace, turn)

    shown = await text_of(diff, "i.png")

    found = item(diff, "i.png")
    assert (found.status, found.added, found.removed, found.note) == ("A", None, None, "binary")
    assert shown == [f"binary: 0 → {len(png)} bytes"]


@needs_git
async def test_invalid_utf8_is_binary(store: CheckpointStore, workspace: Path) -> None:
    turn = await turn_around(store, lambda: (workspace / "l.txt").write_bytes(b"caf\xe9\n"))
    diff = await diff_of(store, workspace, turn)

    assert (item(diff, "l.txt").added, item(diff, "l.txt").note) == (None, "binary")
    assert (await text_of(diff, "l.txt"))[0].startswith("binary:")


@needs_git
async def test_a_file_that_becomes_a_symlink_is_a_type_change(store: CheckpointStore, workspace: Path) -> None:
    (workspace / "f").write_text("data\n")

    def change() -> None:
        (workspace / "f").unlink()
        (workspace / "f").symlink_to("elsewhere")

    diff = await diff_of(store, workspace, await turn_around(store, change))

    assert item(diff, "f").status == "T"
    assert await text_of(diff, "f") == ["mode 100644 → 120000", "old: regular file, 5 bytes", "new -> elsewhere"]


@needs_git
async def test_chmod_alone_is_a_mode_note_with_zero_counts_and_no_hunks(
    store: CheckpointStore, workspace: Path
) -> None:
    (workspace / "run.sh").write_text("#!/bin/sh\n")
    (workspace / "run.sh").chmod(0o644)
    turn = await turn_around(store, lambda: (workspace / "run.sh").chmod(0o755))
    diff = await diff_of(store, workspace, turn)

    found = item(diff, "run.sh")

    assert (found.status, found.added, found.removed, found.note) == ("M", 0, 0, "mode 100644 → 100755")
    assert await text_of(diff, "run.sh") == ["mode 100644 → 100755"]


@needs_git
async def test_a_ten_thousand_line_rewrite_is_cut_at_the_line_limit_with_a_note(
    store: CheckpointStore, workspace: Path
) -> None:
    (workspace / "big.txt").write_text("".join(f"old {n}\n" for n in range(10_000)))
    turn = await turn_around(store, lambda: (workspace / "big.txt").write_text("".join(f"new {n}\n" for n in range(10_000))))
    diff = await diff_of(store, workspace, turn)

    file_diff = await load_file(diff, item(diff, "big.txt"))
    shown = render_lines(file_diff)

    assert len(file_diff.lines) == LINE_LIMIT and file_diff.cut > 0
    assert shown[-1].text == f"[{file_diff.cut} more lines not shown]"
    assert (item(diff, "big.txt").added, item(diff, "big.txt").removed) == (10_000, 10_000)


class FakeStore:
    """diff/numstat of synthetic entries."""

    def __init__(self, count: int) -> None:
        self.entries = [DiffEntry(f"f{n:05}.txt", 0, None, 0o100644, "a" * 40) for n in range(count)]

    async def diff(self, before: str, after: str) -> list[DiffEntry]:
        return self.entries

    async def numstat(
        self, before: str, after: str, skip: Sequence[str] = ()
    ) -> dict[str, tuple[int, int] | None]:
        return {e.path: (1, 0) for e in self.entries}

    async def blob_sizes(self, shas: Sequence[str]) -> dict[str, int]:
        return {sha: 10 for sha in shas}

    async def binary_blobs(self, shas: Sequence[str]) -> set[str]:
        return set()


# --- the file-log fallback ---------------------------------------------------------------------------------------


def logged_write(log: UndoLog, path: Path, data: bytes) -> None:
    log.before(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    log.after(path)


def finished(log: UndoLog) -> UndoTurn:
    log.finish()
    turn = log.peek()
    assert turn is not None
    return turn


async def test_a_deleted_file_from_the_log_shows_its_removal(workspace: Path) -> None:
    (workspace / "gone.txt").write_text("bye\n")
    log = UndoLog(workspace)
    log.begin(0, "p")
    log.before(workspace / "gone.txt")
    (workspace / "gone.txt").unlink()
    log.after(workspace / "gone.txt")

    diff = await build_diff(finished(log), None, workspace)

    assert [(i.status, i.removed) for i in diff.items] == [("D", 1)]
    assert "-bye" in await text_of(diff, "gone.txt")


async def test_a_logged_file_changed_after_the_turn_is_unavailable(workspace: Path) -> None:
    (workspace / "a.txt").write_text("one\n")
    log = UndoLog(workspace)
    log.begin(0, "p")
    logged_write(log, workspace / "a.txt", b"two\n")
    turn = finished(log)
    diff = await build_diff(turn, None, workspace)
    (workspace / "a.txt").write_text("three\n")  # after the list was built

    shown = await text_of(diff, "a.txt")

    assert shown == ["unavailable: the file changed since the turn"]


async def test_a_logged_file_that_is_a_symlink_now_is_never_followed(workspace: Path, tmp_path: Path) -> None:
    secret = tmp_path / "outside.txt"
    secret.write_text("two\n")
    log = UndoLog(workspace)
    log.begin(0, "p")
    logged_write(log, workspace / "a.txt", b"two\n")
    turn = finished(log)
    (workspace / "a.txt").unlink()
    (workspace / "a.txt").symlink_to(secret)  # same bytes, but read through a link

    diff = await build_diff(turn, None, workspace)

    assert diff.items[0].note == "unavailable"


async def test_a_file_the_log_could_not_keep_is_too_large_or_unavailable(workspace: Path) -> None:
    log = UndoLog(workspace, file_limit=4)
    log.begin(0, "p")
    (workspace / "big.txt").write_text("0123456789")
    logged_write(log, workspace / "big.txt", b"x")

    diff = await build_diff(finished(log), None, workspace)

    assert diff.items[0].note == "too large"
    assert await text_of(diff, "big.txt") == ["too large: no diff"]


@needs_git
async def test_a_failing_store_falls_back_to_the_log_when_it_has_entries(
    store: CheckpointStore, workspace: Path
) -> None:
    log = UndoLog(workspace)
    log.begin(0, "p")
    logged_write(log, workspace / "a.txt", b"hi\n")
    turn = finished(log)
    turn.before, turn.after = "a" * 40, "b" * 40  # never snapshotted: git cannot read them

    diff = await build_diff(turn, store, workspace)

    assert diff.fallback and [i.path for i in diff.items] == ["a.txt"]


# --- bounded work, list-time facts, the other details ------------------------------------------------------------


@needs_git
async def test_a_failing_numstat_lists_the_files_with_unknown_counts_and_no_fallback(
    store: CheckpointStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "a.txt").write_text("one\n")

    async def timed_out(*args: object, **kwargs: object) -> dict[str, tuple[int, int] | None]:
        raise GitTimeout("git diff-tree took longer than 5 s")

    monkeypatch.setattr(store, "numstat", timed_out)
    turn = await turn_around(store, lambda: (workspace / "a.txt").write_text("two\nthree\n"))

    diff = await diff_of(store, workspace, turn)

    assert not diff.fallback
    found = item(diff, "a.txt")
    assert (found.status, found.added, found.removed, found.note) == ("M", None, None, None)
    assert "+three" in await text_of(diff, "a.txt")  # the pane does not need the counts


@needs_git
async def test_a_failing_binary_scan_lists_unknown_counts_instead_of_trusting_text_counts(
    store: CheckpointStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failed(shas: Sequence[str]) -> set[str]:
        raise GitError("cat-file failed")

    monkeypatch.setattr(store, "binary_blobs", failed)
    turn = await turn_around(store, lambda: (workspace / "i.png").write_bytes(b"\x00\x01\n"))

    diff = await diff_of(store, workspace, turn)

    assert (item(diff, "i.png").added, item(diff, "i.png").note) == (None, None)


@needs_git
async def test_a_file_over_the_compare_limit_skips_numstat_and_is_listed_as_too_large_to_compare(
    store: CheckpointStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[Sequence[str]] = []
    real = store.numstat

    async def spy(before: str, after: str, skip: Sequence[str] = ()) -> dict[str, tuple[int, int] | None]:
        asked.append(list(skip))
        return await real(before, after, skip)

    monkeypatch.setattr(store, "numstat", spy)
    big = "".join(f"line {n}\n" for n in range(DIFF_BYTES // 8))  # a little over the limit
    turn = await turn_around(
        store,
        lambda: ((workspace / "big.txt").write_text(big), (workspace / "small.txt").write_text("x\n")),
    )

    diff = await diff_of(store, workspace, turn)

    assert asked == [["big.txt"]]
    found = item(diff, "big.txt")
    assert (found.status, found.added, found.note) == ("A", None, "too large to compare")
    assert (item(diff, "small.txt").added, item(diff, "small.txt").note) == (1, None)
    assert await text_of(diff, "big.txt") == [f"too large to compare: 0 → {len(big)} bytes"]


@needs_git
async def test_a_file_that_was_over_the_snapshot_cap_before_the_turn_is_modified_and_too_large(
    workspace: Path,
) -> None:
    turn = UndoTurn(0, "p", before="a" * 40, after="b" * 40, skipped_large=("huge.bin",), large_before=("huge.bin",))

    diff = await build_diff(turn, cast(CheckpointStore, FakeStore(0)), workspace)

    assert [(i.status, i.path, i.note) for i in diff.items] == [("M", "huge.bin", "too large")]
    assert [line.text for line in render_lines(await load_file(diff, diff.items[0]))] == ["too large: no diff"]


def blank_lines(count: int) -> bytes:
    return b"\n" * count


def padded(first: bytes, count: int, last: bytes) -> bytes:
    """`count` blank lines between two different lines: nothing for the prefix/suffix trim to take off."""
    return first + b"\n" * count + last


@needs_git
async def test_a_400_kb_file_of_repeated_lines_is_refused_within_a_second(
    store: CheckpointStore, workspace: Path
) -> None:
    rng = random.Random(7)
    vocabulary = [f"a line of the little vocabulary {n:02}\n" for n in range(20)]
    text = lambda: "".join(rng.choice(vocabulary) for _ in range(14_000)).encode()  # noqa: E731
    (workspace / "rep.txt").write_bytes(text())
    new = text()
    assert 300_000 < len(new) < DIFF_BYTES
    turn = await turn_around(store, lambda: (workspace / "rep.txt").write_bytes(new))
    diff = await diff_of(store, workspace, turn)

    start = time.perf_counter()
    shown = await text_of(diff, "rep.txt")
    seconds = time.perf_counter() - start

    assert seconds < 1
    assert shown[0].startswith("too large to compare:")


async def test_the_fallback_counts_and_the_diff_come_from_one_bounded_comparison(workspace: Path) -> None:
    (workspace / "blank.txt").write_bytes(padded(b"A\n", 20_000, b"B\n"))
    log = UndoLog(workspace)
    log.begin(0, "p")
    logged_write(log, workspace / "blank.txt", padded(b"C\n", 20_000, b"D\n"))
    turn = finished(log)
    start = time.perf_counter()

    diff = await build_diff(turn, None, workspace)
    shown = await text_of(diff, "blank.txt")

    assert time.perf_counter() - start < 2
    found = diff.items[0]
    assert (found.added, found.note) == (None, "too large to compare")
    assert shown[0].startswith("too large to compare:")


async def test_cancelling_a_load_stops_the_thread_so_shutting_down_does_not_wait(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = random.Random(3)
    vocabulary = [f"line {n:02}\n" for n in range(20)]
    old = ("A\n" + "".join(rng.choice(vocabulary) for _ in range(2_000)) + "B\n").encode()
    new = ("C\n" + "".join(rng.choice(vocabulary) for _ in range(2_000)) + "D\n").encode()
    path = workspace / "rep.txt"
    path.write_bytes(new)
    item = DiffItem(
        "rep.txt", "M", None, None,
        old_bytes=old, new_path=path, new_digest=hashlib.sha256(new).digest(),
    )  # fmt: skip
    diff = TurnDiff((item,), 0, True, None, workspace, UndoTurn(0, "p"))
    real = difflib.SequenceMatcher.find_longest_match

    def slow(self: difflib.SequenceMatcher, *args: int) -> difflib.Match:  # many searches, each 50 ms: seconds in all
        time.sleep(0.05)
        return real(self, *args)

    monkeypatch.setattr(difflib.SequenceMatcher, "find_longest_match", slow)
    monkeypatch.setattr("src.engine.turn_diff.DIFF_SECONDS", 60.0)  # only the cancel can end it early
    task = asyncio.create_task(load_file(diff, item))
    await asyncio.sleep(0.3)

    start = time.perf_counter()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.get_running_loop().shutdown_default_executor()  # what the app's exit waits for

    assert time.perf_counter() - start < 0.5


async def test_the_no_newline_marker_is_display_meta_not_a_removal(workspace: Path) -> None:
    (workspace / "a.txt").write_bytes(b"x")
    log = UndoLog(workspace)
    log.begin(0, "p")
    logged_write(log, workspace / "a.txt", b"y")
    diff = await build_diff(finished(log), None, workspace)

    lines = render_lines(await load_file(diff, diff.items[0]))

    assert [line.kind for line in lines if line.text.startswith("\\")] == ["meta", "meta"]


async def test_the_fallback_does_not_read_a_file_over_the_log_cap(workspace: Path) -> None:
    log = UndoLog(workspace)
    log.begin(0, "p")
    path = workspace / "big.bin"
    log.before(path)
    with open(path, "wb") as handle:
        handle.truncate(9 * 1024 * 1024)  # the digest the turn left is the one of the file now
    log.after(path)

    diff = await build_diff(finished(log), None, workspace)

    assert (diff.items[0].note, diff.items[0].added) == ("unavailable", None)


async def test_the_fallback_leaves_out_a_file_whose_content_is_back_to_what_it_was(workspace: Path) -> None:
    (workspace / "a.txt").write_text("one\n")
    (workspace / "b.txt").write_text("keep\n")
    log = UndoLog(workspace)
    log.begin(0, "p")
    logged_write(log, workspace / "a.txt", b"two\n")
    logged_write(log, workspace / "a.txt", b"one\n")  # written back
    logged_write(log, workspace / "b.txt", b"changed\n")

    diff = await build_diff(finished(log), None, workspace)

    assert [i.path for i in diff.items] == ["b.txt"]


async def test_the_fallback_leaves_out_a_file_the_turn_made_and_deleted(workspace: Path) -> None:
    log = UndoLog(workspace)
    log.begin(0, "p")
    logged_write(log, workspace / "temp.txt", b"scratch\n")
    logged_write(log, workspace / "temp.txt", b"scratch 2\n")
    (workspace / "temp.txt").unlink()  # bash removed it; a later logged call sees it gone
    log.before(workspace / "temp.txt")
    log.after(workspace / "temp.txt")
    turn = finished(log)

    diff = await build_diff(turn, None, workspace)

    assert diff.items == () and diff.fallback
