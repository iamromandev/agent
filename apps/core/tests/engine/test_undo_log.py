"""The undo log: what it records around a write, and what it keeps within its limits."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from core.engine.undo import UndoLog, read_nofollow


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root.resolve()


def write(log: UndoLog, path: Path, data: bytes) -> None:
    """What the tools do: before, change, after."""
    log.before(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    log.after(path)


def digest(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def test_a_write_that_changed_nothing_leaves_no_entry(workspace: Path) -> None:
    target = workspace / "a.txt"
    target.write_bytes(b"same")
    missing = workspace / "gone.txt"
    log = UndoLog(workspace)
    log.begin(0, "p")

    log.before(target)
    log.after(target)  # the write failed before changing anything
    log.before(missing)
    log.after(missing)
    log.finish()

    turn = log.pop()
    assert turn is not None
    assert turn.changes == {}
    assert turn.size == 0


def test_the_total_limit_drops_the_oldest_turns_first(workspace: Path) -> None:
    log = UndoLog(workspace, file_limit=10, total_limit=10)
    for index in range(3):
        target = workspace / f"f{index}.txt"
        target.write_bytes(b"123456")
        log.begin(index, f"p{index}")
        write(log, target, b"new")
        log.finish()

    assert log.size <= 10
    newest = log.pop()
    assert newest is not None
    assert newest.prompt == "p2"
    assert log.pop() is None  # the older two did not fit beside it, oldest dropped first


def test_the_current_turn_alone_over_the_total_is_marked_too_large(workspace: Path) -> None:
    log = UndoLog(workspace, file_limit=10, total_limit=10)
    a, b = workspace / "a.txt", workspace / "b.txt"
    a.write_bytes(b"123456")
    b.write_bytes(b"123456")
    log.begin(0, "p")

    write(log, a, b"1")
    write(log, b, b"2")
    log.finish()

    turn = log.pop()
    assert turn is not None
    assert not turn.changes[a].too_large
    assert turn.changes[b].too_large


def test_clear_empties_the_stack_and_stops_journaling_the_current_turn(workspace: Path) -> None:
    target = workspace / "a.txt"
    target.write_bytes(b"v0")
    log = UndoLog(workspace)
    log.begin(0, "one")
    write(log, target, b"v1")
    log.finish()
    log.begin(1, "two")

    log.clear()
    write(log, target, b"v2")
    log.finish()

    assert log.pop() is None
    assert log.size == 0


def test_the_same_file_under_another_spelling_is_one_entry(tmp_path: Path, workspace: Path) -> None:
    probe = tmp_path / "Probe"
    probe.write_bytes(b"")
    if not (tmp_path / "pROBE").exists():
        pytest.skip("case-sensitive filesystem")
    target = workspace / "File.txt"
    target.write_bytes(b"v0")
    log = UndoLog(workspace)
    log.begin(0, "p")

    write(log, target, b"v1")
    write(log, workspace / "FILE.txt", b"v2")
    log.finish()

    turn = log.pop()
    assert turn is not None
    assert len(turn.changes) == 1
    assert next(iter(turn.changes.values())).before == b"v0"


def test_a_turn_without_marks_has_no_commits_and_mark_outside_a_turn_is_ignored(workspace: Path) -> None:
    log = UndoLog(workspace)
    log.mark(before="x")
    log.begin(0, "p")
    log.finish()
    turn = log.pop()

    assert turn is not None
    assert (turn.before, turn.after) == (None, None)


def _newest(log: UndoLog) -> str | None:
    turn = log.peek()
    return turn.prompt if turn is not None else None


def test_read_nofollow_reads_a_regular_file_and_never_a_symlink(workspace: Path) -> None:
    (workspace / "a.txt").write_bytes(b"data")
    (workspace / "link").symlink_to("a.txt")

    assert read_nofollow(workspace / "a.txt") == b"data"
    with pytest.raises(OSError):
        read_nofollow(workspace / "link")
