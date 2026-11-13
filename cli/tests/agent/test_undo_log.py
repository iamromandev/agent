"""The undo log: what it records around a write, and what it keeps within its limits."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from src.agent.undo import UndoLog


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


def test_the_first_touch_wins_over_three_writes(workspace: Path) -> None:
    target = workspace / "a.txt"
    target.write_bytes(b"v0")
    log = UndoLog(workspace)
    log.begin(0, "p")

    for data in (b"v1", b"v2", b"v3"):
        write(log, target, data)
    log.finish()

    turn = log.pop()
    assert turn is not None
    change = turn.changes[target]
    assert change.before == b"v0"
    assert change.existed
    assert change.after == digest(b"v3")


def test_a_missing_file_is_recorded_as_not_existing_with_the_folders_the_write_creates(workspace: Path) -> None:
    (workspace / "src").mkdir()
    target = workspace / "src" / "new" / "deep" / "f.txt"
    log = UndoLog(workspace)
    log.begin(0, "p")

    write(log, target, b"x")
    log.finish()

    turn = log.pop()
    assert turn is not None
    change = turn.changes[target]
    assert not change.existed
    assert change.before is None
    assert change.created_dirs == (workspace / "src" / "new", workspace / "src" / "new" / "deep")


def test_created_dirs_are_empty_when_the_folder_exists(workspace: Path) -> None:
    target = workspace / "f.txt"
    log = UndoLog(workspace)
    log.begin(0, "p")

    write(log, target, b"x")
    log.finish()

    turn = log.pop()
    assert turn is not None
    assert turn.changes[target].created_dirs == ()


def test_created_dirs_stay_inside_the_workspace(tmp_path: Path, workspace: Path) -> None:
    outside = tmp_path / "elsewhere" / "f.txt"
    log = UndoLog(workspace)
    log.begin(0, "p")

    log.before(outside)
    log.finish()

    turn = log.pop()
    assert turn is not None
    assert turn.changes[outside].created_dirs == ()


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


def test_a_failed_write_keeps_an_entry_an_earlier_call_made(workspace: Path) -> None:
    target = workspace / "a.txt"
    target.write_bytes(b"v0")
    log = UndoLog(workspace)
    log.begin(0, "p")
    write(log, target, b"v1")

    log.before(target)
    log.after(target)  # second call changed nothing
    log.finish()

    turn = log.pop()
    assert turn is not None
    assert list(turn.changes) == [target]
    assert turn.changes[target].before == b"v0"


def test_a_file_over_the_limit_is_marked_too_large_and_holds_no_bytes(workspace: Path) -> None:
    big = workspace / "big.bin"
    big.write_bytes(b"x" * 11)
    small = workspace / "small.txt"
    small.write_bytes(b"abc")
    log = UndoLog(workspace, file_limit=10)
    log.begin(0, "p")

    write(log, big, b"y" * 11)
    write(log, small, b"def")
    log.finish()

    turn = log.pop()
    assert turn is not None
    assert turn.changes[big].too_large
    assert turn.changes[big].before is None
    assert turn.changes[small].before == b"abc"
    assert turn.size == 3


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


def test_finish_without_begin_is_a_no_op(workspace: Path) -> None:
    log = UndoLog(workspace)

    log.finish()

    assert log.pop() is None


def test_note_bash_counts_on_the_current_turn(workspace: Path) -> None:
    log = UndoLog(workspace)
    log.begin(4, "run it")

    log.note_bash()
    log.note_bash()
    log.finish()

    turn = log.pop()
    assert turn is not None
    assert (turn.start, turn.prompt, turn.bash) == (4, "run it", 2)


def test_turns_pop_newest_first_and_before_without_a_turn_records_nothing(workspace: Path) -> None:
    target = workspace / "a.txt"
    log = UndoLog(workspace)
    log.before(target)  # no turn open
    log.begin(0, "one")
    log.finish()
    log.begin(2, "two")
    log.finish()

    first, second = log.pop(), log.pop()

    assert first is not None
    assert second is not None
    assert (first.prompt, second.prompt) == ("two", "one")
    assert log.pop() is None


def test_a_file_that_cannot_fit_at_all_evicts_no_older_turn(workspace: Path) -> None:
    log = UndoLog(workspace, file_limit=20, total_limit=10)
    old, big = workspace / "old.txt", workspace / "big.txt"
    old.write_bytes(b"1234")
    big.write_bytes(b"123456789012")
    log.begin(0, "older")
    write(log, old, b"x")
    log.finish()
    log.begin(1, "now")

    write(log, big, b"y")
    log.finish()

    newest = log.pop()
    assert newest is not None
    assert newest.changes[big].too_large
    assert log.pop() is not None  # the older turn survived


def test_a_failed_write_that_made_folders_removes_the_empty_ones(workspace: Path) -> None:
    log = UndoLog(workspace)
    target = workspace / "a" / "b" / "f.txt"
    log.begin(0, "p")

    log.before(target)
    target.parent.mkdir(parents=True)
    log.after(target)  # the write itself failed: no file

    assert not (workspace / "a").exists()
    log.finish()
    turn = log.pop()
    assert turn is not None
    assert turn.changes == {}


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


def test_after_hashes_a_too_large_file_without_reading_it_whole(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = UndoLog(workspace, file_limit=10)
    target = workspace / "big.bin"
    target.write_bytes(b"x" * 50)
    log.begin(0, "p")
    log.before(target)
    target.write_bytes(b"y" * 3_000_000)

    def refuse(path: Path) -> bytes:
        raise AssertionError("the whole file was read into memory")

    monkeypatch.setattr("src.agent.undo._read_nofollow", refuse)

    log.after(target)

    log.finish()
    turn = log.pop()
    assert turn is not None
    assert turn.changes[target].after == digest(b"y" * 3_000_000)
