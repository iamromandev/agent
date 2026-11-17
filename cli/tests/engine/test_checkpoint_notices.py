"""The /undo notice of a checkpoint restore."""

from __future__ import annotations

from src.engine.events import Notice
from src.engine.undo import CHANGED, UndoResult, checkpoint_notices

OUTSIDE = "Effects outside the workspace and running processes are not undone."


def test_counts_restored_and_deleted() -> None:
    result = UndoResult([("restored", "a.py", None), ("restored", "b.py", None), ("deleted", "c.py", None)])

    assert checkpoint_notices(result, ()) == [
        Notice("info", "Undid the last turn: 2 restored, 1 deleted"),
        Notice("info", OUTSIDE),
    ]


def test_names_the_skipped_files_with_a_warning_level() -> None:
    result = UndoResult([("restored", "a.py", None), ("skipped", "b.py", CHANGED), ("skipped", "c.py", CHANGED)])

    assert checkpoint_notices(result, ()) == [
        Notice("warn", "Undid the last turn: 1 restored, 0 deleted, 2 skipped: b.py, c.py"),
        Notice("info", OUTSIDE),
    ]


def test_files_over_the_size_cap_count_as_skipped() -> None:
    result = UndoResult([("deleted", "n.py", None)])

    first = checkpoint_notices(result, ("big.bin",))[0]

    assert first == Notice("warn", "Undid the last turn: 0 restored, 1 deleted, 1 skipped: big.bin")


def test_nothing_to_restore() -> None:
    assert checkpoint_notices(UndoResult([]), ()) == [
        Notice("info", "Undid the last turn: nothing to restore"),
        Notice("info", OUTSIDE),
    ]


def test_at_most_twenty_names_then_and_n_more() -> None:
    result = UndoResult([("skipped", f"f{i:02}", CHANGED) for i in range(23)])

    text = checkpoint_notices(result, ())[0].text

    assert "23 skipped: " in text
    assert text.endswith("f19, and 3 more")
    assert "f20" not in text


def test_names_are_escaped() -> None:
    result = UndoResult([("skipped", "a\x1b[31m\nb", CHANGED)])

    text = checkpoint_notices(result, ())[0].text

    assert "\x1b" not in text and "\n" not in text
    assert "a\\x1b[31m\\x0ab" in text


def test_a_large_file_the_restore_also_skipped_is_counted_once() -> None:
    result = UndoResult([("skipped", "big.bin", CHANGED), ("deleted", "n.py", None)])

    first = checkpoint_notices(result, ("big.bin",))[0]

    assert first == Notice("warn", "Undid the last turn: 0 restored, 1 deleted, 1 skipped: big.bin")


def test_a_large_file_that_was_restored_or_deleted_is_not_skipped() -> None:
    result = UndoResult([("deleted", "big.bin", None), ("restored", "old.bin", None)])

    first = checkpoint_notices(result, ("big.bin", "old.bin"))[0]

    assert first == Notice("info", "Undid the last turn: 1 restored, 1 deleted")
