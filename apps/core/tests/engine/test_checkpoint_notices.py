"""The /undo notice of a checkpoint restore."""

from __future__ import annotations

from core.engine.events import Notice
from core.engine.undo import CHANGED, CHANGED_BY_OTHER, UndoResult, checkpoint_notices

OUTSIDE = "Effects outside the workspace and running processes are not undone."


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


def test_a_path_another_session_changed_is_named_with_that_reason() -> None:
    result = UndoResult([("restored", "b.py", None), ("skipped", "s.py", CHANGED_BY_OTHER), ("skipped", "c.py", CHANGED)])

    text = checkpoint_notices(result, ())[0].text

    assert text == "Undid the last turn: 1 restored, 0 deleted, 2 skipped: s.py (changed by another session), c.py"
