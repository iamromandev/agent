"""The /undo notice of a checkpoint restore."""

from __future__ import annotations

from src.engine.events import Notice
from src.engine.undo import CHANGED, UndoResult, checkpoint_notices

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
