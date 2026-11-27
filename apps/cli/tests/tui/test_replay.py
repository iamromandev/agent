"""Redrawing a saved conversation: `Transcript.replay`, and the status bar's estimated `~`."""

from __future__ import annotations

from core.engine.session import TranscriptItem
from src.tui.styles import ICONS
from src.tui.widgets.transcript import AssistantEntry, NoticeEntry, ToolEntry, Transcript, UserEntry
from textual.widgets._collapsible import CollapsibleTitle

from tests.tui.conftest import AppFactory

INFO = ICONS["info"]


def _title(entry: ToolEntry) -> str:
    return str(entry.query_one(CollapsibleTitle).label)


async def test_replay_creates_the_entries_in_order(make_app: AppFactory) -> None:
    items = [
        TranscriptItem("user", "what is in main?"),
        TranscriptItem("assistant", "Let me look."),
        TranscriptItem("tool", "read src/main.py", "31 lines"),
        TranscriptItem("assistant", "It has **31** lines."),
        TranscriptItem("notice", "Context compacted: pruned 1 tool output (ctx ~900 → ~400)"),
        TranscriptItem("user", "thanks"),
    ]
    app = make_app()
    async with app.run_test() as pilot:
        await app.query_one(Transcript).replay(items)
        await pilot.pause()

        kinds = [type(e).__name__ for e in app.query("#transcript > *")]
        assert kinds == ["UserEntry", "AssistantEntry", "ToolEntry", "AssistantEntry", "NoticeEntry", "UserEntry"]
        assert [e.text for e in app.query(UserEntry)] == ["what is in main?", "thanks"]
        assert [e.text for e in app.query(AssistantEntry)] == ["Let me look.", "It has **31** lines."]
        (notice,) = app.query(NoticeEntry)
        assert notice.text == f"{INFO} Context compacted: pruned 1 tool output (ctx ~900 → ~400)"
