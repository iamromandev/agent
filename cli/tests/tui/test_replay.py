"""Redrawing a saved conversation: `Transcript.replay`, and the status bar's estimated `~`."""

from __future__ import annotations

from typing import Any

import pytest
from src.agent import UsageUpdate
from src.agent.session import TranscriptItem
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import ICONS, AssistantEntry, NoticeEntry, ToolEntry, Transcript, UserEntry
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


async def test_a_replayed_tool_entry_is_collapsed_with_its_summary_and_no_diff(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await app.query_one(Transcript).replay([TranscriptItem("tool", "edit a.py", "replaced 1 occurrence(s)")])
        await pilot.pause()

        (tool,) = app.query(ToolEntry)
        assert tool.title == "edit a.py · replaced 1 occurrence(s)"
        assert tool.collapsed
        assert tool.finished
        assert tool.detail == ""
        assert len(tool.query_one("Collapsible Contents").children) == 0


async def test_a_replayed_tool_without_a_summary_shows_just_its_label(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await app.query_one(Transcript).replay([TranscriptItem("tool", "bash ls", "")])
        await pilot.pause()

        (tool,) = app.query(ToolEntry)
        assert tool.title == "bash ls"


@pytest.mark.parametrize("label", ["app/[id]/page.tsx", "echo [/]", "read [red]x[bold].txt", "[$var] \\[x]\\"])
async def test_a_replayed_label_with_markup_characters_shows_literally(make_app: AppFactory, label: str) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await app.query_one(Transcript).replay([TranscriptItem("tool", label, f"saw {label}")])
        await pilot.pause()

        (tool,) = app.query(ToolEntry)
        assert _title(tool) == f"{label} · saw {label}"


async def test_replay_leaves_no_assistant_entry_open_for_the_next_message(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        transcript = app.query_one(Transcript)
        await transcript.replay([TranscriptItem("assistant", "old reply")])
        await transcript.write_assistant("new")
        await pilot.pause()

        assert [e.text for e in app.query(AssistantEntry)] == ["old reply", "new"]


async def test_replayed_tools_do_not_collide_with_live_ones(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        transcript = app.query_one(Transcript)
        await transcript.replay(
            [TranscriptItem("tool", "read a", "1 lines"), TranscriptItem("tool", "read b", "2 lines")]
        )
        await pilot.pause()

        assert len(list(app.query(ToolEntry))) == 2
        assert all(t.finished for t in app.query(ToolEntry))


async def test_the_status_shows_a_tilde_when_the_count_is_estimated(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        status.window = 8192
        status.used = 900
        status.estimated = True
        await pilot.pause()

        assert "ctx ~900 / 8192" in status.line


async def test_the_status_shows_no_tilde_for_a_real_count(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        status.window = 8192
        status.used = 900
        await pilot.pause()

        assert "ctx 900 / 8192" in status.line
        assert "~" not in status.line


async def test_a_usage_update_sets_and_clears_the_estimated_mark(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test():
        status = app.query_one(StatusBar)
        await app._on_agent_event(UsageUpdate(400, 1000, estimated=True))
        assert (status.used, status.estimated) == (400, True)
        assert "ctx ~400 / 1000" in status.line

        await app._on_agent_event(UsageUpdate(500, 1000))
        assert (status.used, status.estimated) == (500, False)
        assert "ctx 500 / 1000" in status.line
        assert "~" not in status.line


async def test_the_context_warning_uses_the_estimate_too(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        status.window = 1000
        status.used = 850
        status.estimated = True
        await pilot.pause()

        assert "ctx ~850 / 1000" in status.line
        assert "context 85%" in status.line


async def test_a_replay_longer_than_one_batch_keeps_every_entry_in_order(make_app: AppFactory) -> None:
    items = [TranscriptItem("user" if i % 2 == 0 else "notice", f"item {i}") for i in range(1203)]
    app = make_app()
    async with app.run_test() as pilot:
        await app.query_one(Transcript).replay(items)
        await pilot.pause()

        entries: list[Any] = list(app.query("#transcript > *"))
        assert len(entries) == 1203
        assert [e.text.removeprefix("> ").removeprefix(f"{INFO} ") for e in entries] == [i.text for i in items]
