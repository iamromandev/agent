from __future__ import annotations

import time
from pathlib import Path

import pytest
from src.agent.provider.base import Done, ProviderError, TextDelta, ToolCall, ToolCallEvent
from src.tui.app import AgentApp
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import AssistantEntry, NoticeEntry, ToolEntry, UserEntry
from textual.widgets import Markdown

from tests.fakes import Gate
from tests.tui.conftest import AppFactory, reached, settled, type_and_send, until


def _read_call(path: str) -> ToolCallEvent:
    return ToolCallEvent(ToolCall("c1", "read", {"path": path}))


async def test_transcript_shows_user_streamed_text_and_a_tool_entry(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x = 1\n" * 31)
    app = make_app(
        [
            [TextDelta("Let me "), TextDelta("look."), _read_call("src/main.py"), Done(10, 5)],
            [TextDelta("It has "), TextDelta("**31** lines."), Done(20, 5)],
        ]
    )
    async with app.run_test() as pilot:
        await type_and_send(pilot, "what is in main?")
        await settled(app)
        await pilot.pause()

        assert [e.text for e in app.query(UserEntry)] == ["what is in main?"]
        assert [e.text for e in app.query(AssistantEntry)] == ["Let me look.", "It has **31** lines."]
        (tool,) = app.query(ToolEntry)
        assert tool.title == "read src/main.py · 31 lines"
        entries = list(app.query("#transcript > *"))
        kinds = [type(e).__name__ for e in entries]
        assert kinds == ["UserEntry", "AssistantEntry", "ToolEntry", "AssistantEntry"]


async def test_a_failed_turn_shows_the_error_and_the_next_message_works(make_app: AppFactory) -> None:
    app = make_app([[TextDelta("par"), ProviderError("stream dropped")], [TextDelta("fine"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "one")
        await settled(app)
        await pilot.pause()
        notices = [e.text for e in app.query(NoticeEntry)]
        assert any("stream dropped" in n and n.startswith("✗") for n in notices)
        assert not app.query_one(StatusBar).busy

        await type_and_send(pilot, "two")
        await settled(app)
        await pilot.pause()
        assert [e.text for e in app.query(AssistantEntry)][-1] == "fine"


async def test_status_warns_at_80_percent_of_the_window(make_app: AppFactory) -> None:
    app = make_app([[TextDelta("hi"), Done(500, 350)]], context_window=1000)
    async with app.run_test() as pilot:
        assert "⚠" not in app.query_one(StatusBar).line
        await type_and_send(pilot, "x")
        await settled(app)
        await pilot.pause()
        line = app.query_one(StatusBar).line

    assert "ctx 850 / 1000" in line
    assert "⚠ context 85% — /clear" in line


async def test_busy_indicator_runs_from_submit_to_turn_end(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate, TextDelta("late"), Done(1, 1)]])
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        await type_and_send(pilot, "x")
        await reached(gate)
        assert status.busy
        assert "working" in status.line
        gate.release.set()
        await settled(app)
        await pilot.pause()

        assert not status.busy
        assert "working" not in status.line


async def test_escape_interrupts_a_streaming_turn_within_half_a_second(make_app: AppFactory) -> None:
    app = make_app([[TextDelta(f"w{i} ") for i in range(200)] + [Done(1, 1)], [TextDelta("again"), Done(1, 1)]], delay=0.05)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await until(pilot, lambda: any(e.text for e in app.query(AssistantEntry)))

        started = time.monotonic()
        await pilot.press("escape")
        await until(pilot, lambda: any("interrupted" in e.text for e in app.query(NoticeEntry)))
        elapsed = time.monotonic() - started
        await settled(app)

        assert elapsed < 0.5
        assert not app.query_one(StatusBar).busy

        await type_and_send(pilot, "follow up")
        await settled(app)
        await pilot.pause()
        assert [e.text for e in app.query(AssistantEntry)][-1] == "again"


async def test_a_link_in_a_reply_is_not_opened_automatically(
    make_app: AppFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(AgentApp, "open_url", lambda self, url, **_: opened.append(url))
    app = make_app([[TextDelta("see [docs](https://example.invalid/x)"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await settled(app)
        await pilot.pause()
        (entry,) = app.query(AssistantEntry)

        entry.post_message(Markdown.LinkClicked(entry, "https://example.invalid/x"))
        await pilot.pause()

    assert opened == []


async def test_remove_last_turn_removes_that_turn_and_keeps_notices_added_after_it(make_app: AppFactory) -> None:
    from src.tui.widgets.transcript import Transcript

    app = make_app()
    async with app.run_test() as pilot:
        transcript = app.query_one(Transcript)
        first = await transcript.add_user("one")
        await transcript.write_assistant("a1")
        await transcript.end_assistant()
        transcript.end_turn(first)
        second = await transcript.add_user("two")
        await transcript.write_assistant("a2")
        await transcript.add_tool("c1", "write x")
        await transcript.end_assistant()
        transcript.end_turn(second)
        await transcript.add_notice("info", "after the turn")
        await pilot.pause()

        assert await transcript.remove_last_turn() is True
        await pilot.pause()

        assert [type(e).__name__ for e in app.query("#transcript > *")] == [
            "UserEntry",
            "AssistantEntry",
            "NoticeEntry",
        ]
        assert [e.text for e in app.query(UserEntry)] == ["one"]
        assert transcript.tool("c1") is None
        # the next streamed text starts a new entry, not one in the removed turn
        await transcript.write_assistant("fresh")
        await pilot.pause()
        assert [e.text for e in app.query(AssistantEntry)] == ["a1", "fresh"]
        assert await transcript.remove_last_turn() is True  # the first turn is next
        assert [e.text for e in app.query(UserEntry)] == []


async def test_remove_last_turn_with_no_ended_turn_returns_false_and_keeps_the_rest(make_app: AppFactory) -> None:
    from src.tui.widgets.transcript import Transcript

    app = make_app()
    async with app.run_test() as pilot:
        transcript = app.query_one(Transcript)
        await transcript.add_user("never ended")
        await transcript.add_notice("info", "hello")
        await pilot.pause()
        assert await transcript.remove_last_turn() is False
        assert len(list(app.query(NoticeEntry))) == 1
        assert len(list(app.query(UserEntry))) == 1


async def test_discarding_an_orphan_user_entry_removes_only_that_entry(make_app: AppFactory) -> None:
    from src.tui.widgets.transcript import Transcript

    app = make_app()
    async with app.run_test() as pilot:
        transcript = app.query_one(Transcript)
        done = await transcript.add_user("done")
        transcript.end_turn(done)
        orphan = await transcript.add_user("orphan")
        await transcript.add_notice("info", "later notice")
        await pilot.pause()

        assert await transcript.remove_turn(orphan) is True
        await pilot.pause()

        assert [e.text for e in app.query(UserEntry)] == ["done"]
        assert len(list(app.query(NoticeEntry))) == 1
        # an undo now takes the finished turn, not something else
        assert await transcript.remove_last_turn() is True
        assert not list(app.query(UserEntry))


async def test_removing_a_turn_with_an_open_tool_entry_drops_it_from_the_tool_map(make_app: AppFactory) -> None:
    from src.tui.widgets.transcript import Transcript

    app = make_app()
    async with app.run_test() as pilot:
        transcript = app.query_one(Transcript)
        token = await transcript.add_user("go")
        entry = await transcript.add_tool("open", "bash sleep")
        assert not entry.finished
        transcript.end_turn(token)
        await pilot.pause()

        await transcript.remove_last_turn()
        await pilot.pause()

        assert transcript.tool("open") is None
        assert not list(app.query(ToolEntry))
        await transcript.close_open_tools("ended")  # must not touch the removed entry
