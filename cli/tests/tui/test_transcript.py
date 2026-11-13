from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from rich.style import Style
from rich.text import Text
from src.agent.provider.base import Done, ProviderError, TextDelta, ToolCall, ToolCallEvent
from src.agent.tools.base import ToolResult
from src.tui.app import AgentApp
from src.tui.styles import ACCENT, DIM, ERROR, ICONS, NOTICE, OK, TOOL_NAME
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import AssistantEntry, NoticeEntry, ToolEntry, Transcript, UserEntry
from textual.content import Content
from textual.widgets import Collapsible, Markdown, Static

from tests.fakes import Gate
from tests.tui.conftest import AppFactory, reached, settled, type_and_send, until

Level = Literal["info", "warn", "error"]


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
        assert tool.title == "✓ read src/main.py · 31 lines"
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


async def test_a_successful_edit_title_has_a_green_mark_a_bold_name_and_a_dim_summary(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        entry = await app.query_one(Transcript).add_tool("c1", "edit src/app.py")
        await entry.finish(ToolResult("ok", False, "1 replaced"), None, False)
        await pilot.pause()

        title = _title_of(entry)
        assert title.plain == "✓ edit src/app.py · 1 replaced"
        styled = {title.plain[s.start : s.end]: s.style for s in title.spans}
        assert styled["✓"] == _same(OK)
        assert styled["edit"] == _same(TOOL_NAME)
        assert styled["src/app.py"] == _same(ACCENT)
        assert styled["1 replaced"] == _same(DIM)


async def test_an_error_result_gets_a_red_cross(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        entry = await app.query_one(Transcript).add_tool("c1", "read nope.txt")
        await entry.finish(ToolResult("not found", True, "not found"), None, False)
        await pilot.pause()

        title = _title_of(entry)
        assert title.plain == "✗ read nope.txt · not found"
        assert title.spans[0].style == _same(ERROR)


async def test_an_interrupted_call_gets_the_dim_ended_mark(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        entry = await app.query_one(Transcript).add_tool("c1", "bash sleep 9")
        await entry.finish(None, None, False, "interrupted")
        await pilot.pause()

        title = _title_of(entry)
        assert title.plain == "⏹ bash sleep 9 · interrupted"
        assert title.spans[0].style == _same(DIM)


async def test_a_replayed_entry_is_styled_without_an_invented_mark() -> None:
    entry = ToolEntry("edit a.py", "replaced 1 occurrence(s)")
    title = _title_of(entry)
    assert title.plain == "edit a.py · replaced 1 occurrence(s)"
    assert title.spans[0].style == _same(TOOL_NAME)


@pytest.mark.parametrize("level", ["info", "warn", "error"])
async def test_a_notice_is_styled_by_level(make_app: AppFactory, level: str) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await app.query_one(Transcript).add_notice(cast("Level", level), "hello")
        await pilot.pause()
        (notice,) = app.query(NoticeEntry)
        assert notice.text == f"{ICONS[level]} hello"
        content = notice.visual
        assert isinstance(content, Content)
        assert content.plain == f"{ICONS[level]} hello"
        (span,) = content.spans
        assert span.style == _same(NOTICE[level])


async def test_a_finished_bash_entry_shows_its_output_in_color(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        entry = await app.query_one(Transcript).add_tool("c1", "bash x")
        await entry.finish(ToolResult("ok\nexit 0", False, "exit 0", display="\x1b[32mok\x1b[0m\nexit 0"), None, False)
        await pilot.pause()

        (shown,) = entry.query(Collapsible.Contents).first().query(Static)
        content = shown.visual
        assert isinstance(content, Content)
        assert content.plain == "ok\nexit 0"
        (span,) = content.spans
        style = cast("Any", span.style)
        assert (content.plain[span.start : span.end], style.foreground.ansi) == ("ok", 2)
        assert entry.detail == "ok\nexit 0"  # the plain copy stays the record


async def test_output_without_a_display_copy_is_shown_as_plain_text(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        entry = await app.query_one(Transcript).add_tool("c1", "read a")
        await entry.finish(ToolResult("a\x1b[1b"), None, False)
        await pilot.pause()

        (shown,) = entry.query(Collapsible.Contents).first().query(Static)
        assert _plain(shown) == "a\x1b[1b"


async def test_markup_looking_text_in_colored_output_is_shown_literally(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        entry = await app.query_one(Transcript).add_tool("c1", "bash x")
        display = "\x1b[31m[red]x[/red] [$var] [bold]\x1b[0m"
        await entry.finish(ToolResult("[red]x[/red] [$var] [bold]", False, "exit 0", display=display), None, False)
        await pilot.pause()

        (shown,) = entry.query(Collapsible.Contents).first().query(Static)
        assert _plain(shown) == "[red]x[/red] [$var] [bold]"


async def test_cursor_movement_leaves_no_raw_escape_in_the_shown_output(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        entry = await app.query_one(Transcript).add_tool("c1", "bash x")
        result = await _bash_result("printf '\\033[2K\\033[1A\\033[32mok\\033[0m'")
        await entry.finish(result, None, False)
        await pilot.pause()

        (shown,) = entry.query(Collapsible.Contents).first().query(Static)
        assert "\x1b" not in _plain(shown)
        assert _plain(shown) == "ok\nexit 0"


def _title_of(entry: Collapsible) -> Content:
    title = cast("Any", entry.title)
    assert isinstance(title, Content)
    return title


def _same(style: Style) -> object:
    """The Textual style a Rich style turns into when it is shown."""
    text = Text()
    text.append("x", style)
    return Content.from_rich_text(text).spans[0].style


def _plain(static: Static) -> str:
    content = static.visual
    assert isinstance(content, Content)
    return content.plain


async def _bash_result(command: str) -> ToolResult:
    import os
    from pathlib import Path

    from src.agent.tools.base import Prepared, ToolContext
    from src.agent.tools.shell import BashTool

    os.environ.setdefault("SHELL", "/bin/sh")
    prepared = await BashTool().prepare({"command": command}, ToolContext(Path.cwd(), 5.0))
    assert isinstance(prepared, Prepared)
    return await prepared.execute()
