"""The task block in the transcript: child tool lines folded under one entry, then the report."""

from __future__ import annotations

from pathlib import Path

from core.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from core.engine.web import FetchResponse
from core.testing.fakes import FakeFetcher
from src.tui.screens.approval import ApprovalScreen
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import AssistantEntry, NoticeEntry, ToolEntry
from textual.widgets import Collapsible, Static

from tests.tui.conftest import AppFactory, settled, type_and_send, until


def _task_call() -> list:
    args = {"description": "look around", "prompt": "find it"}
    return [ToolCallEvent(ToolCall("t1", "task", args)), Done(10, 5)]


def _child_read() -> list:
    return [ToolCallEvent(ToolCall("r1", "read", {"path": "a.txt"})), Done(300, 20)]


def _contents(entry: ToolEntry) -> str:
    return "\n".join(str(s.render()) for s in entry.query_one(Collapsible.Contents).query(Static))


async def test_a_task_shows_one_block_with_child_lines_then_steps_and_report(
    make_app: AppFactory, tmp_path: Path
) -> None:
    (tmp_path / "a.txt").write_text("hello\n")
    app = make_app(
        [_task_call(), _child_read(), [TextDelta("CHILD REPORT"), Done(400, 30)], [TextDelta("parent answer"), Done(21, 1)]],
        task=True,
    )
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await settled(app)
        await pilot.pause()

        (block,) = app.query(ToolEntry)
        assert block.label == "task look around"
        assert block.collapsed  # finished: folded
        assert "2 steps" in str(block.title) and "k tokens" in str(block.title)
        text = _contents(block)
        assert "read a.txt" in text
        assert "CHILD REPORT" in text
        assert text.index("read a.txt") < text.index("CHILD REPORT")
        assert "".join(e.text for e in app.query(AssistantEntry)) == "parent answer"  # child text is not parent text

        block.collapsed = False
        await pilot.pause()
        assert not block.collapsed


async def test_escape_on_a_child_approval_denies_and_interrupts_the_turn(make_app: AppFactory) -> None:
    url = "https://example.com/docs"
    fetch = [ToolCallEvent(ToolCall("f1", "web_fetch", {"url": url})), Done(1, 1)]
    fetcher = FakeFetcher(FetchResponse(url, 200, "OK", "text/plain", None, b"hi", 2))
    app = make_app([_task_call(), fetch], task=True, task_fetcher=fetcher)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await pilot.press("escape")
        await until(pilot, lambda: not app.query_one(StatusBar).busy)

        assert not isinstance(app.screen, ApprovalScreen)
        assert fetcher.calls == []
        assert any("interrupted" in e.text for e in app.query(NoticeEntry))
        (block,) = app.query(ToolEntry)
        assert block.finished
