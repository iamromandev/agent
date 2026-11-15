"""The task block in the transcript: child tool lines folded under one entry, then the report."""

from __future__ import annotations

from pathlib import Path

from src.engine.provider.base import Done, Message, ProviderError, TextDelta, ToolCall, ToolCallEvent
from src.engine.session import ToolMeta
from src.engine.web import FetchResponse
from src.tui import StartChoice
from src.tui.screens.approval import ApprovalScreen
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import AssistantEntry, NoticeEntry, ToolEntry, Transcript
from textual.widgets import Collapsible, Static
from textual.widgets._collapsible import CollapsibleTitle

from tests.fakes import FakeFetcher, Gate
from tests.tui.conftest import AppFactory, reached, settled, type_and_send, until


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


async def test_a_running_task_is_expanded_and_lists_each_child_call(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n")
    gate = Gate()
    app = make_app([_task_call(), _child_read(), [gate, TextDelta("r"), Done(1, 1)], [TextDelta("ok"), Done(1, 1)]], task=True)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await reached(gate)
        await pilot.pause()

        (block,) = app.query(ToolEntry)
        assert not block.collapsed
        assert "read a.txt" in _contents(block)
        assert "✓" in _contents(block)  # the child call has finished, the task has not

        gate.release.set()
        await settled(app)
        await pilot.pause()
        assert block.collapsed


async def test_child_usage_stays_off_the_context_indicator(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n")
    app = make_app(
        [_task_call(), _child_read(), [TextDelta("r"), Done(900, 50)], [TextDelta("ok"), Done(21, 1)]],
        task=True,
        context_window=100_000,
    )
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await settled(app)
        await pilot.pause()

        assert app.query_one(StatusBar).used == 22  # the parent's last request, not the child's 950
        (block,) = app.query(ToolEntry)
        assert "1.3k tokens" in str(block.title)


async def test_escape_during_a_child_stream_interrupts_the_turn(make_app: AppFactory, tmp_path: Path) -> None:
    gate = Gate()
    app = make_app([_task_call(), [TextDelta("part"), gate, Done(1, 1)]], task=True)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await reached(gate)
        await pilot.press("escape")
        await until(pilot, lambda: not app.query_one(StatusBar).busy)

        assert any("interrupted" in e.text for e in app.query(NoticeEntry))
        (block,) = app.query(ToolEntry)
        assert block.finished
        assert not list(app.query(AssistantEntry))
        assert app.query_one(Transcript)


async def test_clicking_the_title_unfolds_and_folds_a_finished_task(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n")
    app = make_app([_task_call(), _child_read(), [TextDelta("R"), Done(1, 1)], [TextDelta("ok"), Done(1, 1)]], task=True)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await settled(app)
        await pilot.pause()
        (block,) = app.query(ToolEntry)
        assert block.collapsed

        await pilot.click(CollapsibleTitle)
        await pilot.pause()
        assert not block.collapsed
        await pilot.click(CollapsibleTitle)
        await pilot.pause()
        assert block.collapsed


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


async def test_a_resumed_session_shows_a_task_as_one_collapsed_tool_call(
    make_app: AppFactory, save_session
) -> None:
    call = ToolCall("t1", "task", {"description": "look around", "prompt": "find it"})
    save_session(
        [
            (Message("user", "dig"), None),
            (Message("assistant", "", [call]), None),
            (Message("tool", "REPORT", tool_call_id="t1", name="task"), ToolMeta("task look around", "3 steps · 1.2k tokens")),
            (Message("assistant", "found it"), None),
        ]
    )
    app = make_app(start=StartChoice("continue"), task=True)
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()

        (block,) = app.query(ToolEntry)
        assert block.collapsed and block.finished
        assert block.title == "task look around · 3 steps · 1.2k tokens"
        assert len(block.query_one(Collapsible.Contents).children) == 0


async def test_a_child_provider_error_ends_the_block_as_an_error_with_one_error_line(make_app: AppFactory) -> None:
    app = make_app([_task_call(), [ProviderError("model fell over")], [TextDelta("recovered"), Done(1, 1)]], task=True)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await settled(app)
        await pilot.pause()

        (block,) = app.query(ToolEntry)
        assert block.finished and block.collapsed
        assert "model fell over" in _contents(block)
        assert "✗" in str(block.title) or "error" in str(block.title)
        assert not [e for e in app.query(NoticeEntry) if "model fell over" in e.text]  # not repeated as a notice
        assert [e.text for e in app.query(AssistantEntry)] == ["recovered"]


async def test_a_second_child_call_with_a_reused_id_gets_its_own_line(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n")
    app = make_app(
        [_task_call(), _child_read(), _child_read(), [TextDelta("R"), Done(1, 1)], [TextDelta("ok"), Done(1, 1)]],
        task=True,
    )
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await settled(app)
        await pilot.pause()

        (block,) = app.query(ToolEntry)
        assert _contents(block).count("read a.txt") == 2
        assert "⋯" not in _contents(block)
