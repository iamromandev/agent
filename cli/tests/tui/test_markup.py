"""Text from the model, tools and files reaches the widgets literally: brackets are never markup."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.agent.approval import ApprovalRequest
from src.agent.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from src.agent.tools.base import Preview, ToolResult
from src.tui.screens.approval import ApprovalScreen
from src.tui.widgets.transcript import NoticeEntry, ToolEntry, Transcript
from textual.widgets import Collapsible, Label, Static
from textual.widgets._collapsible import CollapsibleTitle

from tests.tui.conftest import AppFactory, settled, type_and_send, until

TRICKY = [
    "echo [/]",
    "[[ -f x ]] && echo [/foo]",
    "read [red]x[bold].txt",
    "app/[id]/page.tsx",
    "[bold",
    "[$var] \\[x] and a trailing backslash\\",
    "[@click=app.quit]q[/]",
]


def _title(entry: ToolEntry) -> str:
    return str(entry.query_one(CollapsibleTitle).label)  # Content's str() is its plain text


def _shown(widget: Static | Label) -> str:
    return str(widget.visual)


@pytest.mark.parametrize("text", TRICKY)
async def test_a_tool_entry_title_shows_the_label_and_summary_literally(make_app: AppFactory, text: str) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        transcript = app.query_one(Transcript)
        entry = await transcript.add_tool("c1", text)
        await pilot.pause()
        assert _title(entry) == text

        await entry.finish(ToolResult("out", False, f"saw {text}"), None, False)
        await pilot.pause()

        assert _title(entry) == f"✓ {text} · saw {text}"
        assert entry.title == f"✓ {text} · saw {text}"


@pytest.mark.parametrize("text", TRICKY)
async def test_a_notice_shows_its_text_literally(make_app: AppFactory, text: str) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await app.query_one(Transcript).add_notice("error", text)
        await pilot.pause()

        (notice,) = app.query(NoticeEntry)
        assert _shown(notice) == f"✗ {text}"


@pytest.mark.parametrize("text", TRICKY)
async def test_the_approval_dialog_shows_title_and_preview_literally(make_app: AppFactory, text: str) -> None:
    app = make_app()
    request = ApprovalRequest("bash", text, Preview("command", f"run {text}", f"body {text}"), True)
    async with app.run_test() as pilot:
        app.push_screen(ApprovalScreen(request))
        await pilot.pause()

        screen = app.screen
        assert _shown(screen.query_one("#title", Label)) == f"Allow run {text}?"
        assert _shown(screen.query_one("#preview", Static)) == f"body {text}"


@pytest.mark.parametrize("path", ["app/[id]/page.tsx", "read [red]x[bold].txt", "[bold"])
async def test_a_model_tool_call_with_brackets_in_its_path_does_not_crash_the_app(
    make_app: AppFactory, tmp_path: Path, path: str
) -> None:
    call = ToolCallEvent(ToolCall("c1", "read", {"path": path}))
    app = make_app([[call, Done(1, 1)], [TextDelta("done"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await settled(app)
        await until(pilot, lambda: bool(list(app.query(ToolEntry))))

        (tool,) = app.query(ToolEntry)
        assert _title(tool).startswith(f"✗ read {path} · ")
        assert tool.finished


@pytest.mark.parametrize("text", TRICKY)
async def test_bash_output_with_markup_characters_is_shown_literally(make_app: AppFactory, text: str) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        entry = await app.query_one(Transcript).add_tool("c1", "bash x")
        await entry.finish(ToolResult(text, False, "exit 0", display=f"\x1b[32m{text}\x1b[0m"), None, False)
        await pilot.pause()

        (shown,) = entry.query(Collapsible.Contents).first().query(Static)
        assert _shown(shown) == text
