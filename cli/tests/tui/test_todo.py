"""The todo panel: marks, updates, hiding, escaping, resume, /new, session switch and /undo."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.engine.provider.base import Done, Message, TextDelta, ToolCall, ToolCallEvent
from src.engine.session import ToolMeta
from src.tui.widgets.todo_panel import TodoPanel
from src.tui.widgets.transcript import ToolEntry

from tests.tui.conftest import AppFactory, settled, type_and_send

SaveSession = Callable[..., Path]
REPLY = [TextDelta("ok"), Done(1, 1)]


def todo(call_id: str, *pairs: tuple[str, str]) -> list:
    items = [{"content": c, "status": s} for c, s in pairs]
    return [ToolCallEvent(ToolCall(call_id, "todo", {"items": items})), Done(1, 1)]


def panel_text(app) -> str:
    return app.query_one(TodoPanel).plain


def shown(app) -> bool:
    return bool(app.query_one(TodoPanel).display)


def saved_todo_turn(call_id: str, *pairs: tuple[str, str]):
    items = [{"content": c, "status": s} for c, s in pairs]
    return [
        (Message("user", "go"), None),
        (Message("assistant", "", [ToolCall(call_id, "todo", {"items": items})]), None),
        (Message("tool", "Updated", tool_call_id=call_id, name="todo"), ToolMeta("todo", "0/1 done")),
        (Message("assistant", "done"), None),
    ]


async def test_a_todo_call_shows_one_line_per_item_with_a_mark_per_status(make_app: AppFactory) -> None:
    app = make_app([todo("t1", ("Find uses", "in_progress"), ("Rename", "pending"), ("Docs", "done")), REPLY])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await settled(app)
        await pilot.pause()

        assert shown(app)
        assert panel_text(app).splitlines() == ["▸ Find uses", "○ Rename", "✓ Docs"]
        [entry] = app.query(ToolEntry)
        assert entry.label == "todo"
        assert "todo \u00b7 1/3 done" in str(entry.title)
