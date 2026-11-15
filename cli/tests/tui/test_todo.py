"""The todo panel: marks, updates, hiding, escaping, resume, /new, session switch and /undo."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.engine.provider.base import Done, Message, TextDelta, ToolCall, ToolCallEvent
from src.engine.session import ToolMeta
from src.tui import StartChoice
from src.tui.screens.sessions import SessionPicker
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.todo_panel import TodoPanel
from src.tui.widgets.transcript import ToolEntry

from tests.tui.conftest import AppFactory, scripted, settled, type_and_send, until

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


async def test_the_panel_is_hidden_without_a_list(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test():
        assert not shown(app)


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


async def test_the_panel_updates_in_place_and_hides_when_all_done(make_app: AppFactory) -> None:
    app = make_app([todo("t1", ("A", "in_progress"), ("B", "pending")), todo("t2", ("A", "done"), ("B", "done")), REPLY])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await settled(app)
        await pilot.pause()

        assert not shown(app)


async def test_an_empty_list_hides_the_panel(make_app: AppFactory) -> None:
    app = make_app([todo("t1", ("A", "pending")), todo("t2"), REPLY])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await settled(app)
        await pilot.pause()

        assert not shown(app)


async def test_item_text_is_escaped(make_app: AppFactory) -> None:
    app = make_app([todo("t1", ("evil\x1b[31m‮text", "pending")), REPLY])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await settled(app)
        await pilot.pause()

        text = panel_text(app)
        assert "\x1b" not in text and "‮" not in text
        assert "\\x1b" in text and "\\u202e" in text


async def test_markup_in_an_item_is_shown_as_text(make_app: AppFactory) -> None:
    app = make_app([todo("t1", ("[bold]x[/bold]", "pending")), REPLY])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await settled(app)
        await pilot.pause()

        assert "[bold]x[/bold]" in panel_text(app)


async def test_resume_shows_the_saved_list_and_the_next_request_has_the_note(
    make_app: AppFactory, save_session: SaveSession
) -> None:
    save_session(saved_todo_turn("t1", ("Saved item", "in_progress")))
    app = make_app([REPLY], start=StartChoice("continue"))
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()
        assert panel_text(app) == "▸ Saved item"

        await type_and_send(pilot, "next")
        await settled(app)

        messages = scripted(app).requests[0][0]
        [(index, note)] = [(i, m.content) for i, m in enumerate(messages) if m.role == "system" and i > 0]
        assert "- [in_progress] Saved item" in note
        assert messages[index + 1] == Message("user", "next")


async def test_new_clears_the_list(make_app: AppFactory) -> None:
    app = make_app([todo("t1", ("A", "pending")), REPLY])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await settled(app)
        await pilot.pause()
        assert shown(app)

        await type_and_send(pilot, "/new")
        await until(pilot, lambda: not shown(app))


async def test_undo_restores_the_list_from_before_the_turn(make_app: AppFactory) -> None:
    app = make_app([todo("t1", ("First", "pending")), REPLY, todo("t2", ("Second", "pending")), REPLY])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "one")
        await settled(app)
        await type_and_send(pilot, "two")
        await settled(app)
        await pilot.pause()
        assert panel_text(app) == "○ Second"

        app.query_one(ChatInput).clear()
        await type_and_send(pilot, "/undo")
        await until(pilot, lambda: panel_text(app) == "○ First")


async def test_switching_sessions_shows_that_sessions_list(make_app: AppFactory, save_session: SaveSession) -> None:
    save_session(saved_todo_turn("t1", ("Old session item", "pending")), age=500)
    app = make_app([todo("t2", ("Live item", "pending")), REPLY])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await settled(app)
        await pilot.pause()
        assert panel_text(app) == "○ Live item"

        await type_and_send(pilot, "/sessions")
        await until(pilot, lambda: isinstance(app.screen, SessionPicker))
        await pilot.press("down")  # the live session is the newest: the saved one is second
        await pilot.press("enter")
        await settled(app)
        await pilot.pause()

        assert panel_text(app) == "\u25cb Old session item"
