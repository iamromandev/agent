"""/undo in the TUI: files, transcript, input box, notices, busy refusal, resumed sessions."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest
from core.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from core.engine.session import FileRecorder
from core.testing.fakes import Gate
from src.tui.app import HELP
from src.tui.screens.approval import ApprovalScreen
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import NoticeEntry, Transcript, UserEntry

from tests.tui.conftest import AppFactory, choose, reached, settled, type_and_send, until

SaveSession = Callable[..., Path]
REPLY = [TextDelta("ok"), Done(1, 1)]


def _write(call_id: str, path: str, content: str) -> ToolCallEvent:
    return ToolCallEvent(ToolCall(call_id, "write", {"path": path, "content": content}))


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _box(app) -> str:
    return app.query_one(ChatInput).text


async def _undo(pilot, app) -> None:
    """Submit /undo from an empty box (the previous undo may have put its prompt back)."""
    app.query_one(ChatInput).clear()
    await type_and_send(pilot, "/undo")
    await pilot.pause()


def _left(app, screen) -> Callable[[], bool]:
    return lambda: app.screen is not screen


async def _approve(pilot, app, times: int = 1) -> None:
    for _ in range(times):
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        screen = app.screen
        await choose(pilot, "once")
        await until(pilot, _left(app, screen))


async def test_undo_while_a_turn_runs_is_refused_and_changes_nothing(
    make_app: AppFactory, tmp_path: Path
) -> None:
    gate = Gate()
    app = make_app([[_write("w1", "a.txt", "new"), Done(1, 1)], [TextDelta("a"), gate, TextDelta("b"), Done(1, 1)]])
    (tmp_path / "a.txt").write_text("old")
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await _approve(pilot, app)
        await reached(gate)  # the write landed; the turn is still running
        assert (tmp_path / "a.txt").read_text() == "new"
        history = list(app.agent.history)
        recorder = app.agent._recorder
        assert isinstance(recorder, FileRecorder) and recorder.path is not None
        saved = recorder.path.read_bytes()

        await _undo(pilot, app)

        assert any("busy" in n and "/undo" in n for n in _notices(app))
        assert app.agent.history == history
        assert (tmp_path / "a.txt").read_text() == "new"
        assert recorder.path.read_bytes() == saved  # no undo record written
        assert b'"undo"' not in saved
        assert [e.text for e in app.query(UserEntry)] == ["go"]
        assert _box(app) == ""
        assert app.query_one(StatusBar).busy and app._turn_active and app.agent.busy  # the turn is untouched
        gate.release.set()
        await settled(app)
        assert not app.query_one(StatusBar).busy and not app._turn_active and not app.agent.busy

        await _undo(pilot, app)  # and the turn is undoable once it ended
        assert (tmp_path / "a.txt").read_text() == "old"
        assert _box(app) == "go"


def _newest(app) -> str:
    info = app.store.latest()
    return info.id if info else ""


@pytest.mark.parametrize("key", ["escape"])
async def test_interrupt_keys_during_an_undo_do_nothing_to_it(
    make_app: AppFactory, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "hello")
        await settled(app)
        real = app.agent.undo
        seen: list[tuple[bool, bool, bool]] = []

        async def undo_with_a_key(emit):
            app.action_interrupt_turn() if key == "escape" else app.action_interrupt_or_quit()
            seen.append((app._turn_active, app.query_one(StatusBar).busy, app.agent.busy))
            return await real(emit)

        monkeypatch.setattr(app.agent, "undo", undo_with_a_key)
        app._turn_entered = False  # stale after a turn that Esc cancelled before its first line
        await _undo(pilot, app)

        assert seen == [(True, True, False)]  # the busy guard stayed up through the key
        assert app.is_running
        assert _box(app) == "hello"
        assert not app._turn_active and not app.query_one(StatusBar).busy


async def test_an_unexpected_error_from_undo_shows_an_error_notice_and_frees_the_app(
    make_app: AppFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "hello")
        await settled(app)

        async def broken(emit):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(app.agent, "undo", broken)
        await _undo(pilot, app)

        assert any("kaboom" in n for n in _notices(app))
        assert not app._turn_active and not app.query_one(StatusBar).busy


async def test_esc_between_the_user_entry_and_the_turn_removes_the_orphan_and_keeps_the_prompt(
    make_app: AppFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app([REPLY, REPLY])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "first")
        await settled(app)
        transcript = app.query_one(Transcript)
        real = transcript.add_user

        async def add_user_then_esc(text):
            token = await real(text)
            if text == "second":
                app.interrupt_turn()  # Esc lands right after the entry is drawn
                await asyncio.sleep(0)
            return token

        monkeypatch.setattr(transcript, "add_user", add_user_then_esc)
        await type_and_send(pilot, "second")
        await settled(app)
        await pilot.pause()

        assert [e.text for e in app.query(UserEntry)] == ["first"]  # no orphan on screen
        assert _box(app) == "second"  # and the message is back in the box
        assert not app._turn_active and not app.query_one(StatusBar).busy

        await _undo(pilot, app)  # removes the real first turn, not something else
        assert not list(app.query(UserEntry))
        assert "first" in _box(app)


async def test_help_lists_undo(make_app: AppFactory) -> None:
    assert "/undo undo the last turn" in HELP
    app = make_app()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/help")
        await pilot.pause()
        assert any("/undo" in n for n in _notices(app))
