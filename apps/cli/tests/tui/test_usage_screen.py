"""/usage in the app: the numbers, closing, arguments, during a turn, resume, escaping."""

from __future__ import annotations

from pathlib import Path

from core.engine.provider.base import Done, TextDelta
from core.engine.session import FileRecorder, sessions_root, workspace_dir
from core.engine.usage import UsageRecord
from src.tui import StartChoice
from src.tui.screens.usage import UsageScreen
from src.tui.widgets.transcript import NoticeEntry, Transcript
from textual.widgets import Static

from tests.tui.conftest import AppFactory, chat, settled, type_and_send, until


def reply(i: int = 100, o: int = 20) -> list:
    return [TextDelta("ok"), Done(i, o)]


def _notices(app) -> list[str]:
    return [e.text.split(" ", 1)[1] if " " in e.text else e.text for e in app.query(NoticeEntry)]


def _shown(app) -> str:
    return app.screen.query_one("#usage", Static).render().plain  # type: ignore[union-attr]


def _entries(app) -> int:
    return len(app.query_one(Transcript).children)


async def test_arguments_give_the_notice_and_no_screen(make_app: AppFactory) -> None:
    app = make_app([])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/usage x")
        await until(pilot, lambda: "/usage takes no arguments" in _notices(app))
        assert not isinstance(app.screen, UsageScreen)


async def test_an_empty_session_says_no_model_calls_yet(make_app: AppFactory) -> None:
    app = make_app([])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/usage")
        await until(pilot, lambda: isinstance(app.screen, UsageScreen))
        assert "No model calls yet" in _shown(app)


async def test_a_resumed_session_sums_its_records_and_a_new_turn_adds_to_them(
    make_app: AppFactory, tmp_path: Path
) -> None:
    root = tmp_path.resolve()
    recorder = FileRecorder(workspace_dir(sessions_root(None), root), root)
    for message, meta in chat("q", "a"):
        recorder.message(message, meta)
    for _ in range(3):
        recorder.usage(UsageRecord("fake", 10, 1, 100, False, "2026-11-18T10:00:00Z"))
    recorder.close()
    assert recorder.path is not None  # make it an old-format file: no usage marker in the header
    recorder.path.write_text(recorder.path.read_text().replace(',"usage":1', "", 1))
    app = make_app([reply()], start=StartChoice("continue"))
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "more")
        await settled(app)
        await type_and_send(pilot, "/usage")
        await until(pilot, lambda: isinstance(app.screen, UsageScreen))
        shown = _shown(app)
        assert "130" in shown  # 3 x 10 + 100 input tokens
        assert "some of this session has no usage records" in shown
