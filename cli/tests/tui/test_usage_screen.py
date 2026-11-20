"""/usage in the app: the numbers, closing, arguments, during a turn, resume, escaping."""

from __future__ import annotations

from pathlib import Path

from src.engine.provider.base import Done, TextDelta
from src.engine.session import FileRecorder, sessions_root, workspace_dir
from src.engine.usage import UsageRecord
from src.tui import StartChoice
from src.tui.app import HELP
from src.tui.screens.usage import UsageScreen
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.transcript import NoticeEntry, Transcript
from textual.widgets import Static

from tests.fakes import Gate
from tests.tui.conftest import AppFactory, chat, reached, scripted, settled, type_and_send, until


def reply(i: int = 100, o: int = 20) -> list:
    return [TextDelta("ok"), Done(i, o)]


def _notices(app) -> list[str]:
    return [e.text.split(" ", 1)[1] if " " in e.text else e.text for e in app.query(NoticeEntry)]


def _shown(app) -> str:
    return app.screen.query_one("#usage", Static).render().plain  # type: ignore[union-attr]


def _entries(app) -> int:
    return len(app.query_one(Transcript).children)


async def test_usage_shows_the_totals_rate_and_context_and_escape_closes_it(make_app: AppFactory) -> None:
    app = make_app([reply(), reply()], model="qwen3:14b")
    async with app.run_test() as pilot:
        for text in ("one", "two"):
            await type_and_send(pilot, text)
            await settled(app)
        entries = _entries(app)
        await type_and_send(pilot, "/usage")
        await until(pilot, lambda: isinstance(app.screen, UsageScreen))
        shown = _shown(app)
        assert "qwen3:14b" in shown
        assert "Input tokens" in shown and "200" in shown and "40" in shown and "240" in shown
        assert "Model calls" in shown and "Turns in context" in shown and "2" in shown
        assert "Tokens/s (replies)" in shown and "Context" in shown and "/ 1,000" in shown
        await pilot.press("escape")
        await until(pilot, lambda: not isinstance(app.screen, UsageScreen))
        assert _entries(app) == entries


async def test_q_closes_it(make_app: AppFactory) -> None:
    app = make_app([])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/usage")
        await until(pilot, lambda: isinstance(app.screen, UsageScreen))
        await pilot.press("q")
        await until(pilot, lambda: not isinstance(app.screen, UsageScreen))


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


async def test_two_models_give_a_row_each(make_app: AppFactory) -> None:
    app = make_app([reply(), reply(5, 6)])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "one")
        await settled(app)
        scripted(app).switch("other", 1000)
        await type_and_send(pilot, "two")
        await settled(app)
        await type_and_send(pilot, "/usage")
        await until(pilot, lambda: isinstance(app.screen, UsageScreen))
        shown = _shown(app)
        assert "By model" in shown and "fake" in shown and "other" in shown


async def test_all_zero_counts_say_unknown(make_app: AppFactory) -> None:
    app = make_app([reply(0, 0)])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "one")
        await settled(app)
        await type_and_send(pilot, "/usage")
        await until(pilot, lambda: isinstance(app.screen, UsageScreen))
        assert "unknown (the model reports no counts)" in _shown(app)


async def test_usage_works_during_a_turn(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([reply(), [gate, *reply()]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "one")
        await settled(app)
        await type_and_send(pilot, "slow")
        await reached(gate)
        await type_and_send(pilot, "/usage")
        await until(pilot, lambda: isinstance(app.screen, UsageScreen))
        assert "Model calls" in _shown(app) and "120" in _shown(app)
        await pilot.press("escape")
        gate.release.set()
        await settled(app)


async def test_a_model_name_with_escapes_is_written_out(make_app: AppFactory) -> None:
    app = make_app([reply()], model="bad\x1b[31mname")
    async with app.run_test() as pilot:
        await type_and_send(pilot, "one")
        await settled(app)
        await type_and_send(pilot, "/usage")
        await until(pilot, lambda: isinstance(app.screen, UsageScreen))
        assert "\x1b" not in _shown(app) and "\\x1b" in _shown(app)


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


async def test_new_resets_the_totals_and_undo_keeps_them(make_app: AppFactory) -> None:
    app = make_app([reply(), reply()])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "one")
        await settled(app)
        await type_and_send(pilot, "/undo")
        await until(pilot, lambda: any("Undid" in n for n in _notices(app)))
        assert app.agent.usage.calls == 1
        app.query_one(ChatInput).clear()  # /undo puts the prompt back in the box
        await type_and_send(pilot, "/new")
        await settled(app)
        assert app.agent.usage.calls == 0


def test_help_lists_usage() -> None:
    assert "/usage" in HELP
