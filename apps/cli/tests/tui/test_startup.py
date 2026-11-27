from __future__ import annotations

from core.engine.provider.base import Done, TextDelta
from core.testing.fakes import Gate
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import NoticeEntry, UserEntry

from tests.tui.conftest import AppFactory, reached, scripted, settled, type_and_send, until

UNREACHABLE = "Cannot reach Ollama at http://localhost:11434 — start it with `ollama serve`"


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


async def test_a_message_not_sent_because_the_recheck_failed_stays_in_the_input(make_app: AppFactory) -> None:
    app = make_app([[TextDelta("hi"), Done(1, 1)]], check_result="Model fake is not pulled")
    async with app.run_test() as pilot:
        await until(pilot, lambda: bool(app.query_one(StatusBar).error))

        await type_and_send(pilot, "one")
        await settled(app)
        await pilot.pause()

        assert app.query_one(ChatInput).text == "one"
        assert list(app.query(UserEntry)) == []
        assert scripted(app).requests == []


async def test_a_message_cancelled_during_the_recheck_stays_in_the_input(make_app: AppFactory) -> None:
    app = make_app([[TextDelta("hi"), Done(1, 1)]], check_result="Model fake is not pulled")
    async with app.run_test() as pilot:
        await until(pilot, lambda: bool(app.query_one(StatusBar).error))
        gate = Gate()
        scripted(app).check_gate = gate

        await type_and_send(pilot, "one")
        await reached(gate)
        await pilot.press("escape")
        await until(pilot, lambda: not app.query_one(StatusBar).busy)

        assert app.query_one(ChatInput).text == "one"
        assert list(app.query(UserEntry)) == []
