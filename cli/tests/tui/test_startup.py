from __future__ import annotations

from src.agent.provider.base import Done, TextDelta
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import AssistantEntry, NoticeEntry, UserEntry

from tests.fakes import Gate
from tests.tui.conftest import AppFactory, reached, scripted, settled, type_and_send, until

UNREACHABLE = "Cannot reach Ollama at http://localhost:11434 — start it with `ollama serve`"


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


async def test_a_failing_check_shows_the_message_and_keeps_the_app_open(make_app: AppFactory) -> None:
    app = make_app(check_result=UNREACHABLE)
    async with app.run_test() as pilot:
        await until(pilot, lambda: bool(_notices(app)))

        assert UNREACHABLE in _notices(app)[0]
        assert _notices(app)[0].startswith("✗")
        assert app.query_one(StatusBar).error
        assert app.is_running


async def test_a_passing_check_shows_nothing(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()

        assert _notices(app) == []
        assert not app.query_one(StatusBar).error


async def test_submit_rechecks_and_runs_the_turn_once_the_problem_is_fixed(make_app: AppFactory) -> None:
    app = make_app([[TextDelta("hi"), Done(1, 1)]], check_result="Model fake is not pulled — run `ollama pull fake`")
    async with app.run_test() as pilot:
        await until(pilot, lambda: bool(app.query_one(StatusBar).error))
        notices_at_start = len(_notices(app))

        await type_and_send(pilot, "one")
        await settled(app)
        await pilot.pause()
        assert scripted(app).requests == []  # still failing: no turn
        assert len(_notices(app)) == notices_at_start + 1
        assert app.query_one(StatusBar).error

        scripted(app).check_result = None
        await type_and_send(pilot, "two")
        await settled(app)
        await pilot.pause()

        assert not app.query_one(StatusBar).error
        assert [e.text for e in app.query(AssistantEntry)] == ["hi"]
        assert len(scripted(app).requests) == 1


async def test_escape_during_the_recheck_leaves_the_app_ready_for_the_next_message(make_app: AppFactory) -> None:
    app = make_app([[TextDelta("hi"), Done(1, 1)]], check_result="Model fake is not pulled")
    async with app.run_test() as pilot:
        await until(pilot, lambda: bool(app.query_one(StatusBar).error))
        gate = Gate()
        scripted(app).check_gate = gate

        await type_and_send(pilot, "one")
        await reached(gate)
        await pilot.press("escape")
        await settled(app)
        await pilot.pause()
        assert not app.query_one(StatusBar).busy

        scripted(app).check_gate = None
        scripted(app).check_result = None
        await type_and_send(pilot, "two")
        await settled(app)
        await pilot.pause()

        assert [e.text for e in app.query(AssistantEntry)] == ["hi"]
        assert not app.query_one(StatusBar).busy


async def test_a_turn_cancelled_before_it_starts_does_not_leave_the_app_busy(make_app: AppFactory) -> None:
    app = make_app([[TextDelta("hi"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await settled(app)
        # no await between the submit and the cancel: the worker has not run its first line
        await app.on_chat_input_submitted(ChatInput.Submitted("one"))
        app.interrupt_turn()
        await until(pilot, lambda: not app.query_one(StatusBar).busy)

        await type_and_send(pilot, "two")
        await settled(app)
        await pilot.pause()

        assert [e.text for e in app.query(AssistantEntry)] == ["hi"]


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


async def test_a_message_that_is_sent_is_not_restored(make_app: AppFactory) -> None:
    app = make_app([[TextDelta("hi"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "one")
        await settled(app)
        await pilot.pause()

        assert app.query_one(ChatInput).text == ""
