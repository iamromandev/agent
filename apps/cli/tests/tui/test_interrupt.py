from __future__ import annotations

import os

import pytest
from core.engine.provider.base import Done, TextDelta
from core.testing.fakes import Gate
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import AssistantEntry, NoticeEntry, Transcript, UserEntry
from textual.pilot import Pilot

from tests.tui.conftest import AppFactory, reached, settled, type_and_send, until


async def test_a_second_escape_while_the_queue_drains_loses_no_events(
    make_app: AppFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider_gate, write_gate = Gate(), Gate()
    app = make_app(
        [
            [TextDelta("a"), TextDelta("b"), TextDelta("c"), provider_gate, Done(1, 1)],
            [TextDelta("next"), Done(1, 1)],
        ],
        context_window=2000,  # the prompt and tool specs alone are about 1,150 tokens
    )
    original = Transcript.write_assistant
    first = True

    async def slow_write(self: Transcript, delta: str) -> None:
        nonlocal first
        if first:  # the consumer stops here while the rest of the reply waits in the queue
            first = False
            write_gate.reached.set()
            await write_gate.release.wait()
        await original(self, delta)

    monkeypatch.setattr(Transcript, "write_assistant", slow_write)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await reached(write_gate)
        await reached(provider_gate)
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        write_gate.release.set()
        await until(pilot, lambda: not app.query_one(StatusBar).busy)

        assert "".join(e.text for e in app.query(AssistantEntry)) == "abc"
        assert any("interrupted" in e.text for e in app.query(NoticeEntry))
        assert not app.query_one(StatusBar).busy

        await type_and_send(pilot, "again")
        await settled(app)
        await pilot.pause()

        kinds = [type(e).__name__ for e in app.query("#transcript > *")]
        assert kinds == ["UserEntry", "AssistantEntry", "NoticeEntry", "UserEntry", "AssistantEntry"]
        assert [e.text for e in app.query(AssistantEntry)] == ["abc", "next"]
        assert [e.text for e in app.query(UserEntry)] == ["go", "again"]


async def test_an_event_that_fails_to_display_does_not_drop_the_events_behind_it(
    make_app: AppFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app([[TextDelta("a"), TextDelta("boom"), TextDelta("c"), Done(1, 1)]])
    original = Transcript.write_assistant

    async def flaky_write(self: Transcript, delta: str) -> None:
        if delta == "boom":
            raise RuntimeError("cannot draw")
        await original(self, delta)

    monkeypatch.setattr(Transcript, "write_assistant", flaky_write)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await settled(app)
        await pilot.pause()

        assert "".join(e.text for e in app.query(AssistantEntry)) == "ac"
        assert any("cannot draw" in e.text and e.text.startswith("✗") for e in app.query(NoticeEntry))
        assert not app.query_one(StatusBar).busy


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _quit_with(pilot: Pilot, how: str) -> None:
    if how == "/quit":
        await type_and_send(pilot, "/quit")
    else:
        await pilot.press(how)


@pytest.mark.parametrize("how", ["ctrl+q"])
async def test_quitting_while_a_turn_streams_exits_cleanly_and_closes_the_client(
    make_app: AppFactory, how: str
) -> None:
    gate = Gate()
    closed: list[bool] = []

    async def on_close() -> None:
        closed.append(True)

    app = make_app([[TextDelta("a"), gate, Done(1, 1)]], on_close=on_close)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await reached(gate)
        await _quit_with(pilot, how)
        await until(pilot, lambda: not app.is_running)

    assert app.return_code in (0, None)
    assert closed == [True]
