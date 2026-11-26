"""The `Instructions:` notice in the transcript (R9, R10, AC10, AC11)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from src.engine.loop import Agent
from src.engine.provider.base import Done, TextDelta
from src.tui.widgets.transcript import NoticeEntry

from tests.tui.conftest import AppFactory, settled, type_and_send

REPLY = [TextDelta("ok"), Done(1, 1)]


def _instructions(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry) if "Instructions:" in e.text]


class LoadGate:
    """Holds the first AGENTS.md load (the startup announcement's thread) until the test lets it go."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.reached = threading.Event()
        self.release = threading.Event()
        self._armed = True
        real = Agent.instructions
        gate = self

        def gated(agent: Agent):
            if gate._armed:
                gate._armed = False
                gate.reached.set()
                assert gate.release.wait(5.0), "the test never released the load"
            return real(agent)

        monkeypatch.setattr(Agent, "instructions", gated)

    def let_go(self) -> None:
        self.release.set()


async def test_a_notice_that_failed_to_draw_is_announced_again(
    make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.md").write_text("rules")
    from src.tui.widgets.transcript import Transcript

    real = Transcript.add_notice
    failed: list[int] = []

    async def flaky(self, level, text):
        if text.startswith("Instructions:") and not failed:
            failed.append(1)
            raise RuntimeError("draw failed")
        return await real(self, level, text)

    monkeypatch.setattr(Transcript, "add_notice", flaky)
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "hi")
        await settled(app)

        assert failed and len(_instructions(app)) == 1
