"""The `Instructions:` notice in the transcript (R9, R10, AC10, AC11)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from src.agent.loop import Agent
from src.agent.provider.base import Done, TextDelta
from src.tui import StartChoice
from src.tui.styles import ICONS
from src.tui.widgets.transcript import NoticeEntry

from tests.tui.conftest import AppFactory, chat, settled, type_and_send, until

REPLY = [TextDelta("ok"), Done(1, 1)]


def _instructions(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry) if "Instructions:" in e.text]


async def test_one_file_is_announced_at_startup(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("rules")
    app = make_app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: bool(_instructions(app)))
        await settled(app)

        (text,) = _instructions(app)
        assert text.startswith(f"{ICONS['info']} Instructions: AGENTS.md (~2) · ~")


async def test_a_cut_file_is_a_warning(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("x" * 5000)
    app = make_app()  # window 1000: a budget of 1000 characters
    async with app.run_test() as pilot:
        await until(pilot, lambda: bool(_instructions(app)))

        (text,) = _instructions(app)
        assert text.startswith(ICONS["warn"])
        assert "cut from ~1250" in text


async def test_no_file_no_notice(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()

        assert _instructions(app) == []


async def test_the_notice_follows_a_continue_redraw(make_app: AppFactory, save_session, tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("rules")
    save_session(chat("old question", "old answer"))
    app = make_app(start=StartChoice("continue"))
    async with app.run_test() as pilot:
        await until(pilot, lambda: bool(_instructions(app)))
        await settled(app)

        entries = [e.text for e in app.query(NoticeEntry)]
        resumed = next(i for i, t in enumerate(entries) if "Resumed" in t)
        assert next(i for i, t in enumerate(entries) if "Instructions:" in t) > resumed


async def test_a_message_sent_at_once_shows_the_notice_only_once(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("rules")
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "hi")
        await settled(app)
        await pilot.pause()

        assert len(_instructions(app)) == 1


async def test_a_file_created_mid_session_is_announced_on_the_next_turn(make_app: AppFactory, tmp_path: Path) -> None:
    app = make_app([REPLY, REPLY, REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "one")
        await settled(app)
        assert _instructions(app) == []

        (tmp_path / "AGENTS.md").write_text("rules")
        await type_and_send(pilot, "two")
        await settled(app)
        assert len(_instructions(app)) == 1

        await type_and_send(pilot, "three")
        await settled(app)
        assert len(_instructions(app)) == 1


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


async def test_a_message_during_the_startup_announcement_is_refused_and_the_notice_shows_once(
    make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.md").write_text("rules")
    gate = LoadGate(monkeypatch)
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await until(pilot, gate.reached.is_set)
        try:
            await type_and_send(pilot, "hi")
            await until(pilot, lambda: any("busy" in e.text for e in app.query(NoticeEntry)))
            assert any("a session is loading" in e.text for e in app.query(NoticeEntry))
            assert app.agent.history == []
        finally:
            gate.let_go()
        await settled(app)
        await until(pilot, lambda: bool(_instructions(app)))
        await type_and_send(pilot, "hi")
        await settled(app)

        assert len(_instructions(app)) == 1


async def test_the_notice_is_back_after_new_and_clear(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("rules")
    app = make_app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: bool(_instructions(app)))
        await settled(app)
        for command in ("/new", "/clear"):
            await type_and_send(pilot, command)
            await settled(app)
            await until(pilot, lambda: len(_instructions(app)) == 1)
            assert len(_instructions(app)) == 1, command
            assert app._switching is False


async def test_the_notice_is_back_after_a_sessions_switch(
    make_app: AppFactory, save_session, tmp_path: Path
) -> None:
    (tmp_path / "AGENTS.md").write_text("rules")
    save_session(chat("old question", "old answer"))
    app = make_app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: bool(_instructions(app)))
        await settled(app)
        await type_and_send(pilot, "/sessions")
        await pilot.pause(0.2)
        await pilot.press("enter")
        await settled(app)
        await until(pilot, lambda: any("Resumed" in e.text for e in app.query(NoticeEntry)))
        await settled(app)

        assert len(_instructions(app)) == 1
        assert app._switching is False


async def test_the_notice_shows_once_after_a_continue_resume(
    make_app: AppFactory, save_session, tmp_path: Path
) -> None:
    (tmp_path / "AGENTS.md").write_text("rules")
    save_session(chat("old question", "old answer"))
    app = make_app(start=StartChoice("continue"))
    async with app.run_test() as pilot:
        await until(pilot, lambda: bool(_instructions(app)))
        await settled(app)
        await pilot.pause()

        assert len(_instructions(app)) == 1


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
