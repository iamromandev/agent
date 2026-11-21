"""generate_image in the app: the input stays usable during a slow run and Esc stops it within a second (AC13, NFR4)."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest
from src.engine.guardrails.setup import Guardrails
from src.engine.imagegen import MfluxGenerator
from src.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from src.tui.screens.approval import ApprovalScreen
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.status import StatusBar

from tests.image_fakes import fake_mflux
from tests.tui.conftest import AppFactory, choose, type_and_send, until


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


async def test_the_input_stays_usable_and_escape_cancels_a_running_generation(
    make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pids = tmp_path / "pids"
    monkeypatch.setenv("FAKE_MFLUX_MODE", "sleep")
    monkeypatch.setenv("FAKE_MFLUX_PIDS", str(pids))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    folder = tmp_path / "bin"
    folder.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    generator = MfluxGenerator(fake_mflux(folder), "x/y", 60, Guardrails.off(workspace))
    call = ToolCallEvent(ToolCall("c1", "generate_image", {"prompt": "p", "path": "a.png", "width": 256, "height": 256}))
    app = make_app([[call, Done(1, 1)], [TextDelta("ok"), Done(1, 1)]], workspace=workspace, imager=generator, context_window=4000)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await choose(pilot, "once")
        await until(pilot, lambda: pids.exists() and len(pids.read_text().split()) == 2)
        pid_list = [int(p) for p in pids.read_text().split()]

        await pilot.press("a", "b", "c")
        await pilot.pause()
        assert app.query_one(ChatInput).text == "abc"

        started = time.monotonic()
        await pilot.press("escape")
        await until(pilot, lambda: not app.query_one(StatusBar).busy)
        assert time.monotonic() - started < 1.5

        deadline = time.monotonic() + 2
        while not all(_gone(p) for p in pid_list):
            assert time.monotonic() < deadline, "mflux group still alive"
            await asyncio.sleep(0.02)
        assert not (workspace / "a.png").exists()
        results = [m.content for m in app.agent.history if m.role == "tool"]
        assert results == ["Interrupted by user"]  # the loop's own wording for an Esc during a call
