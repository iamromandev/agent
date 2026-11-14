"""NFR2 / AC14: a terminal printing 10 MB as fast as it can must not make the chat input or the app sluggish."""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import Any

import pytest
from src.agent.terminal.manager import Terminal, TerminalManager
from src.tui.app import AgentApp
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.terminal import TerminalPane, TerminalView

from tests.tui.conftest import AppFactory, until

FLOOD = "yes 0123456789abcdefghijklmnopqrstuvwxyz0123456789 | head -c 10000000; echo; echo FLOOD-END; sleep 30"
KEY_LIMIT = 0.2  # seconds from typing a key to seeing it in the chat input
REDRAWS_PER_SECOND = 30
WINDOW = 1.0  # seconds over which the redraws are counted
SAMPLES = 30  # keys typed while the flood runs
SLACK = 1  # a timer tick may land at either edge of the window


async def _flooded(app: AgentApp, pilot, manager: TerminalManager) -> tuple[Terminal, TerminalView]:
    """Show the pane with the chat input focused, then start the flood: it is under way when this returns."""
    manager.start("idle", "sleep 30")
    await pilot.press("ctrl+t")
    await until(pilot, lambda: isinstance(app.focused, TerminalView))
    await pilot.press("ctrl+backslash")
    assert isinstance(app.focused, ChatInput)
    flood = manager.start("flood", FLOOD)
    view = app.query_one(TerminalPane).view("flood")
    assert view is not None
    return flood, view


def _flooding(flood: Terminal) -> bool:
    """The process is still printing (what it prints last has not been read yet)."""
    return not any("FLOOD-END" in line for line in flood.buffer.tail(5))


def _on_screen(view: TerminalView) -> bool:
    return any("FLOOD-END" in line for line in view.emulator.display)


async def _typed(app: AgentApp, chat: ChatInput) -> float:
    """Seconds from a key reaching the app to its character being in the chat input."""
    before = len(chat.text)
    start = time.perf_counter()
    app.simulate_key("x")  # not `pilot.press`: that also waits for the app to be idle, which is not the latency
    while len(chat.text) == before:
        assert time.perf_counter() - start < 5, "the key never reached the chat input"
        await asyncio.sleep(0.001)
    return time.perf_counter() - start


async def test_a_key_typed_into_the_chat_input_appears_within_200_ms_during_a_10_mb_flood(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        flood, _ = await _flooded(app, pilot, terminal_manager)
        chat = app.query_one(ChatInput)
        timings: list[float] = []
        while _flooding(flood) and len(timings) < SAMPLES:
            timings.append(await _typed(app, chat))
            await asyncio.sleep(0.005)  # a person types slower than this: the flood gets its turn between keys

        assert len(timings) >= 3, f"the flood was over after {len(timings)} keys: it proves nothing"
        assert statistics.median(timings) < KEY_LIMIT, timings  # not the worst: one slow tick on a busy CI machine


async def test_a_flooded_terminal_is_redrawn_at_most_30_times_a_second(
    make_app: AppFactory, terminal_manager: TerminalManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    refreshes: list[float] = []
    original = TerminalView.refresh

    def counted(self: TerminalView, *args: Any, **kwargs: Any) -> TerminalView:
        refreshes.append(time.perf_counter())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TerminalView, "refresh", counted)
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        await _flooded(app, pilot, terminal_manager)
        refreshes.clear()
        await pilot.pause(WINDOW)
        window = list(refreshes)

    assert len(window) >= 10, len(window)  # the screen was changing the whole time: it was drawn, not stuck
    assert len(window) <= REDRAWS_PER_SECOND * WINDOW + SLACK, len(window)


async def test_the_screen_catches_up_with_the_end_of_a_flood_soon_after_it_ends(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        _, view = await _flooded(app, pilot, terminal_manager)

        await until(pilot, lambda: _on_screen(view), timeout=20)  # output may be skipped, never the end
