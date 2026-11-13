"""NFR3: drawing a finished tool entry with a lot of colored output stays fast."""

from __future__ import annotations

import time

from src.agent.tools.base import ToolResult
from src.tui.widgets.transcript import Transcript

from tests.tui.conftest import AppFactory

LIMIT_SECONDS = 0.1


async def test_a_finished_entry_with_30000_colored_characters_renders_under_100_ms(make_app: AppFactory) -> None:
    line = "\x1b[32mpassed\x1b[0m \x1b[31mFAILED\x1b[0m test_something.py::test_it\n"
    display = (line * 1000)[:30_000]
    result = ToolResult("x", False, "exit 0", display=display)
    app = make_app()
    async with app.run_test() as pilot:
        transcript = app.query_one(Transcript)
        timings: list[float] = []
        for i in range(7):
            entry = await transcript.add_tool(f"c{i}", "bash x")
            start = time.perf_counter()
            await entry.finish(result, None, False)
            await pilot.pause()
            timings.append(time.perf_counter() - start)

    assert min(timings) < LIMIT_SECONDS  # best of several: one slow run on a busy machine proves nothing
