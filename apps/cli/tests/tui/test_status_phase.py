"""The status bar's phase line: pure formatting, the clock-driven timer, token rate and bounded redraws."""

from __future__ import annotations

import pytest
from src.engine.events import Phase
from src.tui.widgets.status import MAX_ACTIVITY, StatusBar, format_elapsed, format_phase, format_phase_text
from textual.app import App, ComposeResult


class Harness(App[None]):
    def compose(self) -> ComposeResult:
        yield StatusBar()


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def activity(bar: StatusBar) -> str:
    return bar.line.split(" · ", 4)[-1]


@pytest.mark.parametrize(("seconds", "text"), [(0, "0s"), (0.9, "0s"), (59, "59s"), (60, "1:00"), (75, "1:15"), (723, "12:03")])
def test_format_elapsed(seconds: float, text: str) -> None:
    assert format_elapsed(seconds) == text


def test_format_phase_plain() -> None:
    assert format_phase("running", "bash", 2, 50, 12, None, None) == "● running bash · step 2/50 · 12s"


def test_format_phase_texts_per_kind() -> None:
    assert format_phase_text("waiting", "") == "waiting for model"
    assert format_phase_text("writing", "") == "writing reply"
    assert format_phase_text("approval", "write") == "awaiting approval: write"
    assert format_phase_text("hook", "Stop") == "running hook Stop"


def test_format_phase_with_tokens_and_rate() -> None:
    assert format_phase("writing", "", 1, 50, 2, 48, 21) == "● writing reply · step 1/50 · 2s · 48 tok · 21 tok/s"
    assert format_phase("writing", "", 1, 50, 0, 48, None) == "● writing reply · step 1/50 · 0s · 48 tok"


def test_format_phase_escapes_a_control_character() -> None:
    text = format_phase("running", "mcp__a\x1b[2J", 1, 50, 0, None, None)

    assert "\x1b" not in text
    assert "\\x1b" in text


def test_format_phase_cuts_a_long_name() -> None:
    assert len(format_phase("running", "x" * 300, 1, 50, 0, None, None)) <= MAX_ACTIVITY


def test_markup_in_a_tool_name_shows_literally() -> None:
    assert "[bold]" in format_phase("running", "[bold]x", 1, 50, 0, None, None)


async def test_the_timer_advances_without_events() -> None:
    clock = Clock()
    async with Harness().run_test() as pilot:
        bar = pilot.app.query_one(StatusBar)
        bar.clock = clock
        bar.busy = True
        bar.set_phase(Phase("waiting", 1, 50))
        assert activity(bar) == "● waiting for model · step 1/50 · 0s"
        for second in (1, 2):
            clock.now = 100.0 + second
            bar._tick()
            assert activity(bar).endswith(f"· {second}s")
        clock.now = 175.0
        bar._tick()
        assert activity(bar).endswith("· 1:15")


async def test_the_phase_line_never_says_working() -> None:
    async with Harness().run_test() as pilot:
        bar = pilot.app.query_one(StatusBar)
        bar.busy = True
        assert activity(bar) == "● working"
        bar.set_phase(Phase("running", 1, 50, "bash"))
        assert "working" not in bar.line


async def test_tokens_grow_and_a_rate_shows_after_half_a_second() -> None:
    clock = Clock()
    async with Harness().run_test() as pilot:
        bar = pilot.app.query_one(StatusBar)
        bar.clock = clock
        bar.busy = True
        bar.set_phase(Phase("waiting", 1, 50))
        bar.set_phase(Phase("writing", 1, 50))
        bar.add_reply_chars(80)
        clock.now += 0.2
        bar._tick()
        assert activity(bar).endswith("· 20 tok")
        bar.add_reply_chars(160)
        clock.now += 1.8
        bar._tick()
        assert activity(bar).endswith("· 60 tok · 30 tok/s")


async def test_the_real_count_replaces_the_estimate() -> None:
    clock = Clock()
    async with Harness().run_test() as pilot:
        bar = pilot.app.query_one(StatusBar)
        bar.clock = clock
        bar.busy = True
        bar.set_phase(Phase("writing", 1, 50))
        bar.add_reply_chars(40)
        bar.set_reply_tokens(137)
        bar._tick()
        assert "137 tok" in bar.line


async def test_a_new_request_resets_the_counts() -> None:
    async with Harness().run_test() as pilot:
        bar = pilot.app.query_one(StatusBar)
        bar.busy = True
        bar.set_phase(Phase("writing", 1, 50))
        bar.add_reply_chars(400)
        bar.set_reply_tokens(99)
        bar.set_phase(Phase("waiting", 2, 50))
        bar.set_phase(Phase("writing", 2, 50))
        assert " tok" in bar.line
        assert "99 tok" not in bar.line
        assert "0 tok" in bar.line


async def test_no_tokens_outside_the_writing_phase() -> None:
    async with Harness().run_test() as pilot:
        bar = pilot.app.query_one(StatusBar)
        bar.busy = True
        bar.set_phase(Phase("running", 1, 50, "bash"))
        bar.add_reply_chars(400)
        bar._tick()
        assert " tok" not in bar.line


async def test_a_thousand_chunks_cause_no_redraw_until_a_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    async with Harness().run_test() as pilot:
        bar = pilot.app.query_one(StatusBar)
        bar.clock = clock
        bar.busy = True
        bar.set_phase(Phase("writing", 1, 50))
        calls: list[str] = []
        monkeypatch.setattr(bar, "update", lambda text="", **_: calls.append(str(text)))

        for _ in range(1000):
            bar.add_reply_chars(4)
        assert calls == []

        for _ in range(8):  # eight ticks over 2 simulated seconds
            clock.now += 0.25
            bar._tick()
        assert len(calls) <= 8
        bar._tick()  # nothing changed: no redraw
        assert len(calls) <= 8


async def test_progress_and_compacting_win_over_the_phase_line() -> None:
    async with Harness().run_test() as pilot:
        bar = pilot.app.query_one(StatusBar)
        bar.busy = True
        bar.set_phase(Phase("running", 1, 50, "generate_image"))
        bar.progress = "step 2/9"
        assert "◐ step 2/9" in bar.line
        assert "●" not in bar.line
        bar.compacting = True
        assert "◐ compacting…" in bar.line
        bar.compacting = False
        bar.progress = ""
        assert "● running generate_image · step 1/50" in bar.line


async def test_clear_phase_and_idle_remove_the_dot() -> None:
    async with Harness().run_test() as pilot:
        bar = pilot.app.query_one(StatusBar)
        bar.busy = True
        bar.set_phase(Phase("running", 1, 50, "bash"))
        bar.clear_phase()
        assert "●" in bar.line  # still busy: back to today's line
        assert "● working" in bar.line
        bar.set_phase(Phase("running", 1, 50, "bash"))
        bar.busy = False
        assert "●" not in bar.line
        bar.busy = True
        assert "● working" in bar.line


async def test_a_long_activity_stays_within_the_limit_in_the_bar() -> None:
    async with Harness().run_test() as pilot:
        bar = pilot.app.query_one(StatusBar)
        bar.busy = True
        bar.set_phase(Phase("running", 1, 50, "n" * 300))
        assert len(activity(bar)) <= MAX_ACTIVITY
