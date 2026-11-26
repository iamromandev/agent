"""`hint_text`: the exact wording for each state, precedence, no terminals, and speed."""

from __future__ import annotations

import itertools
import time

from src.tui.hints import hint_text

IDLE = "enter send · ctrl+j newline · tab mode · / commands · @ files · ctrl+t terminal"
BUSY = "esc interrupt · enter queue message · tab mode · ctrl+t terminal"
SUGGESTIONS = "↑↓ select · tab complete · esc close"
TERMINAL = "ctrl+t chat · ctrl+pageup/pagedown tabs"


def text(*, busy=False, popup=False, terminal=False, terminals=True) -> str:
    return hint_text(busy=busy, popup_open=popup, terminal_focused=terminal, has_terminals=terminals)


def test_each_state_reads_its_exact_text() -> None:
    assert text() == IDLE
    assert text(busy=True) == BUSY
    assert text(popup=True) == SUGGESTIONS
    assert text(terminal=True) == TERMINAL


def test_without_terminals_idle_and_busy_drop_the_ctrl_t_entry() -> None:
    assert text(terminals=False) == "enter send · ctrl+j newline · tab mode · / commands · @ files"
    assert text(busy=True, terminals=False) == "esc interrupt · enter queue message · tab mode"


def test_terminal_beats_popup_and_busy() -> None:
    assert text(terminal=True, popup=True) == TERMINAL
    assert text(terminal=True, busy=True) == TERMINAL
    assert text(terminal=True, popup=True, busy=True) == TERMINAL


def test_popup_beats_busy() -> None:
    assert text(popup=True, busy=True) == SUGGESTIONS


def test_a_focused_terminal_without_a_manager_falls_through() -> None:
    assert text(terminal=True, terminals=False) == text(terminals=False)
    assert text(terminal=True, popup=True, terminals=False) == SUGGESTIONS
    assert text(terminal=True, busy=True, terminals=False) == text(busy=True, terminals=False)


def test_ten_thousand_calls_take_under_a_second() -> None:
    combos = list(itertools.product([False, True], repeat=4))
    start = time.perf_counter()
    for i in range(10_000):
        b, p, t, h = combos[i % 16]
        hint_text(busy=b, popup_open=p, terminal_focused=t, has_terminals=h)
    assert time.perf_counter() - start < 1.0
