"""The queue panel's lines (no app); the queue itself is tested in `apps/core/tests/host/test_queue.py`."""

from __future__ import annotations

import time

from core.host.queue import MessageQueue, Queued
from src.tui.queue import panel_lines


def _q(text: str) -> Queued:
    return Queued(text, text, None)


def test_panel_lines_show_five_then_the_count() -> None:
    lines = panel_lines([_q(str(n)) for n in range(7)])
    assert lines == [f"{n}" for n in range(5)] + ["and 2 more"]


def test_panel_lines_cut_the_first_line_to_80_characters_with_an_ellipsis() -> None:
    assert panel_lines([_q("x" * 100)]) == ["x" * 80 + "…"]
    assert panel_lines([_q("x" * 80)]) == ["x" * 80]


def test_panel_lines_show_only_the_first_line_of_a_multi_line_entry() -> None:
    assert panel_lines([_q("one\ntwo")]) == ["one…"]


def test_panel_lines_escape_control_and_bidi_characters() -> None:
    (line,) = panel_lines([_q("a\x1b[2Jb‮c")])
    assert "\x1b" not in line and "‮" not in line
    assert line.startswith("a")


def test_a_hundred_large_entries_are_quick() -> None:
    queue = MessageQueue()
    start = time.perf_counter()
    for _ in range(100):
        queue.add(_q("y" * 10_000))
    lines = panel_lines(queue.entries)
    assert time.perf_counter() - start < 0.05
    assert lines[-1] == "and 95 more"
