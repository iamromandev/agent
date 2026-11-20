"""The message queue model: entries, merging and the panel's lines (no app)."""

from __future__ import annotations

import time

from src.tui.queue import MessageQueue, Queued, merge, panel_lines


def _q(text: str) -> Queued:
    return Queued(text, text, None)


def test_take_returns_all_entries_in_order_and_empties_the_queue() -> None:
    queue = MessageQueue()
    assert not queue
    queue.add(_q("a"))
    queue.add(_q("b"))
    assert queue and [e.typed for e in queue.entries] == ["a", "b"]
    assert [e.typed for e in queue.take()] == ["a", "b"]
    assert not queue and queue.entries == ()


def test_merge_joins_prompts_with_a_blank_line_and_drafts_with_a_newline() -> None:
    merged = merge([_q("a"), Queued("/review x", "expanded", "/review (global)")])
    assert merged.prompt == "a\n\nexpanded"
    assert merged.draft == "a\n/review x"
    assert merged.label is None


def test_merge_keeps_the_label_of_a_single_command() -> None:
    merged = merge([Queued("/review x", "expanded", "/review (global)")])
    assert merged.prompt == "expanded" and merged.draft == "/review x" and merged.label == "/review (global)"


def test_merge_of_one_plain_message_has_no_label() -> None:
    assert merge([_q("a")]).label is None


def test_panel_lines_show_five_then_the_count() -> None:
    lines = panel_lines([_q(str(n)) for n in range(7)])
    assert lines == [f"queued: {n}" for n in range(5)] + ["and 2 more"]


def test_panel_lines_cut_the_first_line_to_80_characters_with_an_ellipsis() -> None:
    assert panel_lines([_q("x" * 100)]) == ["queued: " + "x" * 80 + "…"]
    assert panel_lines([_q("x" * 80)]) == ["queued: " + "x" * 80]


def test_panel_lines_show_only_the_first_line_of_a_multi_line_entry() -> None:
    assert panel_lines([_q("one\ntwo")]) == ["queued: one…"]


def test_panel_lines_escape_control_and_bidi_characters() -> None:
    (line,) = panel_lines([_q("a\x1b[2Jb‮c")])
    assert "\x1b" not in line and "‮" not in line
    assert line.startswith("queued: a")


def test_a_hundred_large_entries_are_quick() -> None:
    queue = MessageQueue()
    start = time.perf_counter()
    for _ in range(100):
        queue.add(_q("y" * 10_000))
    lines = panel_lines(queue.entries)
    assert time.perf_counter() - start < 0.05
    assert lines[-1] == "and 95 more"
