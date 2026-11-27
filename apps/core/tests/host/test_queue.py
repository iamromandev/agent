"""The message queue model: entries and merging (no UI)."""

from __future__ import annotations

from core.host.queue import MessageQueue, Queued, merge


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
