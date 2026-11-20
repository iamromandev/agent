"""The pure part of compaction: the token estimate, turn boundaries, pruning and the summary input."""

from __future__ import annotations

import dataclasses
import json

from src.engine.compaction import (
    CHARS_PER_TOKEN,
    OMITTED_NOTE,
    SUMMARIZER_INSTRUCTIONS,
    SUMMARY_PREFIX,
    SUMMARY_TRUNCATED,
    cap_summary,
    message_tokens,
    prune,
    summary_input,
)
from src.engine.provider.base import Message, ToolCall, ToolSpec
from src.engine.session import ToolMeta

SYSTEM = Message("system", "x" * 40)  # 10 tokens + overhead
TOOLS = [
    ToolSpec("read", "Read a file.", {"type": "object", "properties": {"path": {"type": "string"}}}),
    ToolSpec("glob", "Find files.", {"type": "object", "properties": {"pattern": {"type": "string"}}}),
]


def spec_tokens(tools: list[ToolSpec]) -> int:
    return len(json.dumps([dataclasses.asdict(t) for t in tools], separators=(",", ":"))) // CHARS_PER_TOKEN


# --- turns ------------------------------------------------------------------------------------


def turns(n: int) -> list[Message]:
    """n user turns of two messages each: user `u<i>`, assistant `a<i>`."""
    return [m for i in range(1, n + 1) for m in (Message("user", f"u{i}"), Message("assistant", f"a{i}"))]


def summary(text: str = "stuff") -> Message:
    return Message("user", SUMMARY_PREFIX + text)


# --- pruning ----------------------------------------------------------------------------------


def tool_turn(n: int, output: str = "x" * 400) -> list[Message]:
    """user u<n>, an assistant call to read, its result, a final assistant message."""
    call = ToolCall(f"c{n}", "read", {"path": f"src/{n}.py"})
    return [
        Message("user", f"u{n}"),
        Message("assistant", "", [call]),
        Message("tool", output, tool_call_id=f"c{n}", name="read"),
        Message("assistant", f"a{n}"),
    ]


def meta_for(*numbers: int) -> dict[str, ToolMeta]:
    return {f"c{n}": ToolMeta(f"read src/{n}.py", f"{n}0 lines") for n in numbers}


def test_pruning_twice_changes_nothing_the_second_time() -> None:
    history = [m for n in (1, 2, 3, 4) for m in tool_turn(n)]
    once, first = prune(history, meta_for(1, 2, 3, 4), 2)

    twice, second = prune(once, meta_for(1, 2, 3, 4), 2)

    assert (first, second) == (2, 0)
    assert twice == once


# --- the summary input ------------------------------------------------------------------------


def test_the_summary_input_keeps_the_most_recent_messages_that_fit_the_budget() -> None:
    older = [Message("user", "old " * 50), Message("assistant", "mid " * 50), Message("user", "new")]
    budget = message_tokens(Message("user", "USER: new")) + 5

    msgs, omitted = summary_input(older, budget)

    assert omitted is True
    assert msgs[1].content == "USER: new"
    assert msgs[0].content == SUMMARIZER_INSTRUCTIONS + OMITTED_NOTE


def test_a_newest_message_bigger_than_the_budget_is_cut_to_fit() -> None:
    older = [Message("tool", "q" * 10_000, tool_call_id="c1", name="read")]

    msgs, omitted = summary_input(older, 100)

    assert omitted is True
    assert 0 < len(msgs[1].content) <= 100 * CHARS_PER_TOKEN


# --- the summary cap ----------------------------------------------------------------------------


def test_the_cap_grows_with_the_window() -> None:
    text = "w " * 5000

    assert len(cap_summary(text, 8192)) > len(cap_summary(text, 1000))
    assert len(cap_summary(text, 8192)) <= int(8192 * 0.2) * CHARS_PER_TOKEN + len(SUMMARY_TRUNCATED)
