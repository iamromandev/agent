"""The pure part of compaction: the token estimate, turn boundaries, pruning and the summary input."""

from __future__ import annotations

import dataclasses
import json

from src.agent.compaction import (
    CHARS_PER_TOKEN,
    MESSAGE_OVERHEAD,
    MIN_SUMMARY_GAIN,
    OMITTED_NOTE,
    SUMMARIZER_INSTRUCTIONS,
    SUMMARY_PREFIX,
    SUMMARY_TRUNCATED,
    Reported,
    cap_summary,
    cut_index,
    estimate_tokens,
    is_summary,
    message_tokens,
    prune,
    pruned_stub,
    render_for_summary,
    summary_input,
    turn_starts,
)
from src.agent.provider.base import Message, ToolCall, ToolSpec
from src.agent.session import ToolMeta

SYSTEM = Message("system", "x" * 40)  # 10 tokens + overhead
TOOLS = [
    ToolSpec("read", "Read a file.", {"type": "object", "properties": {"path": {"type": "string"}}}),
    ToolSpec("glob", "Find files.", {"type": "object", "properties": {"pattern": {"type": "string"}}}),
]


def spec_tokens(tools: list[ToolSpec]) -> int:
    return len(json.dumps([dataclasses.asdict(t) for t in tools], separators=(",", ":"))) // CHARS_PER_TOKEN


def test_a_message_costs_a_quarter_of_its_characters_plus_the_overhead() -> None:
    assert message_tokens(Message("user", "a" * 400)) == 100 + MESSAGE_OVERHEAD


def test_an_empty_message_costs_only_the_overhead() -> None:
    assert message_tokens(Message("assistant")) == MESSAGE_OVERHEAD


def test_tool_calls_count_by_their_json_length() -> None:
    calls = [ToolCall("c1", "read", {"path": "src/a.py"})]
    as_json = json.dumps([{"id": "c1", "name": "read", "arguments": {"path": "src/a.py"}}], ensure_ascii=False)

    assert message_tokens(Message("assistant", "ok", calls)) == (2 + len(as_json)) // 4 + MESSAGE_OVERHEAD
    assert message_tokens(Message("assistant", "ok", calls)) > message_tokens(Message("assistant", "ok"))


def test_without_a_reported_count_everything_is_estimated() -> None:
    history = [Message("user", "a" * 80), Message("assistant", "b" * 40)]

    expected = message_tokens(SYSTEM) + message_tokens(history[0]) + message_tokens(history[1])
    assert estimate_tokens(SYSTEM, history, None) == expected


def test_a_reported_count_is_extended_by_the_messages_added_after_it() -> None:
    history = [Message("user", "a" * 80), Message("assistant", "b" * 40), Message("user", "c" * 120)]

    assert estimate_tokens(SYSTEM, history, Reported(500, 2)) == 500 + message_tokens(history[2])


def test_a_reported_count_that_covers_the_whole_history_is_returned_as_is() -> None:
    history = [Message("user", "hi"), Message("assistant", "hello")]

    assert estimate_tokens(SYSTEM, history, Reported(321, 2)) == 321


def test_without_a_reported_count_the_tool_specs_are_estimated_too() -> None:
    history = [Message("user", "a" * 80)]

    expected = message_tokens(SYSTEM) + message_tokens(history[0]) + spec_tokens(TOOLS)
    assert spec_tokens(TOOLS) > 0
    assert estimate_tokens(SYSTEM, history, None, TOOLS) == expected


def test_no_tools_add_nothing_to_the_estimate() -> None:
    history = [Message("user", "a" * 80)]

    assert estimate_tokens(SYSTEM, history, None, []) == estimate_tokens(SYSTEM, history, None)


def test_a_reported_count_already_includes_the_tool_specs() -> None:
    history = [Message("user", "a" * 80), Message("assistant", "b" * 40), Message("user", "c" * 120)]

    assert estimate_tokens(SYSTEM, history, Reported(500, 2), TOOLS) == 500 + message_tokens(history[2])


# --- turns ------------------------------------------------------------------------------------


def turns(n: int) -> list[Message]:
    """n user turns of two messages each: user `u<i>`, assistant `a<i>`."""
    return [m for i in range(1, n + 1) for m in (Message("user", f"u{i}"), Message("assistant", f"a{i}"))]


def summary(text: str = "stuff") -> Message:
    return Message("user", SUMMARY_PREFIX + text)


def test_a_user_message_starting_with_the_prefix_at_the_start_is_a_summary() -> None:
    assert is_summary([summary()], 0)
    assert not is_summary([Message("user", "Summary of the earlier conversation: no newline")], 0)
    assert not is_summary([Message("assistant", SUMMARY_PREFIX + "x")], 0)


def test_a_message_starting_with_the_prefix_anywhere_but_the_start_is_not_a_summary() -> None:
    history = [*turns(1), summary()]

    assert not is_summary(history, 2)
    assert not is_summary([], 0)


def test_turn_starts_are_the_real_user_messages_not_the_summary() -> None:
    history = [summary(), *turns(2)]

    assert turn_starts(history) == [1, 3]


def test_a_real_user_message_that_starts_with_the_prefix_counts_as_a_turn() -> None:
    history = [*turns(1), summary("pasted by the user"), Message("assistant", "ok"), *turns(1)]

    assert turn_starts(history) == [0, 2, 4]
    assert cut_index(history, 2) == 2


def test_turn_starts_of_an_empty_history_is_empty() -> None:
    assert turn_starts([]) == []


def test_the_cut_is_the_start_of_the_last_keep_turns_turns() -> None:
    history = turns(5)

    assert cut_index(history, 2) == 6  # turn 4 starts at index 6
    assert cut_index(history, 1) == 8
    assert cut_index(history, 5) == 0


def test_the_cut_is_zero_with_no_more_turns_than_are_kept() -> None:
    assert cut_index(turns(2), 2) == 0
    assert cut_index(turns(1), 2) == 0
    assert cut_index([], 2) == 0


def test_the_cut_ignores_a_leading_summary() -> None:
    history = [summary(), *turns(3)]

    assert cut_index(history, 2) == 3  # turn 2 starts at index 3; the summary is not a turn
    assert cut_index([summary(), *turns(2)], 2) == 0


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


def test_a_stub_names_the_label_and_the_summary() -> None:
    assert pruned_stub(ToolMeta("read src/a.py", "120 lines"), "read") == "[output pruned: read src/a.py \u00b7 120 lines]"


def test_a_stub_without_meta_names_the_tool() -> None:
    assert pruned_stub(None, "read") == "[output pruned: read]"


def test_prune_stubs_the_tool_results_before_the_kept_turns() -> None:
    history = [m for n in (1, 2, 3, 4) for m in tool_turn(n)]

    new, count = prune(history, meta_for(1, 2, 3, 4), 2)

    assert count == 2
    assert new[2].content == "[output pruned: read src/1.py \u00b7 10 lines]"
    assert new[6].content == "[output pruned: read src/2.py \u00b7 20 lines]"
    assert new[2].tool_call_id == "c1" and new[2].name == "read" and new[2].role == "tool"


def test_prune_leaves_the_kept_turns_byte_identical() -> None:
    history = [m for n in (1, 2, 3, 4) for m in tool_turn(n)]

    new, _ = prune(history, meta_for(1, 2, 3, 4), 2)

    assert new[8:] == history[8:]
    assert all(a is b for a, b in zip(new[8:], history[8:], strict=True))


def test_prune_leaves_user_and_assistant_messages_and_tool_calls_untouched() -> None:
    history = [m for n in (1, 2, 3) for m in tool_turn(n)]

    new, _ = prune(history, meta_for(1, 2, 3), 1)

    assert [(m.role, m.content, m.tool_calls) for m in new if m.role != "tool"] == [
        (m.role, m.content, m.tool_calls) for m in history if m.role != "tool"
    ]


def test_prune_does_not_change_the_messages_it_was_given() -> None:
    history = [m for n in (1, 2, 3) for m in tool_turn(n)]
    before = [(m.role, m.content) for m in history]

    prune(history, meta_for(1, 2, 3), 1)

    assert [(m.role, m.content) for m in history] == before


def test_pruning_twice_changes_nothing_the_second_time() -> None:
    history = [m for n in (1, 2, 3, 4) for m in tool_turn(n)]
    once, first = prune(history, meta_for(1, 2, 3, 4), 2)

    twice, second = prune(once, meta_for(1, 2, 3, 4), 2)

    assert (first, second) == (2, 0)
    assert twice == once


def test_prune_leaves_a_result_alone_when_its_stub_would_not_be_shorter() -> None:
    history = [m for n in (1, 2, 3) for m in tool_turn(n, "ok" if n == 1 else "x" * 400)]

    new, count = prune(history, meta_for(1, 2, 3), 1)

    assert count == 1  # only turn 2: the "ok" of turn 1 is shorter than its stub
    assert new[2].content == "ok"
    assert new[6].content == "[output pruned: read src/2.py \u00b7 20 lines]"


def test_prune_treats_a_stub_as_long_as_the_content_as_not_worth_it() -> None:
    stub = pruned_stub(ToolMeta("read src/1.py", "10 lines"), "read")
    history = [m for n in (1, 2) for m in tool_turn(n, "y" * len(stub) if n == 1 else "x" * 400)]

    new, count = prune(history, meta_for(1, 2), 1)

    assert count == 0
    assert new[2].content == "y" * len(stub)


def test_prune_without_meta_uses_the_tool_name() -> None:
    history = [m for n in (1, 2, 3) for m in tool_turn(n)]

    new, count = prune(history, {}, 1)

    assert count == 2
    assert new[2].content == "[output pruned: read]"


def test_prune_does_nothing_when_all_turns_are_kept() -> None:
    history = [m for n in (1, 2) for m in tool_turn(n)]

    new, count = prune(history, meta_for(1, 2), 2)

    assert count == 0
    assert new == history


def test_prune_keeps_a_leading_summary_as_is() -> None:
    history = [summary(), *[m for n in (1, 2, 3) for m in tool_turn(n)]]

    new, count = prune(history, meta_for(1, 2, 3), 1)

    assert new[0] == summary()
    assert count == 2


# --- the summary input ------------------------------------------------------------------------


def test_each_role_is_rendered_with_its_label() -> None:
    call = ToolCall("c1", "read", {"path": "src/a.py"})

    assert render_for_summary(Message("user", "fix it")) == "USER: fix it"
    assert render_for_summary(Message("assistant", "ok")) == "ASSISTANT: ok"
    assert render_for_summary(Message("assistant", "looking", [call])) == (
        "ASSISTANT: looking [called read(path='src/a.py')]"
    )
    assert render_for_summary(Message("tool", "print(1)", tool_call_id="c1", name="read")) == "TOOL read: print(1)"


def test_a_long_call_argument_is_shortened_in_the_rendering() -> None:
    call = ToolCall("c1", "write", {"path": "a.py", "content": "z" * 5000})

    rendered = render_for_summary(Message("assistant", "", [call]))

    assert len(rendered) < 500
    assert "path='a.py'" in rendered


def test_the_summary_input_is_instructions_then_the_older_messages() -> None:
    msgs, omitted = summary_input(turns(2), 1000)

    assert omitted is False
    assert [m.role for m in msgs] == ["system", "user"]
    assert msgs[0].content == SUMMARIZER_INSTRUCTIONS
    assert msgs[1].content == "USER: u1\n\nASSISTANT: a1\n\nUSER: u2\n\nASSISTANT: a2"


def test_the_summary_input_keeps_the_most_recent_messages_that_fit_the_budget() -> None:
    older = [Message("user", "old " * 50), Message("assistant", "mid " * 50), Message("user", "new")]
    budget = message_tokens(Message("user", "USER: new")) + 5

    msgs, omitted = summary_input(older, budget)

    assert omitted is True
    assert msgs[1].content == "USER: new"
    assert msgs[0].content == SUMMARIZER_INSTRUCTIONS + OMITTED_NOTE


def test_the_summary_input_keeps_a_contiguous_tail_not_a_later_small_message_past_a_big_one() -> None:
    older = [Message("user", "small"), Message("assistant", "big " * 400), Message("user", "tail")]

    msgs, omitted = summary_input(older, 30)

    assert omitted is True
    assert msgs[1].content == "USER: tail"


def test_a_newest_message_bigger_than_the_budget_is_cut_to_fit() -> None:
    older = [Message("tool", "q" * 10_000, tool_call_id="c1", name="read")]

    msgs, omitted = summary_input(older, 100)

    assert omitted is True
    assert 0 < len(msgs[1].content) <= 100 * CHARS_PER_TOKEN


def test_the_summary_input_of_nothing_is_empty_text() -> None:
    msgs, omitted = summary_input([], 100)

    assert omitted is False
    assert msgs[1].content == ""


# --- the summary cap ----------------------------------------------------------------------------


def test_a_summary_within_the_cap_is_left_as_it_is() -> None:
    text = "word " * 100  # 500 characters; the cap for a window of 1000 is 200 tokens = 800

    assert cap_summary(text, 1000) == text


def test_a_summary_over_the_cap_is_cut_at_the_last_whitespace_and_marked() -> None:
    text = "word " * 400  # 2000 characters

    capped = cap_summary(text, 1000)

    assert capped.endswith(SUMMARY_TRUNCATED)
    assert SUMMARY_TRUNCATED == " \u2026[summary truncated]"
    body = capped[: -len(SUMMARY_TRUNCATED)]
    assert len(body) <= 200 * CHARS_PER_TOKEN
    assert not body.endswith(" ")
    assert set(body.split()) == {"word"}  # no word was cut in half


def test_a_summary_over_the_cap_without_any_whitespace_is_cut_at_the_cap() -> None:
    capped = cap_summary("z" * 5000, 1000)

    assert capped == "z" * 800 + SUMMARY_TRUNCATED


def test_the_cap_grows_with_the_window() -> None:
    text = "w " * 5000

    assert len(cap_summary(text, 8192)) > len(cap_summary(text, 1000))
    assert len(cap_summary(text, 8192)) <= int(8192 * 0.2) * CHARS_PER_TOKEN + len(SUMMARY_TRUNCATED)


def test_the_minimum_gain_is_fifteen_percent_of_the_window() -> None:
    assert MIN_SUMMARY_GAIN == 0.15
