"""Pure helpers for keeping the model history inside the context window (no I/O)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.agent.provider.base import Message, ToolSpec
from src.agent.session.records import ToolMeta

SUMMARY_PREFIX = "Summary of the earlier conversation:\n"
CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD = 4  # tokens per message for role/formatting
MIN_SUMMARY_GAIN = 0.15  # of the window: an automatic summary of less than this is not worth its ~10 s
SUMMARY_MAX = 0.2  # of the window: the most a summary may take
SUMMARY_TRUNCATED = " \u2026[summary truncated]"


@dataclass(frozen=True)
class Reported:
    total: int  # prompt_tokens + completion_tokens of the last Done
    history_len: int  # len(history) when that request was sent, plus 1 for its assistant reply


def message_tokens(m: Message) -> int:
    calls = (
        json.dumps(
            [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in m.tool_calls],
            ensure_ascii=False,
            default=str,
        )
        if m.tool_calls
        else ""
    )
    return (len(m.content) + len(calls)) // CHARS_PER_TOKEN + MESSAGE_OVERHEAD


def estimate_tokens(
    system: Message, history: Sequence[Message], reported: Reported | None, tools: Sequence[ToolSpec] = ()
) -> int:
    """The model's own count when there is one, plus a guess for what was added since.

    Without a count the guess covers the whole request, tool specs included; the model's count
    already includes them, so `tools` is not used after a report.
    """
    if reported is None:
        specs = json.dumps([dataclasses.asdict(t) for t in tools], separators=(",", ":"))
        return message_tokens(system) + sum(message_tokens(m) for m in history) + len(specs) // CHARS_PER_TOKEN
    return reported.total + sum(message_tokens(m) for m in history[reported.history_len :])


SUMMARIZER_INSTRUCTIONS = (
    "Summarize this coding-agent conversation for the agent itself. Keep: the user's goals, decisions made, "
    "files read or changed (with paths), commands run and their outcomes, and open tasks. Be concise; no preamble."
)
OMITTED_NOTE = " The start of the conversation is not included: earlier details omitted."


def is_summary(history: Sequence[Message], index: int) -> bool:
    """A summary stands first in the history; the same words in a later user message are the user's own."""
    if index != 0 or not history:
        return False
    return history[0].role == "user" and history[0].content.startswith(SUMMARY_PREFIX)


def cap_summary(text: str, window: int) -> str:
    """`text` cut to `SUMMARY_MAX` of the window (in characters), at a whitespace, with a mark; short text is as is."""
    limit = int(window * SUMMARY_MAX) * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    head = text[:limit]
    if not text[limit].isspace():  # the cut fell inside a word: go back to the whitespace before it
        parts = head.rsplit(None, 1)
        head = parts[0] if len(parts) == 2 else head
    return head.rstrip() + SUMMARY_TRUNCATED


def turn_starts(history: Sequence[Message]) -> list[int]:
    """Indices of the real user messages: each starts a turn. A summary stands in for old turns, so it is not one."""
    return [i for i, m in enumerate(history) if m.role == "user" and not is_summary(history, i)]


def cut_index(history: Sequence[Message], keep_turns: int) -> int:
    """Where the `keep_turns` most recent turns begin; 0 when there are no more turns than that (nothing is older)."""
    starts = turn_starts(history)
    return starts[-keep_turns] if len(starts) > keep_turns else 0


def pruned_stub(meta: ToolMeta | None, name: str | None) -> str:
    what = f"{meta.label} \u00b7 {meta.summary}" if meta else name or "tool"
    return f"[output pruned: {what}]"


def prune(
    history: Sequence[Message], tool_meta: Mapping[str, ToolMeta], keep_turns: int
) -> tuple[list[Message], int]:
    """Replace the content of tool results before the kept turns with a stub; returns the new list and the count.

    Messages are replaced, never edited (earlier session records share them), and the kept turns come back
    as the same objects. A result that already holds its stub is left alone, so pruning twice changes nothing.
    """
    cut = cut_index(history, keep_turns)
    new = list(history)
    count = 0
    for i in range(cut):
        m = history[i]
        if m.role != "tool":
            continue
        stub = pruned_stub(tool_meta.get(m.tool_call_id or ""), m.name)
        if len(stub) < len(m.content):  # an already pruned result, or one too short to be worth it, stays
            new[i] = dataclasses.replace(m, content=stub)
            count += 1
    return new, count


def render_for_summary(m: Message) -> str:
    match m.role:
        case "assistant":
            calls = "".join(f" [called {c.name}({_render_arguments(c.arguments)})]" for c in m.tool_calls)
            return f"ASSISTANT: {m.content}{calls}"
        case "tool":
            return f"TOOL {m.name or 'tool'}: {m.content}"
        case _:
            return f"{m.role.upper()}: {m.content}"


def _render_arguments(arguments: Mapping[str, object]) -> str:
    return ", ".join(f"{key}={_shorten(repr(value))}" for key, value in arguments.items())


def _shorten(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def summary_input(older: Sequence[Message], budget_tokens: int) -> tuple[list[Message], bool]:
    """The request that summarizes `older`: the instructions, then the most recent messages that fit the budget.

    The bool says that some older ones were left out; the instructions then tell the model so. A newest
    message bigger than the whole budget is cut to fit, so the model always gets something.
    """
    lines: list[str] = []
    used = 0
    cut = False
    for m in reversed(older):
        line = render_for_summary(m)
        cost = len(line) // CHARS_PER_TOKEN + MESSAGE_OVERHEAD
        if used + cost > budget_tokens:
            if not lines:  # nothing fits whole: keep the start of the newest message
                lines.append(line[: max(budget_tokens, 1) * CHARS_PER_TOKEN])
                cut = True
            break
        lines.append(line)
        used += cost
    omitted = cut or len(lines) < len(older)
    lines.reverse()
    instructions = SUMMARIZER_INSTRUCTIONS + (OMITTED_NOTE if omitted else "")
    return [Message("system", instructions), Message("user", "\n\n".join(lines))], omitted
