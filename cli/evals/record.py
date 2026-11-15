"""What one eval run leaves behind, and the event sink that fills it."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src.engine.events import (
    AgentEvent,
    AssistantText,
    EventSink,
    Notice,
    TaskProgress,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    UsageUpdate,
)

REFUSED = ("denied", "refused")  # the summaries the loop gives a call it did not run
OUTPUT_HEAD = 200  # characters of a tool output kept in `tool_results`


@dataclass
class RunRecord:
    task: str
    run: int
    passed: bool = False
    reasons: list[str] = field(default_factory=list)
    steps: int = 0
    tokens: int | None = None  # None: the provider reported no usage
    seconds: float = 0.0
    tools: Counter[str] = field(default_factory=Counter)
    tool_results: list[tuple[str, str, str]] = field(default_factory=list)  # (name, label, output head)
    calls: list[tuple[str, dict[str, Any], str]] = field(default_factory=list)  # (name, arguments, output), from history
    approvals: list[dict[str, Any]] = field(default_factory=list)
    refusals: list[tuple[str, str]] = field(default_factory=list)  # (label, summary) of refused tool calls
    notes: list[str] = field(default_factory=list)  # harness warnings, e.g. a failed `git init`
    error: str | None = None
    ended: str = ""  # how the turn ended: done, step_limit, error, interrupted
    final_text: str = ""
    first_prompt: str = ""


def sink(record: RunRecord) -> EventSink:
    """An event sink that fills `record`. Child (`task`) usage adds tokens; only the main agent's calls are steps."""
    labels: dict[str, str] = {}
    buffer = ""  # text of the model call in progress

    def emit(event: AgentEvent) -> None:
        nonlocal buffer
        match event:
            case ToolStarted(call_id, label):
                labels[call_id] = label
                record.tools[label.split(" ", 1)[0]] += 1
            case ToolFinished(call_id, result):
                label = labels.get(call_id, "")
                record.tool_results.append((label.split(" ", 1)[0], label, result.output[:OUTPUT_HEAD]))
                if result.is_error and result.summary.startswith(REFUSED):
                    record.refusals.append((label, result.summary))
            case AssistantText(delta):
                buffer += delta
            case UsageUpdate(used, _, estimated=False):
                record.steps += 1
                if used > 0:  # a provider without usage reports 0
                    record.tokens = (record.tokens or 0) + used
                record.final_text, buffer = buffer.strip(), ""
            case TaskProgress(_, UsageUpdate(used, _, estimated=False)) if used > 0:
                record.tokens = (record.tokens or 0) + used
            case Notice("error", text):
                record.error = text
            case TurnEnded(status):
                if buffer.strip():  # a reply that ended without a usage count
                    record.final_text, buffer = buffer.strip(), ""
                record.ended = status
                if status == "error" and record.error is None:
                    record.error = "turn ended with an error"

    return emit
