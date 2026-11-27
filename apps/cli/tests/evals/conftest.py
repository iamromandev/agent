from __future__ import annotations

from core.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from core.testing.fakes import Gate

Reply = list[ProviderEvent | Gate]


def call(name: str, call_id: str = "c1", **args: object) -> Reply:
    """A model reply that makes one tool call."""
    return [ToolCallEvent(ToolCall(call_id, name, dict(args))), Done(10, 5)]


def say(text: str) -> Reply:
    """A model reply that is a final answer."""
    return [TextDelta(text), Done(10, 5)]
