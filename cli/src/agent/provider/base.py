"""Provider-neutral contract between the agent loop and a model backend."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant only
    tool_call_id: str | None = None  # tool only
    name: str | None = None  # tool only: tool name


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    call: ToolCall


@dataclass(frozen=True)
class Done:
    # prompt_tokens is the full size of the prompt that was sent, cached tokens included: a backend
    # that reports the cached part separately must add it. completion_tokens is the reply's size.
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class ProviderError:
    message: str


ProviderEvent = TextDelta | ToolCallEvent | Done | ProviderError


class Provider(Protocol):
    model: str
    context_window: int

    def stream(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> AsyncGenerator[ProviderEvent]:
        """Never raises (except CancelledError); ends with exactly one Done or ProviderError."""
        ...

    async def check(self) -> str | None:
        """None when ready, else a user-facing error that says how to fix it."""
        ...
