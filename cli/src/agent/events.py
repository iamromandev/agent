"""Plain events the loop reports to the UI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from src.agent.tools.base import Preview, ToolResult


@dataclass(frozen=True)
class AssistantText:
    delta: str


@dataclass(frozen=True)
class ToolStarted:
    call_id: str
    label: str


@dataclass(frozen=True)
class ToolFinished:
    call_id: str
    result: ToolResult
    preview: Preview | None
    applied: bool  # the call ran and succeeded (so a diff preview was actually applied)


@dataclass(frozen=True)
class Notice:
    level: Literal["info", "warn", "error"]
    text: str


@dataclass(frozen=True)
class UsageUpdate:
    used: int
    window: int


@dataclass(frozen=True)
class TurnEnded:
    status: Literal["done", "interrupted", "step_limit", "error"]


AgentEvent = AssistantText | ToolStarted | ToolFinished | Notice | UsageUpdate | TurnEnded
EventSink = Callable[[AgentEvent], None]
