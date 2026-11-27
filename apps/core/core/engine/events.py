"""Plain events the loop reports to the UI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from core.engine.todo import TodoItem
from core.engine.tools.base import Preview, ToolResult


@dataclass(frozen=True)
class AssistantText:
    delta: str


@dataclass(frozen=True)
class ToolStarted:
    call_id: str
    label: str
    task: bool = False  # a `task` call: the UI shows it as a block that unfolds while it runs
    name: str = ""  # the tool name as called
    args: Mapping[str, object] = field(default_factory=dict)  # the raw call arguments (not redacted)


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
    estimated: bool = False  # True: a guess from text length, not a count from the model; shown with a "~"
    prompt_tokens: int = 0  # set only on the update that follows a provider Done
    completion_tokens: int = 0


@dataclass(frozen=True)
class Compacting:
    """The slow part of a compaction (the summary request, about 10 s on a local model) starts or ends."""

    active: bool


@dataclass(frozen=True)
class TurnEnded:
    status: Literal["done", "interrupted", "step_limit", "error"]


@dataclass(frozen=True)
class TodosChanged:
    """The todo list after a `todo` call or an undo (the whole list; empty or all done: nothing to show)."""

    items: tuple[TodoItem, ...]


@dataclass(frozen=True)
class TaskProgress:
    """An event of the child agent of a `task` call, tagged with the parent's call."""

    call_id: str  # the parent's task call
    event: AgentEvent  # ToolStarted, ToolFinished, Notice, UsageUpdate or Compacting of the child


@dataclass(frozen=True)
class ToolProgress:
    """A display-only status line of a running tool call (inside `TaskProgress`); never recorded."""

    text: str


@dataclass(frozen=True)
class WorkflowProgress:
    """The live status line of a workflow run (display only: never recorded). The text is escaped and redacted."""

    text: str


@dataclass(frozen=True)
class WorkflowLine:
    """One finished call, or a notice, of a workflow run. The text is escaped and redacted."""

    level: Literal["info", "warn"]
    text: str


PhaseKind = Literal["waiting", "writing", "approval", "running", "hook"]


@dataclass(frozen=True)
class Phase:
    """What the main agent is doing now (display only: never recorded). Not sent by the `task` child."""

    kind: PhaseKind
    step: int  # 1-based model request number in this user turn (Stop continuations count)
    max_steps: int  # Agent.max_steps (AGENT_MAX_STEPS)
    detail: str = ""  # tool name for approval/running, hook event name for hook; raw (the UI escapes it)


AgentEvent = (
    AssistantText
    | ToolStarted
    | ToolFinished
    | Notice
    | UsageUpdate
    | Compacting
    | TurnEnded
    | TaskProgress
    | TodosChanged
    | ToolProgress
    | Phase
    | WorkflowProgress
    | WorkflowLine
)
EventSink = Callable[[AgentEvent], None]
