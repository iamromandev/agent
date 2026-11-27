"""Wire messages of the session protocol (version 1). Mirrors the engine events; never carries image bytes."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.engine.approval import Choice
from core.engine.todo import Status

PROTOCOL_VERSION = 1


class Message(BaseModel):
    """What every message on the wire has."""

    model_config = ConfigDict(extra="forbid")

    v: Literal[1] = 1
    type: str
    session_id: str | None = None
    seq: int = 0


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PreviewModel(Strict):
    kind: Literal["diff", "content", "command"]
    title: str
    body: str


class ImageMeta(Strict):
    format: str
    name: str
    size: int


class TodoItem(Strict):
    content: str
    status: Status


class AssistantText(Message):
    type: Literal["assistant_text"] = "assistant_text"
    delta: str


class ToolStarted(Message):
    type: Literal["tool_started"] = "tool_started"
    call_id: str
    label: str
    task: bool = False
    name: str = ""
    args: dict[str, Any] = Field(default_factory=dict)


class ToolFinished(Message):
    type: Literal["tool_finished"] = "tool_finished"
    call_id: str
    output: str
    display: str | None = None
    summary: str = ""
    is_error: bool = False
    images: list[ImageMeta] = Field(default_factory=list)
    preview: PreviewModel | None = None
    applied: bool = False


class Notice(Message):
    type: Literal["notice"] = "notice"
    level: Literal["info", "warn", "error"]
    text: str


class UsageUpdate(Message):
    type: Literal["usage_update"] = "usage_update"
    used: int
    window: int
    estimated: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Compacting(Message):
    type: Literal["compacting"] = "compacting"
    active: bool


class TurnEnded(Message):
    type: Literal["turn_ended"] = "turn_ended"
    status: Literal["done", "interrupted", "step_limit", "error"]


class TodosChanged(Message):
    type: Literal["todos_changed"] = "todos_changed"
    items: list[TodoItem] = Field(default_factory=list)


class ToolProgress(Message):
    type: Literal["tool_progress"] = "tool_progress"
    text: str


class Phase(Message):
    type: Literal["phase"] = "phase"
    kind: Literal["waiting", "writing", "approval", "running", "hook"]
    step: int
    max_steps: int
    detail: str = ""


class TaskProgress(Message):
    type: Literal["task_progress"] = "task_progress"
    call_id: str
    event: WireEvent


WireEvent = Annotated[
    AssistantText
    | ToolStarted
    | ToolFinished
    | Notice
    | UsageUpdate
    | Compacting
    | TurnEnded
    | TodosChanged
    | TaskProgress
    | ToolProgress
    | Phase,
    Field(discriminator="type"),
]
TaskProgress.model_rebuild()


class ApprovalRequest(Strict):
    tool: str
    label: str
    preview: PreviewModel | None = None
    allow_always: bool
    save_rule: str | None = None
    protected: bool = False
    protected_path: str = ""
    subagent: str = ""


class TrustAsk(Strict):
    kind: Literal["global_trust", "mcp_trust", "permission_trust"]
    lines: list[str]
    note: str | None = None


class Ask(Message):
    type: Literal["ask"] = "ask"
    ask_id: str
    kind: Literal["approval", "protected", "global_trust", "mcp_trust", "permission_trust"]
    request: ApprovalRequest | TrustAsk

    @model_validator(mode="after")
    def _kind_matches_request(self) -> Ask:
        request = self.request
        if isinstance(request, TrustAsk):
            if self.kind != request.kind:
                raise ValueError("ask kind does not match the trust request")
        elif self.kind not in ("approval", "protected") or (self.kind == "protected") != request.protected:
            raise ValueError("ask kind does not match the approval request")
        return self


class DecisionAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choice: Choice
    note: str = ""


class TrustAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trust: bool


Answer = DecisionAnswer | TrustAnswer


class AskResolved(Message):
    type: Literal["ask_resolved"] = "ask_resolved"
    ask_id: str
    by: str
    decision: Answer


class QueueEntry(Strict):
    typed: str
    label: str = ""


class Snapshot(Message):
    type: Literal["snapshot"] = "snapshot"
    model: str = ""
    mode: str = ""
    busy: bool = False
    queue: list[QueueEntry] = Field(default_factory=list)
    todos: list[TodoItem] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    asks: list[Ask] = Field(default_factory=list)
    records: list[dict[str, Any]] = Field(default_factory=list)


class Overflow(Message):
    type: Literal["overflow"] = "overflow"
    dropped: int = 0


class SessionClosed(Message):
    type: Literal["session_closed"] = "session_closed"
    reason: str = ""


WireMessage = Annotated[
    AssistantText
    | ToolStarted
    | ToolFinished
    | Notice
    | UsageUpdate
    | Compacting
    | TurnEnded
    | TodosChanged
    | TaskProgress
    | ToolProgress
    | Phase
    | Snapshot
    | Ask
    | AskResolved
    | Overflow
    | SessionClosed,
    Field(discriminator="type"),
]
