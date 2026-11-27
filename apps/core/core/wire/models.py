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


class ProviderState(Message):
    type: Literal["provider_state"] = "provider_state"
    problem: str | None = None


class ModelChanged(Message):
    type: Literal["model_changed"] = "model_changed"
    model: str
    window: int


class UserEntry(Message):
    type: Literal["user_entry"] = "user_entry"
    text: str
    label: str | None = None
    client: str = ""


class Delivering(Message):
    type: Literal["delivering"] = "delivering"


class JobEventsDone(Message):
    type: Literal["job_events_done"] = "job_events_done"


class EntryEnd(Message):
    type: Literal["entry_end"] = "entry_end"
    discard: bool


class InitCheck(Strict):
    level: str
    text: str


class Restore(Strict):
    client: str
    text: str


class TurnOutcome(Message):
    # The texts that go back are the user's own draft, sent as typed (never redacted): a client must get its exact
    # text back in its input box.
    type: Literal["turn_outcome"] = "turn_outcome"
    kind: Literal["turn", "compact", "workflow"]
    status: str | None = None
    continued: bool = False
    restore_text: str | None = None
    queue_restore: str | None = None
    delivered: bool = False
    turn: int = 0
    init_check: InitCheck | None = None
    ack: int = 0
    client: str = ""
    interrupted_by: str = ""
    restores: list[Restore] = Field(default_factory=list)


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
    title: str = ""
    header: list[str] = Field(default_factory=list)
    footer: list[str] = Field(default_factory=list)
    decline: str = ""
    accept: str = ""


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
    decision: Answer | None = None  # None: withdrawn (its turn was interrupted, or the session closed)


class QueueEntry(Strict):
    typed: str
    label: str = ""
    client: str = ""


class QueueChanged(Message):
    type: Literal["queue_changed"] = "queue_changed"
    entries: list[QueueEntry] = Field(default_factory=list)


class TurnUndone(Message):
    type: Literal["turn_undone"] = "turn_undone"
    client: str


class ModeChanged(Message):
    type: Literal["mode_changed"] = "mode_changed"
    mode: Literal["build", "edits", "plan"]


class Claims(Strict):
    """Who holds the session now: a turn (or compaction, undo, workflow), a switch, and whether a message queues."""

    turn_active: bool = False
    switching: bool = False
    undoing: bool = False
    workflow_active: bool = False
    queueing: bool = False


class Snapshot(Message):
    type: Literal["snapshot"] = "snapshot"
    model: str = ""
    window: int = 0
    mode: str = ""
    busy: bool = False
    claims: Claims = Field(default_factory=Claims)
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
    | ProviderState
    | ModelChanged
    | QueueChanged
    | ModeChanged
    | TurnUndone
    | UserEntry
    | Delivering
    | JobEventsDone
    | EntryEnd
    | TurnOutcome
    | Snapshot
    | Ask
    | AskResolved
    | Overflow
    | SessionClosed,
    Field(discriminator="type"),
]
