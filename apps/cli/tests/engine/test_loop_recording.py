"""Recording (R2, AC2): every message the loop adds to the history goes through the Recorder, in order."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import pytest
from src.engine.approval import Choice, Decision
from src.engine.compaction import CHARS_PER_TOKEN, message_tokens
from src.engine.events import AgentEvent, Notice, ToolStarted, TurnEnded, UsageUpdate
from src.engine.loop import NOT_COMPLETED, Agent, close_recorder
from src.engine.provider.base import (
    Done,
    Message,
    ProviderError,
    ProviderEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
)
from src.engine.session import ToolMeta
from src.engine.tools import default_tools
from src.engine.tools.base import Prepared, Tool, ToolContext, ToolResult
from src.engine.usage import UsageRecord

from tests.engine.test_loop_interrupt import SlowWrite
from tests.fakes import BlockingApprover, Gate, MemoryRecorder, RecordingApprover, ScriptedProvider

TODAY = date(2026, 11, 12)

# The system prompt and tool specs alone take about 1,150 tokens, so the window must be above that.
WINDOW = 2000


@pytest.fixture(autouse=True)
def _posix_sh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", "/bin/sh")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.py").write_text("print('hi')\n")
    return root.resolve()


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, bash_timeout=5.0)


class Events:
    def __init__(self) -> None:
        self.items: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.items.append(event)

    def of(self, kind: type) -> list[Any]:
        return [e for e in self.items if isinstance(e, kind)]


def call(name: str, call_id: str = "c1", **arguments: Any) -> ToolCallEvent:
    return ToolCallEvent(ToolCall(call_id, name, arguments))


def reply(*events: ProviderEvent | Gate) -> list[ProviderEvent | Gate]:
    return [*events, Done(10, 5)]


def make_agent(
    ctx: ToolContext,
    script: list[list[ProviderEvent | Gate]],
    approver: Any = None,
    recorder: MemoryRecorder | None = None,
    max_steps: int = 25,
    extra_tools: list[Tool] | None = None,
) -> tuple[Agent, ScriptedProvider, MemoryRecorder]:
    provider = ScriptedProvider(script, context_window=WINDOW)
    recorder = recorder if recorder is not None else MemoryRecorder()
    agent = Agent(
        provider,
        [*default_tools(), *(extra_tools or [])],
        approver or RecordingApprover(),
        ctx,
        max_steps,
        today=lambda: TODAY,
        recorder=recorder,
    )
    return agent, provider, recorder


def recorded(recorder: MemoryRecorder) -> list[Message]:
    return [m for m, _ in recorder.messages]


def meta_of(recorder: MemoryRecorder, call_id: str) -> ToolMeta | None:
    return next(meta for m, meta in recorder.messages if m.tool_call_id == call_id)


def spec_tokens(specs: Sequence[ToolSpec]) -> int:
    return len(json.dumps([dataclasses.asdict(t) for t in specs], separators=(",", ":"))) // CHARS_PER_TOKEN


async def cancel(task: asyncio.Task[None]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# --- AC2: what was recorded equals the in-memory history, whatever happened ------------------


async def test_a_provider_error_with_partial_text_is_recorded(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [[TextDelta("par"), ProviderError("boom")]])

    await agent.run_turn("q", Events())

    assert [(m.role, m.content) for m in agent.history] == [("user", "q"), ("assistant", "par")]
    assert recorded(recorder) == agent.history


async def test_a_cancel_during_an_approved_write_records_the_real_result(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    slow = SlowWrite(monkeypatch, "new.txt")
    script = [reply(call("write", path="new.txt", content="one\ntwo\n"))]
    agent, _, recorder = make_agent(ctx, script, RecordingApprover([Decision(Choice.ONCE)]))
    task = asyncio.create_task(agent.run_turn("q", Events()))
    await slow.wait_started()

    task.cancel()
    slow.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert agent.history[-1].content == "wrote 2 lines to new.txt"
    assert recorded(recorder) == agent.history


class BreakingProvider:
    model = "broken"
    context_window = 1000

    async def stream(self, messages: Any, tools: Any) -> Any:
        yield TextDelta("half")
        raise RuntimeError("socket exploded")

    async def check(self) -> str | None:
        return None


async def test_an_internal_error_is_recorded_with_the_repair_messages(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [reply(call("read", "c1", path="a.py"), call("glob", "c2", pattern="*"))])

    def emit(event: AgentEvent) -> None:
        if isinstance(event, ToolStarted) and event.call_id == "c1":
            raise RuntimeError("ui exploded")

    await agent.run_turn("q", emit)

    assert [m.content for m in agent.history if m.role == "tool"] == [NOT_COMPLETED, NOT_COMPLETED]
    assert recorded(recorder) == agent.history


# --- tool meta -------------------------------------------------------------------------------


class ExplodingExecute(Tool):
    name: ClassVar[str] = "explode"
    description: ClassVar[str] = "Raises while preparing."
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {"x": {"type": "string"}}, "required": []}

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        raise RuntimeError("disk on fire")


class Colorful(Tool):
    """Returns plain output with a colored display copy, like bash does."""

    name: ClassVar[str] = "colorful"
    description: ClassVar[str] = "Colored output."
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": []}

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        n = int(args.get("n", 5))
        return ToolResult("x" * n, summary="ok", display="\x1b[32m" + "x" * n + "\x1b[0m")


class BrokenLabel(Tool):
    name: ClassVar[str] = "broken"
    description: ClassVar[str] = "Its label raises."
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {"x": {"type": "string"}}, "required": []}

    def label(self, args: dict[str, Any]) -> str:
        raise RuntimeError("no label for you")

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        return ToolResult("unreachable", False, "")


async def test_a_repair_for_a_tool_that_cannot_be_labelled_falls_back_to_its_name(ctx: ToolContext) -> None:
    approver = BlockingApprover()
    script = [reply(call("bash", "c1", command="ls"), call("nope", "c2"), call("broken", "c3", x="v"))]
    agent, _, recorder = make_agent(ctx, script, approver, extra_tools=[BrokenLabel()])
    task = asyncio.create_task(agent.run_turn("q", Events()))
    await approver.asked.wait()

    await cancel(task)

    assert meta_of(recorder, "c2") == ToolMeta("nope", "skipped")  # unknown tool
    assert meta_of(recorder, "c3") == ToolMeta("broken", "skipped")  # label() raised
    assert recorded(recorder) == agent.history


# --- write errors never stop the turn --------------------------------------------------------


def notices(events: Events) -> list[Notice]:
    return events.of(Notice)


class FailWhen(MemoryRecorder):
    """Fails only the records the predicate picks, so earlier flush points have nothing to report."""

    def __init__(self, predicate: Callable[[Message], bool]) -> None:
        super().__init__()
        self.predicate = predicate

    def message(self, message: Message, meta: ToolMeta | None) -> None:
        self.fail = 1 if self.predicate(message) else 0
        super().message(message, meta)


def save_notices(events: Events) -> list[Notice]:
    return [n for n in events.of(Notice) if n.text.startswith("Could not save")]


def assert_save_notice_once_before_turn_end(events: Events) -> None:
    assert len(save_notices(events)) == 1
    assert events.items.index(save_notices(events)[0]) < events.items.index(events.of(TurnEnded)[0])
    assert len(events.of(TurnEnded)) == 1


class FlakySink:
    """Collects events, but raises on the first save-error notice it is given."""

    def __init__(self) -> None:
        self.events = Events()
        self.raised = False

    def __call__(self, event: AgentEvent) -> None:
        if isinstance(event, Notice) and event.text.startswith("Could not save") and not self.raised:
            self.raised = True
            raise RuntimeError("sink broke")
        self.events(event)


async def test_a_save_notice_that_the_sink_refused_is_retried_at_the_next_flush_point(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(TextDelta("x"))], recorder=FailWhen(lambda m: m.role == "user"))
    sink = FlakySink()

    await agent.run_turn("q", sink)

    assert sink.raised
    assert_save_notice_once_before_turn_end(sink.events)


# --- usage tracking --------------------------------------------------------------------------


def growth_from_a_new_message(agent: Agent, size: int = 4000) -> int:
    """How much `estimate()` rises when one more user message is appended."""
    before = agent.estimate()
    agent.history.append(Message("user", "n" * size))
    return agent.estimate() - before


async def test_a_reply_that_was_never_recorded_does_not_hide_the_next_message_after_a_failing_sink(
    ctx: ToolContext,
) -> None:
    agent, _, _ = make_agent(ctx, [[call("read", path="a.py"), Done(500, 20)]])

    def emit(event: AgentEvent) -> None:
        if isinstance(event, UsageUpdate):
            raise RuntimeError("ui exploded")

    await agent.run_turn("q", emit)
    assert [m.role for m in agent.history] == ["user"]

    assert growth_from_a_new_message(agent) == message_tokens(Message("user", "n" * 4000))


# --- reset() ---------------------------------------------------------------------------------


async def test_reset_tolerates_a_recorder_without_close_or_whose_close_fails(ctx: ToolContext) -> None:
    class NoClose:
        def message(self, message: Message, meta: ToolMeta | None) -> None: ...
        def compaction(
            self,
            history: Sequence[Message],
            tool_meta: Mapping[str, ToolMeta],
            pruned: int,
            summarized: int,
            before: int,
            after: int,
        ) -> None: ...
        def undo(self, keep: int) -> None: ...
        def model(self, model: str, window: int) -> None: ...
        def usage(self, record: UsageRecord) -> None: ...

    class BadClose(NoClose):
        def close(self) -> None:
            raise RuntimeError("close failed")

    agent = Agent(ScriptedProvider([]), default_tools(), RecordingApprover(), ctx, 5, recorder=NoClose())

    bad = BadClose()
    assert agent.reset(recorder=bad) is not None
    assert agent.reset(recorder=NoClose()) is bad
    close_recorder(bad)  # never raises
    close_recorder(NoClose())

    assert agent.history == []


# --- estimate() ------------------------------------------------------------------------------


# --- colored display copy: the model and the session file only get the plain text ------------


async def test_model_switched_is_recorded_without_touching_the_history(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [])

    agent.model_switched("b", 4096)

    assert recorder.models == [("b", 4096)]
    assert agent.history == []
