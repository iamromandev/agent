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
from src.agent.approval import Choice, Decision
from src.agent.compaction import CHARS_PER_TOKEN, Reported, estimate_tokens, message_tokens
from src.agent.events import AgentEvent, AssistantText, Notice, ToolStarted, TurnEnded, UsageUpdate
from src.agent.loop import INTERRUPTED, NOT_COMPLETED, PARTIAL_MARK, SKIPPED, Agent, AgentBusy, close_recorder
from src.agent.mode import Mode
from src.agent.provider.base import (
    Done,
    Message,
    ProviderError,
    ProviderEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
)
from src.agent.session import SessionWriteError, ToolMeta
from src.agent.tools import default_tools
from src.agent.tools.base import Prepared, Tool, ToolContext, ToolResult

from tests.agent.test_loop_interrupt import SlowWrite
from tests.fakes import BlockingApprover, Gate, MemoryRecorder, RecordingApprover, ScriptedProvider

TODAY = date(2026, 11, 12)


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
    provider = ScriptedProvider(script)
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


async def test_a_plain_turn_is_recorded(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [reply(TextDelta("hello"))])

    await agent.run_turn("hi", Events())

    assert [m.role for m in agent.history] == ["user", "assistant"]
    assert recorded(recorder) == agent.history


async def test_a_tool_turn_is_recorded(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [reply(call("read", path="a.py")), reply(TextDelta("done"))])

    await agent.run_turn("q", Events())

    assert [m.role for m in agent.history] == ["user", "assistant", "tool", "assistant"]
    assert recorded(recorder) == agent.history


async def test_a_denied_call_is_recorded(ctx: ToolContext) -> None:
    edit = call("edit", path="a.py", old_string="print('hi')", new_string="x")
    script = [reply(edit), reply(TextDelta("ok"))]
    agent, _, recorder = make_agent(ctx, script, RecordingApprover([Decision(Choice.DENY)]))

    await agent.run_turn("q", Events())

    assert agent.history[2].content.startswith("User denied")
    assert recorded(recorder) == agent.history


async def test_an_unknown_tool_is_recorded(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [reply(call("nope")), reply(TextDelta("ok"))])

    await agent.run_turn("q", Events())

    assert agent.history[2].content == "Unknown tool: nope"
    assert recorded(recorder) == agent.history


async def test_a_provider_error_with_partial_text_is_recorded(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [[TextDelta("par"), ProviderError("boom")]])

    await agent.run_turn("q", Events())

    assert [(m.role, m.content) for m in agent.history] == [("user", "q"), ("assistant", "par")]
    assert recorded(recorder) == agent.history


async def test_a_cancel_mid_stream_records_the_partial_text(ctx: ToolContext) -> None:
    gate = Gate()
    agent, _, recorder = make_agent(ctx, [reply(TextDelta("par"), gate, TextDelta("tial"))])
    task = asyncio.create_task(agent.run_turn("q", Events()))
    await gate.reached.wait()

    await cancel(task)

    assert agent.history[-1] == Message("assistant", "par" + PARTIAL_MARK)
    assert recorded(recorder) == agent.history


async def test_a_cancel_mid_approval_records_the_repair_messages(ctx: ToolContext) -> None:
    """AC2: the assistant's calls and their "Interrupted" / "Skipped" results are on record."""
    approver = BlockingApprover()
    script = [reply(call("read", "c1", path="a.py"), call("bash", "c2", command="ls"), call("read", "c3", path="a.py"))]
    agent, _, recorder = make_agent(ctx, script, approver)
    task = asyncio.create_task(agent.run_turn("q", Events()))
    await approver.asked.wait()

    await cancel(task)

    assert [m.content for m in agent.history if m.role == "tool"][1:] == [INTERRUPTED, SKIPPED]
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


async def test_a_provider_that_raises_leaves_recording_and_history_equal(ctx: ToolContext) -> None:
    recorder = MemoryRecorder()
    agent = Agent(BreakingProvider(), default_tools(), RecordingApprover(), ctx, 25, recorder=recorder)

    await agent.run_turn("q", Events())

    assert recorded(recorder) == agent.history == [Message("user", "q")]


async def test_the_step_limit_is_recorded(ctx: ToolContext) -> None:
    script = [reply(call("read", f"c{i}", path="a.py")) for i in range(5)]
    agent, _, recorder = make_agent(ctx, script, max_steps=2)

    events = Events()
    await agent.run_turn("q", events)

    assert events.items[-1] == TurnEnded("step_limit")
    assert len(agent.history) == 5
    assert recorded(recorder) == agent.history


async def test_a_second_turn_appends_to_the_same_record(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [reply(TextDelta("one")), reply(TextDelta("two"))])

    await agent.run_turn("a", Events())
    await agent.run_turn("b", Events())

    assert len(agent.history) == 4
    assert recorded(recorder) == agent.history


async def test_every_message_is_on_record_before_the_next_model_request_is_sent(ctx: ToolContext) -> None:
    """R2: the record is written as the history grows, not at the end of the turn."""
    gate = Gate()
    script: list[list[ProviderEvent | Gate]] = [
        reply(call("read", "c1", path="a.py")),
        [gate, TextDelta("done"), Done(10, 5)],
    ]
    agent, provider, recorder = make_agent(ctx, script)
    task = asyncio.create_task(agent.run_turn("q", Events()))
    await gate.reached.wait()  # the second request is out and its stream is waiting

    sent = provider.requests[1][0][1:]  # without the system prompt
    assert [m.role for m in sent] == ["user", "assistant", "tool"]
    assert recorded(recorder) == sent

    gate.release.set()
    await task


async def test_a_blank_message_records_nothing(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [reply(TextDelta("x"))])

    await agent.run_turn("  ", Events())

    assert recorder.messages == []


# --- tool meta -------------------------------------------------------------------------------


async def test_a_tool_result_is_recorded_with_its_label_and_summary(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [reply(call("read", "c1", path="a.py")), reply(TextDelta("ok"))])

    await agent.run_turn("q", Events())

    assert meta_of(recorder, "c1") == ToolMeta("read a.py", "1 lines")
    assert agent.tool_meta == {"c1": ToolMeta("read a.py", "1 lines")}


async def test_an_assistant_or_user_message_has_no_meta(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [reply(TextDelta("hi"))])

    await agent.run_turn("q", Events())

    assert [meta for _, meta in recorder.messages] == [None, None]
    assert agent.tool_meta == {}


async def test_an_unknown_tool_is_labelled_with_its_name(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [reply(call("nope", "c1")), reply(TextDelta("ok"))])

    await agent.run_turn("q", Events())

    assert meta_of(recorder, "c1") == ToolMeta("nope", "unknown tool")


async def test_a_denied_call_keeps_its_label_and_summary(ctx: ToolContext) -> None:
    edit = call("edit", "c1", path="a.py", old_string="print('hi')", new_string="x")
    agent, _, recorder = make_agent(ctx, [reply(edit), reply(TextDelta("ok"))], RecordingApprover([Decision(Choice.DENY)]))

    await agent.run_turn("q", Events())

    assert meta_of(recorder, "c1") == ToolMeta("edit a.py", "denied")


async def test_interrupt_repairs_are_recorded_as_interrupted_and_skipped(ctx: ToolContext) -> None:
    approver = BlockingApprover()
    script = [reply(call("bash", "c1", command="ls"), call("read", "c2", path="a.py"))]
    agent, _, recorder = make_agent(ctx, script, approver)
    task = asyncio.create_task(agent.run_turn("q", Events()))
    await approver.asked.wait()

    await cancel(task)

    assert meta_of(recorder, "c1") == ToolMeta("bash ls", "interrupted")
    assert meta_of(recorder, "c2") == ToolMeta("read a.py", "skipped")
    assert agent.tool_meta == {"c1": ToolMeta("bash ls", "interrupted"), "c2": ToolMeta("read a.py", "skipped")}


async def test_internal_error_repairs_are_recorded_as_not_completed(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [reply(call("read", "c1", path="a.py"), call("glob", "c2", pattern="*"))])

    def emit(event: AgentEvent) -> None:
        if isinstance(event, ToolStarted) and event.call_id == "c1":
            raise RuntimeError("ui exploded")

    await agent.run_turn("q", emit)

    assert meta_of(recorder, "c1") == ToolMeta("read a.py", "not completed")
    assert meta_of(recorder, "c2") == ToolMeta("glob *", "not completed")


class ExplodingExecute(Tool):
    name: ClassVar[str] = "explode"
    description: ClassVar[str] = "Raises while preparing."
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {"x": {"type": "string"}}, "required": []}

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        raise RuntimeError("disk on fire")


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


async def test_a_failing_tool_is_recorded_with_its_failed_summary(ctx: ToolContext) -> None:
    script = [reply(call("explode", "c1", x="v")), reply(TextDelta("ok"))]
    agent, _, recorder = make_agent(ctx, script, extra_tools=[ExplodingExecute()])

    await agent.run_turn("q", Events())

    assert meta_of(recorder, "c1") == ToolMeta("explode v", "failed")


# --- write errors never stop the turn --------------------------------------------------------


def notices(events: Events) -> list[Notice]:
    return events.of(Notice)


async def test_a_failing_recorder_does_not_stop_the_turn_and_one_notice_is_shown(ctx: ToolContext) -> None:
    recorder = MemoryRecorder(fail=2)
    agent, _, _ = make_agent(ctx, [reply(call("read", path="a.py")), reply(TextDelta("done"))], recorder=recorder)
    events = Events()

    await agent.run_turn("q", events)

    assert events.items[-1] == TurnEnded("done")
    assert [m.role for m in agent.history] == ["user", "assistant", "tool", "assistant"]
    assert notices(events) == [Notice("error", "Could not save the session: disk full")]
    assert recorded(recorder) == agent.history  # the queued ones were written once it recovered


async def test_the_notice_comes_before_the_turn_ends(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(TextDelta("x"))], recorder=MemoryRecorder(fail=5))
    events = Events()

    await agent.run_turn("q", events)

    kinds = [type(e).__name__ for e in events.items]
    assert kinds.index("Notice") < kinds.index("TurnEnded")
    assert events.items[-1] == TurnEnded("done")


async def test_a_failure_after_a_success_gives_a_new_notice(ctx: ToolContext) -> None:
    class Flaky(MemoryRecorder):
        def message(self, message: Message, meta: ToolMeta | None) -> None:
            # user ok, assistant fails, next turn: user ok again, assistant fails again
            self.fail = 1 if message.role == "assistant" else 0
            super().message(message, meta)

    agent, _, _ = make_agent(ctx, [reply(TextDelta("one")), reply(TextDelta("two"))], recorder=Flaky())
    events = Events()

    await agent.run_turn("a", events)
    await agent.run_turn("b", events)

    assert len(notices(events)) == 2
    assert events.items[-1] == TurnEnded("done")


async def test_a_failure_streak_across_turns_is_one_notice(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(
        ctx, [reply(TextDelta("one")), reply(TextDelta("two"))], recorder=MemoryRecorder(fail=100)
    )
    events = Events()

    await agent.run_turn("a", events)
    await agent.run_turn("b", events)

    assert len(notices(events)) == 1


async def test_a_recorder_that_raises_something_else_is_a_save_error_too(ctx: ToolContext) -> None:
    recorder = MemoryRecorder(fail=1, error=RuntimeError("bug in recorder"))
    agent, _, _ = make_agent(ctx, [reply(TextDelta("x"))], recorder=recorder)
    events = Events()

    await agent.run_turn("q", events)

    assert events.items[-1] == TurnEnded("done")
    assert notices(events) == [Notice("error", "Could not save the session: bug in recorder")]
    assert [m.role for m in agent.history] == ["user", "assistant"]


async def test_a_save_error_after_a_cancel_is_reported_before_interrupted(ctx: ToolContext) -> None:
    gate = Gate()
    agent, _, _ = make_agent(ctx, [reply(TextDelta("par"), gate)], recorder=MemoryRecorder(fail=9))
    events = Events()
    task = asyncio.create_task(agent.run_turn("q", events))
    await gate.reached.wait()

    await cancel(task)

    assert events.items[-1] == TurnEnded("interrupted")
    assert len(notices(events)) == 1
    assert [m.role for m in agent.history] == ["user", "assistant"]


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


async def test_a_late_failure_saving_partial_text_on_cancel_is_reported_before_interrupted(ctx: ToolContext) -> None:
    gate = Gate()
    recorder = FailWhen(lambda m: m.role == "assistant")
    agent, _, _ = make_agent(ctx, [reply(TextDelta("par"), gate)], recorder=recorder)
    events = Events()
    task = asyncio.create_task(agent.run_turn("q", events))
    await gate.reached.wait()

    await cancel(task)

    assert_save_notice_once_before_turn_end(events)
    assert events.items[-1] == TurnEnded("interrupted")


async def test_a_late_failure_saving_the_interrupt_repairs_is_reported_before_interrupted(ctx: ToolContext) -> None:
    approver = BlockingApprover()
    script = [reply(call("bash", "c1", command="ls"), call("read", "c2", path="a.py"))]
    agent, _, _ = make_agent(ctx, script, approver, recorder=FailWhen(lambda m: m.role == "tool"))
    events = Events()
    task = asyncio.create_task(agent.run_turn("q", events))
    await approver.asked.wait()

    await cancel(task)

    assert_save_notice_once_before_turn_end(events)
    assert events.items[-1] == TurnEnded("interrupted")


async def test_a_late_failure_saving_the_internal_error_repairs_is_reported_before_the_turn_ends(
    ctx: ToolContext,
) -> None:
    script = [reply(call("read", "c1", path="a.py"), call("glob", "c2", pattern="*"))]
    agent, _, _ = make_agent(ctx, script, recorder=FailWhen(lambda m: m.role == "tool"))
    events = Events()

    def emit(event: AgentEvent) -> None:
        events(event)
        if isinstance(event, ToolStarted) and event.call_id == "c1":
            raise RuntimeError("ui exploded")

    await agent.run_turn("q", emit)

    assert_save_notice_once_before_turn_end(events)
    assert events.items[-1] == TurnEnded("error")


async def test_a_late_failure_saving_partial_text_after_a_provider_error_is_reported_before_the_turn_ends(
    ctx: ToolContext,
) -> None:
    recorder = FailWhen(lambda m: m.role == "assistant")
    agent, _, _ = make_agent(ctx, [[TextDelta("par"), ProviderError("boom")]], recorder=recorder)
    events = Events()

    await agent.run_turn("q", events)

    assert_save_notice_once_before_turn_end(events)
    assert events.items[-1] == TurnEnded("error")


async def test_a_late_failure_saving_a_tool_result_is_reported_before_the_turn_ends(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(
        ctx, [reply(call("read", "c1", path="a.py"))], recorder=FailWhen(lambda m: m.role == "tool"), max_steps=1
    )
    events = Events()

    await agent.run_turn("q", events)

    assert_save_notice_once_before_turn_end(events)
    assert events.items[-1] == TurnEnded("step_limit")


async def test_a_failure_saving_the_user_message_is_reported_before_the_reply_starts(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(TextDelta("x"))], recorder=FailWhen(lambda m: m.role == "user"))
    events = Events()

    await agent.run_turn("q", events)

    assert isinstance(events.items[0], Notice)
    assert isinstance(events.items[1], AssistantText)
    assert_save_notice_once_before_turn_end(events)


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


async def test_a_save_notice_the_sink_refuses_on_the_cancel_path_is_kept_for_the_next_turn(ctx: ToolContext) -> None:
    gate = Gate()
    recorder = FailWhen(lambda m: m.content.endswith(PARTIAL_MARK))  # only the cut-short text fails
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("par"), gate)], recorder=recorder)
    sink = FlakySink()
    task = asyncio.create_task(agent.run_turn("q", sink))
    await gate.reached.wait()

    await cancel(task)  # the cancel still goes through although the flush raised

    assert sink.events.items[-1] == TurnEnded("interrupted")
    assert save_notices(sink.events) == []
    provider.script.append(reply(TextDelta("again")))
    await agent.run_turn("next", sink)
    assert len(save_notices(sink.events)) == 1


async def test_a_session_write_error_is_not_swallowed_by_the_recorder_double(ctx: ToolContext) -> None:
    """Guards the fake itself: a failing call keeps the record queued, so `messages` stays gapless."""
    recorder = MemoryRecorder(fail=1)

    with pytest.raises(SessionWriteError):
        recorder.message(Message("user", "a"), None)
    recorder.message(Message("assistant", "b"), None)

    assert [m.content for m in recorded(recorder)] == ["a", "b"]


# --- usage tracking --------------------------------------------------------------------------


async def test_done_sets_the_reported_count_and_emits_a_real_usage_update(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [[TextDelta("hi"), Done(700, 123)]])
    events = Events()

    await agent.run_turn("q", events)

    assert events.of(UsageUpdate) == [UsageUpdate(823, 1000, estimated=False)]
    assert agent._reported == Reported(823, 2)  # the user message plus the assistant reply


def growth_from_a_new_message(agent: Agent, size: int = 4000) -> int:
    """How much `estimate()` rises when one more user message is appended."""
    before = agent.estimate()
    agent.history.append(Message("user", "n" * size))
    return agent.estimate() - before


async def test_a_reply_that_was_never_recorded_does_not_hide_the_next_message_after_a_provider_error(
    ctx: ToolContext,
) -> None:
    agent, _, _ = make_agent(ctx, [[call("read", path="a.py"), Done(500, 20), ProviderError("boom")]])
    await agent.run_turn("q", Events())
    assert [m.role for m in agent.history] == ["user"]  # the tool-call-only reply is dropped

    assert growth_from_a_new_message(agent) == message_tokens(Message("user", "n" * 4000))


async def test_a_reply_that_was_never_recorded_does_not_hide_the_next_message_after_a_cancel(
    ctx: ToolContext,
) -> None:
    gate = Gate()
    agent, _, _ = make_agent(ctx, [[call("read", path="a.py"), Done(500, 20), gate]])
    task = asyncio.create_task(agent.run_turn("q", Events()))
    await gate.reached.wait()
    await cancel(task)
    assert [m.role for m in agent.history] == ["user"]

    assert growth_from_a_new_message(agent) == message_tokens(Message("user", "n" * 4000))


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


async def test_partial_text_recorded_after_done_and_a_provider_error_is_covered_by_the_reported_count(
    ctx: ToolContext,
) -> None:
    agent, _, _ = make_agent(ctx, [[TextDelta("par"), Done(500, 20), ProviderError("boom")]])
    await agent.run_turn("q", Events())
    assert [m.role for m in agent.history] == ["user", "assistant"]

    assert agent.estimate() == 520
    assert growth_from_a_new_message(agent) == message_tokens(Message("user", "n" * 4000))


async def test_partial_text_recorded_after_done_and_a_cancel_is_covered_by_the_reported_count(
    ctx: ToolContext,
) -> None:
    gate = Gate()
    agent, _, _ = make_agent(ctx, [[TextDelta("par"), Done(500, 20), gate]])
    task = asyncio.create_task(agent.run_turn("q", Events()))
    await gate.reached.wait()
    await cancel(task)
    assert [m.role for m in agent.history] == ["user", "assistant"]

    assert agent.estimate() == 520
    assert growth_from_a_new_message(agent) == message_tokens(Message("user", "n" * 4000))


async def test_a_second_request_is_anchored_on_its_own_reply(ctx: ToolContext) -> None:
    script: list[list[ProviderEvent | Gate]] = [
        [call("read", "c1", path="a.py"), Done(100, 10)],
        [TextDelta("ok"), Done(300, 20)],
    ]
    agent, _, _ = make_agent(ctx, script)

    await agent.run_turn("q", Events())

    assert agent._reported == Reported(320, 4)
    assert agent.estimate() == 320


# --- reset() ---------------------------------------------------------------------------------


async def test_reset_while_a_turn_runs_raises_agent_busy(ctx: ToolContext) -> None:
    gate = Gate()
    agent, _, _ = make_agent(ctx, [reply(TextDelta("a"), gate)])
    task = asyncio.create_task(agent.run_turn("q", Events()))
    await gate.reached.wait()

    with pytest.raises(AgentBusy):
        agent.reset(recorder=MemoryRecorder())

    assert [m.role for m in agent.history] == ["user"]
    await cancel(task)


async def test_reset_requires_a_recorder(ctx: ToolContext) -> None:
    agent, _, old = make_agent(ctx, [])

    with pytest.raises(TypeError):
        agent.reset()  # ty: ignore[missing-argument]

    assert old.closed == 0  # nothing happened


async def test_reset_replaces_history_and_tool_meta(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(TextDelta("one"))])
    await agent.run_turn("old", Events())
    kept = [Message("user", "u"), Message("assistant", "a")]

    agent.reset(history=kept, tool_meta={"c9": ToolMeta("read x", "3 lines")}, recorder=MemoryRecorder())

    assert agent.history == kept
    assert agent.history is not kept  # the agent owns its list
    assert agent.tool_meta == {"c9": ToolMeta("read x", "3 lines")}


async def test_reset_with_only_a_recorder_empties_everything(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(call("read", "c1", path="a.py")), reply(TextDelta("ok"))])
    await agent.run_turn("q", Events())

    agent.reset(recorder=MemoryRecorder())

    assert agent.history == []
    assert agent.tool_meta == {}
    assert agent._reported is None


async def test_reset_goes_back_to_build_mode_with_no_grants(ctx: ToolContext) -> None:
    edit = call("edit", path="a.py", old_string="print('hi')", new_string="print('bye')")
    approver = RecordingApprover([Decision(Choice.ALWAYS), Decision(Choice.ONCE)])
    agent, provider, _ = make_agent(ctx, [reply(edit), reply(TextDelta("ok"))], approver)
    await agent.run_turn("first", Events())
    assert "edit" in agent.grants.tools
    agent.set_mode(Mode.PLAN)

    agent.reset(recorder=MemoryRecorder())

    assert agent.mode is Mode.BUILD
    assert "edit" not in agent.grants.tools
    provider.script.extend(
        [reply(call("edit", path="a.py", old_string="print('bye')", new_string="print('again')")), reply(TextDelta("k"))]
    )
    await agent.run_turn("second", Events())
    assert len(approver.requests) == 2  # asked again: the "always" did not survive the reset


async def test_reset_hands_back_the_old_recorder_unclosed_and_uses_the_new_one(ctx: ToolContext) -> None:
    agent, _, old = make_agent(ctx, [reply(TextDelta("x")), reply(TextDelta("y"))])
    new = MemoryRecorder()

    returned = agent.reset(recorder=new)
    await agent.run_turn("q", Events())

    assert returned is old
    assert old.closed == 0  # the caller closes it, off its own thread
    assert new.closed == 0
    assert old.messages == []
    assert recorded(new) == agent.history


async def test_reset_with_the_same_recorder_hands_back_nothing(ctx: ToolContext) -> None:
    agent, _, old = make_agent(ctx, [])

    assert agent.reset(recorder=old) is None
    assert old.closed == 0


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


async def test_reset_forgets_a_pending_save_error(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(TextDelta("x"))], recorder=MemoryRecorder(fail=1))
    agent._record(Message("user", "q"))
    assert agent._save_error is not None

    agent.reset(recorder=MemoryRecorder())

    assert agent._save_error is None


async def test_close_closes_the_recorder_and_never_raises(ctx: ToolContext) -> None:
    class BadClose(MemoryRecorder):
        def close(self) -> None:
            raise RuntimeError("boom")

    agent, _, recorder = make_agent(ctx, [])
    agent.close()
    assert recorder.closed == 1

    bad = Agent(ScriptedProvider([]), default_tools(), RecordingApprover(), ctx, 5, recorder=BadClose())
    bad.close()  # no exception


async def test_close_makes_a_recorder_without_close_a_no_op(ctx: ToolContext) -> None:
    class NoClose:
        def message(self, message: Message, meta: ToolMeta | None) -> None: ...

        def compaction(self, *args: object) -> None: ...

        def undo(self, keep: int) -> None: ...

    Agent(ScriptedProvider([]), default_tools(), RecordingApprover(), ctx, 5, recorder=NoClose()).close()  # ty: ignore[invalid-argument-type]


# --- estimate() ------------------------------------------------------------------------------


async def test_estimate_without_a_reported_count_is_the_system_prompt_plus_the_history_plus_the_tool_specs(
    ctx: ToolContext,
) -> None:
    agent, _, _ = make_agent(ctx, [])
    system = agent._system(agent.mode)
    specs = [t.spec() for t in agent.tools_for(agent.mode)]
    agent.reset(history=[Message("user", "a" * 400)], recorder=MemoryRecorder())
    without_tools = message_tokens(system) + message_tokens(Message("user", "a" * 400))

    assert agent.estimate() == estimate_tokens(system, agent.history, None, specs)
    assert agent.estimate() >= without_tools + spec_tokens(specs)
    assert spec_tokens(specs) > 100  # the default tools are not free


async def test_estimate_in_plan_mode_counts_only_the_tools_sent_in_that_mode(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [])
    build = agent.estimate()

    agent.set_mode(Mode.PLAN)

    system = agent._system(Mode.PLAN)
    specs = [t.spec() for t in agent.tools_for(Mode.PLAN)]
    assert len(specs) < len(agent.tools)
    assert agent.estimate() == estimate_tokens(system, [], None, specs)
    assert agent.estimate() != build


async def test_estimate_after_a_reply_uses_the_models_count(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [[TextDelta("hi"), Done(700, 123)]])
    await agent.run_turn("q", Events())

    assert agent.estimate() == 823


async def test_estimate_adds_a_guess_for_messages_after_the_reported_count(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [[TextDelta("hi"), Done(700, 123)]])
    await agent.run_turn("q", Events())
    extra = Message("user", "b" * 80)

    agent.history.append(extra)

    assert agent.estimate() == 823 + message_tokens(extra)


async def test_estimate_falls_back_to_a_guess_after_a_reset(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [[TextDelta("hi"), Done(700, 123)]])
    await agent.run_turn("q", Events())
    assert agent.estimate() == 823

    agent.reset(history=[Message("user", "short")], recorder=MemoryRecorder())

    specs = [t.spec() for t in agent.tools_for(agent.mode)]
    expected = message_tokens(agent._system(agent.mode)) + message_tokens(Message("user", "short")) + spec_tokens(specs)
    assert agent.estimate() == expected
