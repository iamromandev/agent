"""Interrupt (AC8): cancel run_turn at each await and check the history stays valid."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import AsyncGenerator, Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import pytest
from src.agent.approval import Choice, Decision
from src.agent.events import AgentEvent, AssistantText, Notice, ToolFinished, TurnEnded
from src.agent.loop import Agent
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
from src.agent.tools import default_tools
from src.agent.tools.base import Prepared, Tool, ToolContext, ToolResult

from tests.fakes import BlockingApprover, Gate, RecordingApprover, ScriptedProvider

INTERRUPTED = "Interrupted by user"
SKIPPED = "Skipped: the turn was interrupted before this call ran"
PARTIAL_MARK = "\n[interrupted by user]"


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
    return ToolContext(workspace=workspace, bash_timeout=60.0)


class Events:
    def __init__(self) -> None:
        self.items: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.items.append(event)

    def count(self, kind: type) -> int:
        return sum(isinstance(e, kind) for e in self.items)


def call(name: str, call_id: str = "c1", **arguments: Any) -> ToolCallEvent:
    return ToolCallEvent(ToolCall(call_id, name, arguments))


def reply(*events: ProviderEvent | Gate) -> list[ProviderEvent | Gate]:
    return [*events, Done(10, 5)]


async def until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.01)


async def cancel(task: asyncio.Task[None]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


def assert_valid_history(history: Sequence[Message]) -> None:
    """Every assistant tool call is answered, in order, right after it; no stray tool message."""
    i = 0
    while i < len(history):
        message = history[i]
        if message.role == "tool":
            raise AssertionError(f"tool message without a call at {i}: {message}")
        i += 1
        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                assert i < len(history), f"no result for call {tool_call.id}"
                answer = history[i]
                assert answer.role == "tool", f"call {tool_call.id} is followed by {answer.role}"
                assert answer.tool_call_id == tool_call.id
                assert answer.name == tool_call.name
                i += 1


def make_agent(
    ctx: ToolContext, script: list[list[ProviderEvent | Gate]], approver: Any
) -> tuple[Agent, ScriptedProvider]:
    provider = ScriptedProvider(script)
    agent = Agent(provider, default_tools(), approver, ctx, 25, today=lambda: date(2026, 11, 12))
    return agent, provider


async def follow_up_works(agent: Agent, provider: ScriptedProvider) -> None:
    """A following run_turn on a fresh script entry ends normally."""
    provider.script.append(reply(TextDelta("fine")))
    events = Events()
    await agent.run_turn("again", events)
    assert events.items[-1] == TurnEnded("done")
    assert agent.history[-1] == Message("assistant", "fine")
    assert_valid_history(agent.history)


# --- mid-stream ------------------------------------------------------------------------------


async def test_cancel_mid_stream_keeps_the_partial_text_and_reports_interrupted(ctx: ToolContext) -> None:
    gate = Gate()
    agent, provider = make_agent(ctx, [reply(TextDelta("par"), gate, TextDelta("tial"))], RecordingApprover())
    events = Events()
    task = asyncio.create_task(agent.run_turn("q", events))
    await gate.reached.wait()
    assert events.count(AssistantText) == 1

    await cancel(task)

    assert events.items[-1] == TurnEnded("interrupted")
    assert [(m.role, m.content) for m in agent.history] == [
        ("user", "q"),
        ("assistant", "par" + PARTIAL_MARK),
    ]
    await follow_up_works(agent, provider)


async def test_cancel_mid_stream_drops_tool_calls_that_were_not_run(ctx: ToolContext) -> None:
    gate = Gate()
    script = [reply(TextDelta("let me"), call("read", path="a.py"), TextDelta(" see"), gate)]
    agent, provider = make_agent(ctx, script, RecordingApprover())
    events = Events()
    task = asyncio.create_task(agent.run_turn("q", events))
    await gate.reached.wait()  # the call arrived between the two deltas
    assert events.count(AssistantText) == 2

    await cancel(task)

    assert [(m.role, m.content, m.tool_calls) for m in agent.history] == [
        ("user", "q", []),
        ("assistant", "let me see" + PARTIAL_MARK, []),
    ]
    assert events.count(ToolFinished) == 0
    await follow_up_works(agent, provider)


async def test_cancel_before_any_text_leaves_only_the_user_message(ctx: ToolContext) -> None:
    gate = Gate()
    agent, provider = make_agent(ctx, [reply(gate, TextDelta("never seen"))], RecordingApprover())
    events = Events()
    task = asyncio.create_task(agent.run_turn("q", events))
    await gate.reached.wait()

    await cancel(task)

    assert events.items == [TurnEnded("interrupted")]
    assert [m.role for m in agent.history] == ["user"]
    await follow_up_works(agent, provider)


async def test_cancel_during_the_second_stream_keeps_the_tool_round_trip(ctx: ToolContext) -> None:
    gate = Gate()
    script = [reply(call("read", path="a.py")), reply(TextDelta("par"), gate, TextDelta("tial"))]
    agent, provider = make_agent(ctx, script, RecordingApprover())
    events = Events()
    task = asyncio.create_task(agent.run_turn("q", events))
    await gate.reached.wait()
    assert events.count(ToolFinished) == 1
    assert events.count(AssistantText) == 1

    await cancel(task)

    assert [m.role for m in agent.history] == ["user", "assistant", "tool", "assistant"]
    assert agent.history[2].content != INTERRUPTED  # the finished call keeps its real result
    assert agent.history[3].content == "par" + PARTIAL_MARK
    assert events.items[-1] == TurnEnded("interrupted")
    assert_valid_history(agent.history)
    await follow_up_works(agent, provider)


class HangingProvider:
    """Its stream blocks forever; records that it started and that it was closed."""

    model = "hang"
    context_window = 1000

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = False

    async def stream(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> AsyncGenerator[ProviderEvent]:
        try:
            self.started.set()
            await asyncio.Event().wait()
            yield TextDelta("unreachable")
        finally:
            self.closed = True

    async def check(self) -> str | None:
        return None


async def test_cancel_closes_the_provider_stream(ctx: ToolContext) -> None:
    provider = HangingProvider()
    agent = Agent(provider, default_tools(), RecordingApprover(), ctx, 25)
    task = asyncio.create_task(agent.run_turn("q", Events()))
    await provider.started.wait()

    await cancel(task)

    assert provider.closed


# --- mid-approval ----------------------------------------------------------------------------


async def test_cancel_mid_approval_answers_the_pending_call_and_changes_nothing(
    ctx: ToolContext, workspace: Path
) -> None:
    approver = BlockingApprover()
    edit = call("edit", path="a.py", old_string="print('hi')", new_string="print('bye')")
    agent, provider = make_agent(ctx, [reply(edit)], approver)
    events = Events()
    task = asyncio.create_task(agent.run_turn("q", events))
    await approver.asked.wait()

    await cancel(task)

    assert events.items[-1] == TurnEnded("interrupted")
    assert [m.role for m in agent.history] == ["user", "assistant", "tool"]
    assert agent.history[2].content == INTERRUPTED
    assert agent.history[2].tool_call_id == "c1"
    assert_valid_history(agent.history)
    assert (workspace / "a.py").read_text() == "print('hi')\n"
    await follow_up_works(agent, provider)


async def test_cancel_mid_approval_answers_every_unanswered_call_in_order(ctx: ToolContext) -> None:
    approver = BlockingApprover()
    script = [
        reply(
            call("read", "c1", path="a.py"),
            call("bash", "c2", command="echo hi"),
            call("read", "c3", path="a.py"),
        )
    ]
    agent, provider = make_agent(ctx, script, approver)
    task = asyncio.create_task(agent.run_turn("q", Events()))
    await approver.asked.wait()

    await cancel(task)

    results = [(m.tool_call_id, m.content) for m in agent.history if m.role == "tool"]
    assert [r[0] for r in results] == ["c1", "c2", "c3"]
    assert "print('hi')" in results[0][1]  # finished before the dialog opened
    assert results[1][1] == INTERRUPTED  # the call in flight
    assert results[2][1] == SKIPPED  # never started
    assert_valid_history(agent.history)
    await follow_up_works(agent, provider)


# --- mid-bash --------------------------------------------------------------------------------


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


async def test_cancel_mid_bash_kills_the_command_and_repairs_history(ctx: ToolContext, workspace: Path) -> None:
    shell_pid, child_pid = workspace / "shell.pid", workspace / "child.pid"
    command = f"echo $$ > {shell_pid}; sleep 30 & echo $! > {child_pid}; wait"
    approver = RecordingApprover([Decision(Choice.ONCE)])
    agent, provider = make_agent(ctx, [reply(call("bash", command=command))], approver)
    events = Events()
    task = asyncio.create_task(agent.run_turn("q", events))
    await until(lambda: all(f.exists() and f.read_text().strip() for f in (shell_pid, child_pid)))
    pids = [int(f.read_text().strip()) for f in (shell_pid, child_pid)]

    started = time.monotonic()
    await cancel(task)
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    await until(lambda: all(_gone(p) for p in pids), timeout=3.0)
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert events.items[-1] == TurnEnded("interrupted")
    assert [m.role for m in agent.history] == ["user", "assistant", "tool"]
    assert agent.history[2].content == INTERRUPTED
    assert_valid_history(agent.history)
    await follow_up_works(agent, provider)


async def test_each_interrupted_call_gets_exactly_one_tool_message(ctx: ToolContext) -> None:
    approver = BlockingApprover()
    agent, provider = make_agent(ctx, [reply(call("bash", "c1", command="ls"))], approver)
    task = asyncio.create_task(agent.run_turn("q", Events()))
    await approver.asked.wait()

    await cancel(task)

    assert sum(m.role == "tool" for m in agent.history) == 1
    await follow_up_works(agent, provider)


# --- mid-execute of a write/edit (S1) --------------------------------------------------------


class SlowWrite:
    """Patches Path.write_bytes so the write, which runs in a worker thread, waits for the test."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        real = Path.write_bytes

        def write_bytes(path: Path, data: Any) -> int:
            if path.name == name:
                self.started.set()
                assert self.release.wait(5.0), "the test never released the write"
            return real(path, data)

        monkeypatch.setattr(Path, "write_bytes", write_bytes)

    async def wait_started(self) -> None:
        assert await asyncio.to_thread(self.started.wait, 5.0), "the write never started"


async def _cancel_mid_write(task: asyncio.Task[None], slow: SlowWrite, times: int = 1) -> None:
    await slow.wait_started()
    for _ in range(times):
        task.cancel()
    slow.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


async def test_cancel_during_an_approved_write_records_the_real_result(
    ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slow = SlowWrite(monkeypatch, "new.txt")
    script = [reply(call("write", path="new.txt", content="one\ntwo\n"))]
    agent, provider = make_agent(ctx, script, RecordingApprover([Decision(Choice.ONCE)]))
    events = Events()
    task = asyncio.create_task(agent.run_turn("q", events))

    await _cancel_mid_write(task, slow)

    assert (workspace / "new.txt").read_text() == "one\ntwo\n"
    assert [m.role for m in agent.history] == ["user", "assistant", "tool"]
    assert agent.history[2].content == "wrote 2 lines to new.txt"
    finished = [e for e in events.items if isinstance(e, ToolFinished)]
    assert len(finished) == 1
    assert finished[0].applied is True
    assert events.count(TurnEnded) == 1
    assert events.items[-1] == TurnEnded("interrupted")
    assert_valid_history(agent.history)
    await follow_up_works(agent, provider)


async def test_cancel_during_an_approved_edit_records_the_real_result(
    ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slow = SlowWrite(monkeypatch, "a.py")
    edit = call("edit", path="a.py", old_string="print('hi')", new_string="print('bye')")
    agent, provider = make_agent(ctx, [reply(edit)], RecordingApprover([Decision(Choice.ONCE)]))
    events = Events()
    task = asyncio.create_task(agent.run_turn("q", events))

    await _cancel_mid_write(task, slow)

    assert (workspace / "a.py").read_text() == "print('bye')\n"
    assert agent.history[2].content == "replaced 1 occurrence(s) in a.py"
    finished = [e for e in events.items if isinstance(e, ToolFinished)]
    assert len(finished) == 1
    assert finished[0].applied is True
    assert events.items[-1] == TurnEnded("interrupted")
    assert_valid_history(agent.history)
    await follow_up_works(agent, provider)


async def test_repeated_cancels_during_a_write_do_not_abandon_it(
    ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slow = SlowWrite(monkeypatch, "new.txt")
    script = [reply(call("write", path="new.txt", content="x\n"))]
    agent, _ = make_agent(ctx, script, RecordingApprover([Decision(Choice.ONCE)]))
    events = Events()
    task = asyncio.create_task(agent.run_turn("q", events))
    await slow.wait_started()
    task.cancel()
    await asyncio.sleep(0)  # let the first cancel land
    task.cancel()
    slow.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert agent.history[2].content == "wrote 1 lines to new.txt"
    assert events.count(TurnEnded) == 1


# --- S2: a cancel during stream cleanup after a ProviderError --------------------------------


class SlowCloseProvider:
    """Reports a ProviderError, then takes its time closing the stream."""

    model = "slow-close"
    context_window = 1000

    def __init__(self) -> None:
        self.closing = asyncio.Event()
        self.finish_closing = asyncio.Event()  # nothing sets it: the close hangs until cancelled

    async def stream(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> AsyncGenerator[ProviderEvent]:
        try:
            yield TextDelta("partial")
            yield ProviderError("backend failed")
        finally:
            self.closing.set()
            await self.finish_closing.wait()

    async def check(self) -> str | None:
        return None


async def test_cancel_while_closing_after_a_provider_error_ends_the_turn_once(ctx: ToolContext) -> None:
    provider = SlowCloseProvider()
    agent = Agent(provider, default_tools(), RecordingApprover(), ctx, 25)
    events = Events()
    task = asyncio.create_task(agent.run_turn("q", events))
    await provider.closing.wait()

    await cancel(task)

    assert events.count(TurnEnded) == 1
    assert events.items[-1] == TurnEnded("interrupted")
    assert [m.role for m in agent.history] == ["user", "assistant"]  # no duplicate
    assert events.count(Notice) == 0


# --- N5: calls that never started are told apart from the one that was cut -------------------


async def test_the_call_in_flight_is_interrupted_and_later_calls_are_skipped(ctx: ToolContext) -> None:
    approver = BlockingApprover()
    script = [reply(call("bash", "c1", command="ls"), call("read", "c2", path="a.py"), call("glob", "c3", pattern="*"))]
    agent, _ = make_agent(ctx, script, approver)
    task = asyncio.create_task(agent.run_turn("q", Events()))
    await approver.asked.wait()

    await cancel(task)

    results = {m.tool_call_id: m.content for m in agent.history if m.role == "tool"}
    assert results == {"c1": INTERRUPTED, "c2": SKIPPED, "c3": SKIPPED}
    assert_valid_history(agent.history)


# --- N7: more interrupt shapes ---------------------------------------------------------------


async def test_a_double_cancel_mid_approval_still_ends_once(ctx: ToolContext) -> None:
    approver = BlockingApprover()
    agent, provider = make_agent(ctx, [reply(call("bash", "c1", command="ls"))], approver)
    events = Events()
    task = asyncio.create_task(agent.run_turn("q", events))
    await approver.asked.wait()

    task.cancel()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events.count(TurnEnded) == 1
    assert events.items[-1] == TurnEnded("interrupted")
    assert [m.role for m in agent.history] == ["user", "assistant", "tool"]
    assert_valid_history(agent.history)
    await follow_up_works(agent, provider)


class SlowReadTool(Tool):
    name: ClassVar[str] = "slow_read"
    description: ClassVar[str] = "A read that waits for the test."
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}, "required": []}
    started: ClassVar[asyncio.Event]

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def test_cancel_between_two_calls_while_a_read_runs_leaves_valid_history(ctx: ToolContext) -> None:
    SlowReadTool.started = asyncio.Event()
    script = [reply(call("slow_read", "c1"), call("read", "c2", path="a.py"))]
    provider = ScriptedProvider(script)
    agent = Agent(provider, [*default_tools(), SlowReadTool()], RecordingApprover(), ctx, 25)
    events = Events()
    task = asyncio.create_task(agent.run_turn("q", events))
    await SlowReadTool.started.wait()

    await cancel(task)

    results = {m.tool_call_id: m.content for m in agent.history if m.role == "tool"}
    assert results == {"c1": INTERRUPTED, "c2": SKIPPED}
    assert events.items[-1] == TurnEnded("interrupted")
    assert_valid_history(agent.history)
    await follow_up_works(agent, provider)
