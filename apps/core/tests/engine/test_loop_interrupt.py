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
from core.engine.events import AgentEvent, Notice, TurnEnded
from core.engine.loop import Agent
from core.engine.provider.base import (
    Done,
    Message,
    ProviderError,
    ProviderEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
)
from core.engine.tools import default_tools
from core.engine.tools.base import Prepared, Tool, ToolContext, ToolResult
from core.testing.fakes import Gate, RecordingApprover, ScriptedProvider

INTERRUPTED = "Interrupted by user"
SKIPPED = "Skipped: the turn was interrupted before this call ran"
PARTIAL_MARK = "\n[interrupted by user]"

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
    provider = ScriptedProvider(script, context_window=WINDOW)
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


# --- mid-approval ----------------------------------------------------------------------------


# --- mid-bash --------------------------------------------------------------------------------


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


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


# --- S2: a cancel during stream cleanup after a ProviderError --------------------------------


class SlowCloseProvider:
    """Reports a ProviderError, then takes its time closing the stream."""

    model = "slow-close"
    context_window = WINDOW

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


# --- N7: more interrupt shapes ---------------------------------------------------------------


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
    provider = ScriptedProvider(script, context_window=WINDOW)
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
