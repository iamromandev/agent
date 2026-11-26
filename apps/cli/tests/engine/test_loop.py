from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import pytest
import src.engine
from src.engine.events import (
    AgentEvent,
    Notice,
    ToolFinished,
    TurnEnded,
)
from src.engine.loop import Agent, AgentBusy
from src.engine.mode import Mode
from src.engine.provider.base import Done, Message, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.session import NullRecorder
from src.engine.tools import default_tools
from src.engine.tools.base import Prepared, Tool, ToolContext, ToolResult

from tests.fakes import BlockingApprover, Gate, RecordingApprover, ScriptedProvider

TODAY = date(2026, 11, 12)

# The system prompt and tool specs alone take about 1,150 tokens, so the window must be above that.
WINDOW = 2000


class Events:
    """An EventSink that keeps what it was given."""

    def __init__(self) -> None:
        self.items: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.items.append(event)

    def of(self, kind: type) -> list[Any]:
        return [e for e in self.items if isinstance(e, kind)]

    @property
    def last(self) -> AgentEvent:
        return self.items[-1]


class BigTool(Tool):
    name: ClassVar[str] = "big"
    description: ClassVar[str] = "Returns 100,000 characters."
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}, "required": []}

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        return ToolResult("x" * 100_000, summary="100000 chars")


class BoomTool(Tool):
    name: ClassVar[str] = "boom"
    description: ClassVar[str] = "Always raises."
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}, "required": []}

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        raise RuntimeError("disk on fire")


class BoomOnExecuteTool(Tool):
    name: ClassVar[str] = "boom_exec"
    description: ClassVar[str] = "Raises while executing."
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}, "required": []}
    needs_approval: ClassVar[bool] = True

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        async def execute() -> ToolResult:
            raise RuntimeError("late failure")

        return Prepared(None, execute)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.py").write_text("print('hi')\n")
    return root.resolve()


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, bash_timeout=5.0)


def make_agent(
    ctx: ToolContext,
    script: list[list[ProviderEvent | Gate]],
    approver: RecordingApprover | None = None,
    max_steps: int = 25,
    extra_tools: list[Tool] | None = None,
) -> tuple[Agent, ScriptedProvider, RecordingApprover]:
    provider = ScriptedProvider(script, context_window=WINDOW)
    approver = approver or RecordingApprover()
    agent = Agent(provider, [*default_tools(), *(extra_tools or [])], approver, ctx, max_steps, today=lambda: TODAY)
    return agent, provider, approver


def call(name: str, call_id: str = "c1", **arguments: Any) -> ToolCallEvent:
    return ToolCallEvent(ToolCall(call_id, name, arguments))


def reply(*events: ProviderEvent | Gate) -> list[ProviderEvent | Gate]:
    return [*events, Done(10, 5)]


def tool_messages(agent: Agent) -> list[Message]:
    return [m for m in agent.history if m.role == "tool"]


# --- AC6: the basic loop ---------------------------------------------------------------------


async def test_blank_text_makes_no_request(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("x"))])
    events = Events()

    await agent.run_turn("  \n\t", events)

    assert provider.requests == []
    assert agent.history == []
    assert events.items == []


# --- AC7: step limit -------------------------------------------------------------------------


# --- AC10: bad calls -------------------------------------------------------------------------


async def test_missing_arguments_give_an_argument_error_without_a_dialog(ctx: ToolContext) -> None:
    agent, _, approver = make_agent(
        ctx, [reply(call("edit", path="a.py", new_string="x")), reply(TextDelta("ok"))]
    )

    await agent.run_turn("q", Events())

    assert tool_messages(agent)[0].content == 'edit: missing "old_string"'
    assert approver.requests == []


# --- NFR4: a failing tool never escapes ------------------------------------------------------


async def test_a_tool_that_raises_in_prepare_becomes_an_error_result(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(call("boom")), reply(TextDelta("ok"))], extra_tools=[BoomTool()])
    events = Events()

    await agent.run_turn("q", events)

    assert tool_messages(agent)[0].content == "boom failed: disk on fire"
    assert events.of(ToolFinished)[0].result.is_error
    assert events.last == TurnEnded("done")


# --- AC15: output limit ----------------------------------------------------------------------


# --- AC17: approvals -------------------------------------------------------------------------


def _edit(new: str = "print('bye')", call_id: str = "c1") -> ToolCallEvent:
    return call("edit", call_id, path="a.py", old_string="print('hi')", new_string=new)


async def test_read_glob_and_grep_never_ask(ctx: ToolContext) -> None:
    agent, _, approver = make_agent(
        ctx,
        [
            reply(call("read", "c1", path="a.py"), call("glob", "c2", pattern="*.py"), call("grep", "c3", pattern="print")),
            reply(TextDelta("ok")),
        ],
    )

    await agent.run_turn("q", Events())

    assert approver.requests == []
    assert [m.name for m in tool_messages(agent)] == ["read", "glob", "grep"]
    assert not any("denied" in m.content for m in tool_messages(agent))


async def test_a_tool_error_without_a_dialog_is_not_asked(ctx: ToolContext) -> None:
    agent, _, approver = make_agent(
        ctx, [reply(call("edit", path="a.py", old_string="nope", new_string="x")), reply(TextDelta("ok"))]
    )

    await agent.run_turn("q", Events())

    assert approver.requests == []
    assert "not found" in tool_messages(agent)[0].content


# --- AC18: modes -----------------------------------------------------------------------------


async def test_a_forced_write_in_plan_mode_is_refused(ctx: ToolContext, workspace: Path) -> None:
    agent, _, approver = make_agent(ctx, [reply(call("write", path="n.txt", content="x")), reply(TextDelta("ok"))])
    agent.set_mode(Mode.PLAN)

    await agent.run_turn("q", Events())

    assert tool_messages(agent)[0].content == "refused: write is unavailable in plan mode"
    assert approver.requests == []
    assert not (workspace / "n.txt").exists()


# --- AC5 loop part: provider errors ----------------------------------------------------------


# --- N1: an approval answer that is not an allow is a denial ---------------------------------


# --- N7: plan mode matches tool names exactly ------------------------------------------------


# --- S3: one turn at a time ------------------------------------------------------------------


async def test_run_turn_while_a_turn_runs_raises_busy(ctx: ToolContext) -> None:
    approver = BlockingApprover()
    agent, _, _ = make_agent(ctx, [reply(call("bash", command="ls"))])
    agent.approver = approver
    first = asyncio.create_task(agent.run_turn("one", Events()))
    await approver.asked.wait()

    with pytest.raises(AgentBusy):
        await agent.run_turn("two", Events())

    assert [m.content for m in agent.history if m.role == "user"] == ["one"]
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first


async def test_reset_while_a_turn_runs_raises_busy(ctx: ToolContext) -> None:
    approver = BlockingApprover()
    agent, _, _ = make_agent(ctx, [reply(call("bash", command="ls"))])
    agent.approver = approver
    first = asyncio.create_task(agent.run_turn("one", Events()))
    await approver.asked.wait()

    with pytest.raises(AgentBusy):
        agent.reset(recorder=NullRecorder())

    assert agent.history  # untouched
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first


# --- N4: a surprise never escapes run_turn ---------------------------------------------------


class BreakingProvider:
    """Breaks the provider contract: raises mid-stream instead of yielding ProviderError."""

    model = "broken"
    context_window = 2000

    async def stream(self, messages: Any, tools: Any) -> Any:
        yield TextDelta("half")
        raise RuntimeError("socket exploded")

    async def check(self) -> str | None:
        return None


async def test_a_turn_works_after_an_internal_error(ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("fine"))])
    real = src.engine.loop.build_system_prompt
    monkeypatch.setattr(src.engine.loop, "build_system_prompt", _raise_once(real))
    events = Events()

    await agent.run_turn("one", events)

    assert events.of(Notice) == [Notice("error", "Internal error: prompt broke")]
    assert events.last == TurnEnded("error")
    again = Events()
    await agent.run_turn("two", again)
    assert again.last == TurnEnded("done")
    assert agent.history[-1] == Message("assistant", "fine")
    assert len(provider.requests) == 1


def _raise_once(real: Any) -> Any:
    state = {"raised": False}

    def build(*args: Any, **kwargs: Any) -> str:
        if not state["raised"]:
            state["raised"] = True
            raise RuntimeError("prompt broke")
        return real(*args, **kwargs)

    return build


async def test_an_emit_that_always_raises_does_not_escape_run_turn(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(TextDelta("hi"))])

    def emit(event: AgentEvent) -> None:
        raise RuntimeError("ui is gone")

    await agent.run_turn("q", emit)  # must not raise
