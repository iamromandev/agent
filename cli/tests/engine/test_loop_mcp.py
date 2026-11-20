"""MCP tools inside the loop: offered per request, approved every time, hidden and refused in plan mode."""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

import pytest
from src.engine.approval import Choice, Decision
from src.engine.events import AgentEvent, TurnEnded
from src.engine.loop import Agent
from src.engine.mcp.config import ServerConfig
from src.engine.mcp.contract import CallResult, ToolInfo
from src.engine.mcp.hub import McpHub
from src.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.tools import default_tools
from src.engine.tools.base import ToolContext

from tests.fakes import FakeConnection, FakeConnector, Gate, MemoryRecorder, RecordingApprover, ScriptedProvider
from tests.helpers import wait_until

ECHO = ToolInfo("echo", "Echo it", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})
OTHER = ToolInfo("other", "Another", {"type": "object", "properties": {}})


def _call(call_id: str, name: str, **args: object) -> list[ProviderEvent | Gate]:
    return [ToolCallEvent(ToolCall(call_id, name, dict(args))), Done(10, 5)]


def _done(text: str = "done") -> list[ProviderEvent | Gate]:
    return [TextDelta(text), Done(1, 1)]


def _script(*replies: list[ProviderEvent | Gate]) -> list[list[ProviderEvent | Gate]]:
    return list(replies)


async def _hub(workspace: Path, connection: FakeConnection | None = None) -> tuple[McpHub, FakeConnection]:
    connection = connection or FakeConnection(
        [ECHO, OTHER], {"echo": CallResult(["hi back"], False), "other": CallResult(["other ran"], False)}
    )
    hub = McpHub(FakeConnector({"fake": connection}), workspace, [ServerConfig("fake", "cmd")], 5.0)
    hub.start()
    await wait_until(lambda: hub.states()[0].status == "ready")
    return hub, connection


def _agent(
    ctx: ToolContext, hub: McpHub, script: list[list[ProviderEvent | Gate]], decisions: list[Decision]
) -> tuple[Agent, ScriptedProvider, RecordingApprover]:
    provider = ScriptedProvider(script, context_window=1000)
    approver = RecordingApprover(decisions)
    agent = Agent(
        provider,
        default_tools(),
        approver,
        ctx,
        25,
        today=lambda: date(2026, 11, 14),
        recorder=MemoryRecorder(),
        extra_tools=hub.tools,
    )
    return agent, provider, approver


def _ignore(_: AgentEvent) -> None:
    return None


async def test_cancelling_a_turn_during_a_hanging_mcp_call_returns_promptly_and_cancels_the_call(
    ctx: ToolContext,
) -> None:
    connection = FakeConnection([ECHO])
    connection.call_gate = asyncio.Event()  # the call never answers
    hub, _ = await _hub(ctx.workspace, connection)
    script = _script(_call("c1", "mcp__fake__echo", text="a"), _done())
    agent, _, _ = _agent(ctx, hub, script, [Decision(Choice.ONCE)])
    events: list[AgentEvent] = []

    turn = asyncio.ensure_future(agent.run_turn("go", events.append))
    await wait_until(lambda: connection.calls != [])
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(turn, 2)

    assert connection.cancelled == 1
    assert any(isinstance(e, TurnEnded) and e.status == "interrupted" for e in events)
    tool = next(m for m in agent.history if m.role == "tool")
    assert tool.content == "Interrupted by user"
    await hub.close_all()


async def test_always_for_a_tool_does_not_carry_over_to_another_server_that_takes_its_name(
    ctx: ToolContext,
) -> None:
    first = FakeConnection([OTHER], {"other": CallResult(["from a.b"], False)})
    second = FakeConnection([OTHER], {"other": CallResult(["from a_b"], False)})
    hub = McpHub(
        FakeConnector({"a.b": first, "a_b": second}),
        ctx.workspace,
        [ServerConfig("a.b", "cmd"), ServerConfig("a_b", "cmd")],
        5.0,
    )
    hub.start()
    await wait_until(lambda: all(s.status == "ready" for s in hub.states()))
    assert [t.name for t in hub.tools()] == ["mcp__a_b__other"]  # both clean to this name: the first server has it
    script = _script(
        _call("c1", "mcp__a_b__other"), _done("one"), _call("c2", "mcp__a_b__other"), _done("two")
    )
    agent, _, approver = _agent(ctx, hub, script, [Decision(Choice.ALWAYS), Decision(Choice.ONCE)])

    await agent.run_turn("first", _ignore)
    first.trigger_exit("the server exited")  # the second server's tool now has the same name
    await agent.run_turn("second", _ignore)

    assert len(approver.requests) == 2  # the grant for the first server's tool is gone
    assert first.calls != [] and second.calls != []
    await hub.close_all()
