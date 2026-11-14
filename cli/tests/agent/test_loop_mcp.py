"""MCP tools inside the loop: offered per request, approved every time, hidden and refused in plan mode."""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import pytest
from src.agent.approval import Choice, Decision
from src.agent.events import AgentEvent, ToolFinished, ToolStarted, TurnEnded
from src.agent.loop import Agent
from src.agent.mcp.config import ServerConfig
from src.agent.mcp.contract import CallResult, ToolInfo
from src.agent.mcp.hub import McpHub
from src.agent.mode import Mode
from src.agent.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.agent.tools import default_tools
from src.agent.tools.base import ToolContext

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


async def test_the_models_tools_include_the_mcp_tools_and_the_built_in_ones(ctx: ToolContext) -> None:
    hub, _ = await _hub(ctx.workspace)
    agent, provider, _ = _agent(ctx, hub, [[TextDelta("hi"), Done(1, 1)]], [])

    await agent.run_turn("hello", _ignore)

    offered = provider.requests[0][1]
    assert {"mcp__fake__echo", "mcp__fake__other", "read", "bash"} <= set(offered)
    assert [t.name for t in agent.tools if t.name.startswith("mcp__")] == []  # self.tools stays the built-ins
    await hub.close_all()


async def test_a_tool_of_a_server_that_becomes_ready_later_is_offered_from_the_next_turn(ctx: ToolContext) -> None:
    connector = FakeConnector({"fake": FakeConnection([ECHO])})
    connector.gates["fake"] = asyncio.Event()
    hub = McpHub(connector, ctx.workspace, [ServerConfig("fake", "cmd")], 5.0)
    hub.start()
    script = _script(_done("one"), _done("two"))
    agent, provider, _ = _agent(ctx, hub, script, [])

    await agent.run_turn("first", _ignore)
    connector.gates["fake"].set()
    await wait_until(lambda: hub.states()[0].status == "ready")
    await agent.run_turn("second", _ignore)

    assert "mcp__fake__echo" not in provider.requests[0][1]
    assert "mcp__fake__echo" in provider.requests[1][1]
    await hub.close_all()


async def test_a_tool_of_a_server_that_exited_is_no_longer_offered(ctx: ToolContext) -> None:
    hub, connection = await _hub(ctx.workspace)
    script = _script(_done("one"), _done("two"))
    agent, provider, _ = _agent(ctx, hub, script, [])

    await agent.run_turn("first", _ignore)
    connection.trigger_exit("the server exited")
    await agent.run_turn("second", _ignore)

    assert "mcp__fake__echo" in provider.requests[0][1]
    assert "mcp__fake__echo" not in provider.requests[1][1]
    await hub.close_all()


async def test_a_call_asks_with_the_name_and_the_json_arguments_then_runs_on_once(ctx: ToolContext) -> None:
    hub, connection = await _hub(ctx.workspace)
    script = _script(_call("c1", "mcp__fake__echo", text="hi"), _done("done"))
    agent, _, approver = _agent(ctx, hub, script, [Decision(Choice.ONCE)])
    events: list[AgentEvent] = []

    await agent.run_turn("go", events.append)

    (request,) = approver.requests
    assert request.tool == "mcp__fake__echo"
    assert request.preview is not None
    assert request.preview.title == "mcp__fake__echo"
    assert request.preview.body == '{\n  "text": "hi"\n}'
    assert connection.calls == [("echo", {"text": "hi"}, 5.0)]
    tool = next(m for m in agent.history if m.role == "tool")
    assert tool.content == "hi back"
    assert any(isinstance(e, ToolFinished) and e.result.summary == "ok" for e in events)
    await hub.close_all()


async def test_once_asks_again_on_the_next_call(ctx: ToolContext) -> None:
    hub, connection = await _hub(ctx.workspace)
    script = _script(
        _call("c1", "mcp__fake__echo", text="a"),
        _call("c2", "mcp__fake__echo", text="b"),
        _done(),
    )
    agent, _, approver = _agent(ctx, hub, script, [Decision(Choice.ONCE), Decision(Choice.ONCE)])

    await agent.run_turn("go", _ignore)

    assert len(approver.requests) == 2
    assert len(connection.calls) == 2
    await hub.close_all()


async def test_always_covers_later_calls_of_that_tool_but_not_another_tool_of_the_server(ctx: ToolContext) -> None:
    hub, connection = await _hub(ctx.workspace)
    script = _script(
        _call("c1", "mcp__fake__echo", text="a"),
        _call("c2", "mcp__fake__echo", text="b"),
        _call("c3", "mcp__fake__other"),
        _done(),
    )
    agent, _, approver = _agent(ctx, hub, script, [Decision(Choice.ALWAYS), Decision(Choice.ONCE)])

    await agent.run_turn("go", _ignore)

    assert [r.tool for r in approver.requests] == ["mcp__fake__echo", "mcp__fake__other"]
    assert [name for name, _, _ in connection.calls] == ["echo", "echo", "other"]
    await hub.close_all()


async def test_deny_gives_the_model_a_refusal_and_does_not_call_the_server(ctx: ToolContext) -> None:
    hub, connection = await _hub(ctx.workspace)
    script = _script(_call("c1", "mcp__fake__echo", text="a"), _done("ok"))
    agent, _, _ = _agent(ctx, hub, script, [Decision(Choice.DENY)])

    await agent.run_turn("go", _ignore)

    assert connection.calls == []
    tool = next(m for m in agent.history if m.role == "tool")
    assert tool.content.startswith("User denied this call.")
    await hub.close_all()


async def test_plan_mode_hides_the_mcp_tools(ctx: ToolContext) -> None:
    hub, _ = await _hub(ctx.workspace)
    agent, provider, _ = _agent(ctx, hub, [[TextDelta("hi"), Done(1, 1)]], [])
    agent.set_mode(Mode.PLAN)

    await agent.run_turn("hello", _ignore)

    assert not any(name.startswith("mcp__") for name in provider.requests[0][1])
    await hub.close_all()


async def test_plan_mode_refuses_a_call_without_asking_or_contacting_the_server(ctx: ToolContext) -> None:
    hub, connection = await _hub(ctx.workspace)
    script = _script(_call("c1", "mcp__fake__echo", text="a"), _done("ok"))
    agent, _, approver = _agent(ctx, hub, script, [])
    agent.set_mode(Mode.PLAN)

    await agent.run_turn("go", _ignore)

    assert approver.requests == []
    assert connection.calls == []
    tool = next(m for m in agent.history if m.role == "tool")
    assert "unavailable in plan mode" in tool.content
    await hub.close_all()


async def test_an_error_result_from_the_server_reaches_the_model_as_an_error(ctx: ToolContext) -> None:
    connection = FakeConnection([ECHO], {"echo": CallResult(["it broke"], True)})
    hub, _ = await _hub(ctx.workspace, connection)
    script = _script(_call("c1", "mcp__fake__echo", text="a"), _done("ok"))
    agent, _, _ = _agent(ctx, hub, script, [Decision(Choice.ONCE)])
    events: list[AgentEvent] = []

    await agent.run_turn("go", events.append)

    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert [(e.result.output, e.result.is_error) for e in finished] == [("it broke", True)]
    await hub.close_all()


async def test_a_call_to_an_unknown_mcp_tool_is_an_unknown_tool_error(ctx: ToolContext) -> None:
    hub, _ = await _hub(ctx.workspace)
    script = _script(_call("c1", "mcp__fake__nope"), _done("ok"))
    agent, _, approver = _agent(ctx, hub, script, [])

    await agent.run_turn("go", _ignore)

    assert approver.requests == []
    assert next(m for m in agent.history if m.role == "tool").content == "Unknown tool: mcp__fake__nope"
    await hub.close_all()


async def test_without_extra_tools_the_agent_offers_only_its_own(ctx: ToolContext) -> None:
    provider = ScriptedProvider([[TextDelta("hi"), Done(1, 1)]], context_window=1000)
    agent = Agent(provider, default_tools(), RecordingApprover(), ctx, 25, recorder=MemoryRecorder())

    await agent.run_turn("hello", _ignore)

    assert not any(name.startswith("mcp__") for name in provider.requests[0][1])
    assert [t.name for t in agent.all_tools()] == [t.name for t in agent.tools]


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


async def test_a_write_still_runs_to_the_end_when_the_turn_is_cancelled(ctx: ToolContext) -> None:
    from src.agent.tools.base import Prepared, Preview, Tool, ToolResult

    finished = asyncio.Event()

    class SlowWrite(Tool):
        name = "slow_write"
        description = "writes slowly"
        parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}
        needs_approval = False
        mutating = True

        async def prepare(self, args, ctx):  # type: ignore[no-untyped-def]
            async def execute() -> ToolResult:
                await asyncio.sleep(0.2)
                finished.set()
                return ToolResult("written", False, "ok")

            return Prepared(Preview("content", "slow_write", ""), execute)

    provider = ScriptedProvider(_script(_call("c1", "slow_write"), _done()), context_window=1000)
    agent = Agent(provider, [SlowWrite()], RecordingApprover(), ctx, 25, recorder=MemoryRecorder())
    started = asyncio.Event()
    turn = asyncio.ensure_future(agent.run_turn("go", lambda e: started.set() if isinstance(e, ToolStarted) else None))
    await started.wait()
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    assert finished.is_set()


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


async def test_always_for_a_tool_that_is_still_offered_keeps_covering_it(ctx: ToolContext) -> None:
    hub, connection = await _hub(ctx.workspace)
    script = _script(_call("c1", "mcp__fake__echo", text="a"), _done("one"), _call("c2", "mcp__fake__echo", text="b"), _done("two"))
    agent, _, approver = _agent(ctx, hub, script, [Decision(Choice.ALWAYS)])

    await agent.run_turn("first", _ignore)
    await agent.run_turn("second", _ignore)

    assert len(approver.requests) == 1
    assert len(connection.calls) == 2
    await hub.close_all()
