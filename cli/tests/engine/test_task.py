"""The `task` tool: a read-only child agent whose final report is all the parent keeps."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, ClassVar

from src.engine.agents.loader import AgentDef, Agents
from src.engine.approval import Decision
from src.engine.events import AgentEvent, ToolFinished, TurnEnded
from src.engine.loop import Agent
from src.engine.provider.base import Done, ProviderError, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.task import AGENT_REPORT_HEADER, REPORT_HEADER, TaskTool
from src.engine.tools import default_tools, read_only_tools
from src.engine.tools.base import Prepared, Tool, ToolContext, ToolResult
from src.engine.web import FetchResponse

from tests.fakes import FakeFetcher, Gate, MemoryRecorder, RecordingApprover, ScriptedProvider

URL = "https://example.com/docs"
Script = list[list[ProviderEvent | Gate]]


def _call(call_id: str, name: str, **args: object) -> list[ProviderEvent | Gate]:
    return [ToolCallEvent(ToolCall(call_id, name, dict(args))), Done(10, 5)]


def _task(call_id: str = "t1", prompt: str = "find it") -> list[ProviderEvent | Gate]:
    return _call(call_id, "task", description="look around", prompt=prompt)


def _say(text: str, used: int = 2) -> list[ProviderEvent | Gate]:
    return [TextDelta(text), Done(used, 1)]


def _page() -> FetchResponse:
    return FetchResponse(URL, 200, "OK", "text/plain", None, b"hello", 5)


def _make(
    ctx: ToolContext,
    script: Script,
    decisions: list[Decision] | None = None,
    *,
    fetcher: FakeFetcher | None = None,
    max_steps: int = 15,
    recorder: Any = None,
    extra: list[Tool] | None = None,
    delay: float = 0.0,
    approver: Any = None,
    agents: Agents | None = None,
) -> tuple[Agent, ScriptedProvider, RecordingApprover]:
    provider = ScriptedProvider(script, delay=delay, context_window=100_000)
    approver = approver or RecordingApprover(decisions)
    agent = Agent(
        provider,
        default_tools(fetcher=fetcher),
        approver,
        ctx,
        25,
        today=lambda: date(2026, 11, 14),
        recorder=recorder if recorder is not None else MemoryRecorder(),
        extra_tools=(lambda: extra) if extra is not None else None,
        task_tools=read_only_tools(fetcher),
        task_max_steps=max_steps,
        agents=agents,
    )
    return agent, provider, approver


async def _run(agent: Agent, text: str = "go") -> list[AgentEvent]:
    events: list[AgentEvent] = []
    await agent.run_turn(text, events.append)
    return events


def _tool_message(agent: Agent, call_id: str) -> str:
    return next(m.content for m in agent.history if m.role == "tool" and m.tool_call_id == call_id)


async def test_the_step_limit_without_any_text_gives_only_the_note(ctx: ToolContext) -> None:
    (ctx.workspace / "a.txt").write_text("a")
    looking: list[ProviderEvent | Gate] = _call("r", "read", path="a.txt")
    agent, _, _ = _make(ctx, [_task(), looking, looking, _say("done")], max_steps=2)
    await _run(agent)

    assert _tool_message(agent, "t1") == f"{REPORT_HEADER}\n[step limit (2) reached before the subagent finished]"


async def test_an_empty_report_is_an_error_no_report(ctx: ToolContext) -> None:
    agent, _, _ = _make(ctx, [_task(), [Done(1, 1)], _say("done")])
    events = await _run(agent)

    assert _tool_message(agent, "t1") == "(no report)"
    finished = next(e for e in events if isinstance(e, ToolFinished) and e.call_id == "t1")
    assert finished.result.is_error


async def test_a_blank_prompt_is_refused_and_starts_no_child(ctx: ToolContext) -> None:
    agent, provider, _ = _make(ctx, [_task(prompt="   "), _say("done")])
    await _run(agent)

    assert _tool_message(agent, "t1") == "task: prompt is empty"
    assert len(provider.requests) == 2


async def test_a_child_provider_error_is_an_error_result_and_the_parent_goes_on(ctx: ToolContext) -> None:
    agent, _, _ = _make(ctx, [_task(), [ProviderError("model fell over")], _say("recovered")])
    events = await _run(agent)

    finished = next(e for e in events if isinstance(e, ToolFinished) and e.call_id == "t1")
    assert finished.result.is_error
    assert "model fell over" in finished.result.output
    assert agent.history[-1].content == "recovered"
    assert [e for e in events if isinstance(e, TurnEnded)][-1].status == "done"


class _Extra(Tool):
    name: ClassVar[str] = "ext_echo"
    description: ClassVar[str] = "an extra tool"
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}, "required": []}
    needs_approval: ClassVar[bool] = True

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        async def run() -> ToolResult:
            return ToolResult("echoed", False, "ok")

        return Prepared(None, run)


def _agents(*items: AgentDef) -> Agents:
    return Agents({a.name: a for a in items}, ())


def _def(name: str, prompt: str = "You are the reviewer.", tools: tuple[str, ...] | None = None) -> AgentDef:
    return AgentDef(name, f"{name} agent", prompt, tools, None, None, "global", Path("/x") / f"{name}.md")


def _agent_task(agent: object, call_id: str = "t1", prompt: str = "p") -> list[ProviderEvent | Gate]:
    return _call(call_id, "task", description="review", prompt=prompt, agent=agent)


async def test_ac8_no_agent_empty_or_default_run_todays_child(ctx: ToolContext) -> None:
    baseline, base_provider, _ = _make(ctx, [_task(prompt="p"), _say("report"), _say("done")])
    await _run(baseline)
    child_request = base_provider.requests[1]
    assert child_request[0][0].content.startswith("You are a research subagent.")
    expected = (child_request, _tool_message(baseline, "t1"))
    assert expected[1] == f"{REPORT_HEADER}\nreport"

    agents = _agents(_def("reviewer"))
    for spelling in (None, "", "default"):
        args: dict[str, object] = {"description": "look around", "prompt": "p"}
        if spelling is not None:
            args["agent"] = spelling
        call: list[ProviderEvent | Gate] = [ToolCallEvent(ToolCall("t1", "task", args)), Done(10, 5)]
        agent, provider, _ = _make(ctx, [call, _say("report"), _say("done")], agents=agents)
        await _run(agent)
        assert (provider.requests[1], _tool_message(agent, "t1")) == expected


async def test_ac9_a_custom_agent_runs_with_its_prompt_and_header(ctx: ToolContext) -> None:
    agents = _agents(_def("reviewer", prompt="You are the reviewer.\nBe strict."))
    agent, provider, _ = _make(ctx, [_agent_task("reviewer"), _say("found 2"), _say("done")], agents=agents)
    await _run(agent)

    messages, _tools = provider.requests[1]
    assert messages[0].role == "system"
    assert messages[0].content.startswith("You are the reviewer.\nBe strict.\n\nWorkspace:")
    assert "research subagent" not in messages[0].content
    assert "Available agents" not in messages[0].content
    assert [(m.role, m.content) for m in messages[1:]] == [("user", "p")]
    assert _tool_message(agent, "t1") == f"{AGENT_REPORT_HEADER.format(name='reviewer')}\nfound 2"


async def test_the_main_prompt_lists_the_agents(ctx: ToolContext) -> None:
    agents = _agents(_def("reviewer"))
    agent, provider, _ = _make(ctx, [_say("hi")], agents=agents)
    await _run(agent)
    assert "- reviewer: reviewer agent" in provider.requests[0][0][0].content


async def test_ac6_an_agent_without_tools_gets_todays_child_tools(ctx: ToolContext) -> None:
    agent, provider, _ = _make(ctx, [_agent_task("reviewer"), _say("r"), _say("done")], agents=_agents(_def("reviewer")))
    await _run(agent)
    assert provider.requests[1][1] == ["read", "glob", "grep"]


async def test_ac5_the_listed_tools_are_taken_in_order_and_unavailable_ones_dropped(ctx: ToolContext) -> None:
    tools = ("grep", "web_fetch", "read", "write")  # no fetcher; write is a Phase 2 tool
    agent, provider, _ = _make(
        ctx, [_agent_task("a"), _say("r"), _say("done")], agents=_agents(_def("a", tools=tools))
    )
    await _run(agent)
    assert provider.requests[1][1] == ["grep", "read"]


async def test_a_custom_agent_with_no_valid_tools_gets_none(ctx: ToolContext) -> None:
    agent, provider, _ = _make(ctx, [_agent_task("a"), _say("r"), _say("done")], agents=_agents(_def("a", tools=())))
    await _run(agent)
    assert provider.requests[1][1] == []


async def test_ac10_an_unknown_agent_is_an_error_and_starts_no_child(ctx: ToolContext) -> None:
    agents = _agents(_def("b"), _def("a"))
    agent, provider, _ = _make(ctx, [_agent_task("c\x1b"), _say("done")], agents=agents)
    await _run(agent)
    assert _tool_message(agent, "t1") == "task: unknown agent 'c\\x1b' (available: a, b)"
    assert len(provider.requests) == 2


async def test_an_unknown_agent_with_none_loaded_says_none(ctx: ToolContext) -> None:
    agent, _, _ = _make(ctx, [_agent_task("c"), _say("done")])
    await _run(agent)
    assert _tool_message(agent, "t1") == "task: unknown agent 'c' (available: none)"


async def test_a_non_string_agent_is_refused(ctx: ToolContext) -> None:
    agent, provider, _ = _make(ctx, [_agent_task(3), _say("done")])
    await _run(agent)
    assert "must be a string" in _tool_message(agent, "t1")  # the loop's own argument check
    assert len(provider.requests) == 2


def test_the_label_names_a_custom_agent_only(ctx: ToolContext) -> None:
    agent, _, _ = _make(ctx, [])
    tool = next(t for t in agent.tools if isinstance(t, TaskTool))
    assert tool.label({"description": "review", "agent": "reviewer"}) == "task review (reviewer)"
    assert tool.label({"description": "review", "agent": "re\x1bv"}) == "task review (re\\x1bv)"
    for spelling in (None, "", "default"):
        assert tool.label({"description": "review", "agent": spelling}) == "task review"
    assert "agent" in TaskTool.parameters["properties"]
    assert TaskTool.parameters["required"] == ["description", "prompt"]


async def test_ac24_a_custom_agent_failures_match_todays_results(ctx: ToolContext) -> None:
    agents = _agents(_def("a"))
    agent, _, _ = _make(ctx, [_agent_task("a"), [ProviderError("model fell over")], _say("ok")], agents=agents)
    events = await _run(agent)
    finished = next(e for e in events if isinstance(e, ToolFinished) and e.call_id == "t1")
    assert finished.result.is_error and finished.result.output == "task failed: model fell over"

    (ctx.workspace / "a.txt").write_text("a")
    looking = _call("r", "read", path="a.txt")
    agent, _, _ = _make(ctx, [_agent_task("a"), looking, looking, _say("done")], max_steps=2, agents=agents)
    await _run(agent)
    assert _tool_message(agent, "t1") == (
        f"{AGENT_REPORT_HEADER.format(name='a')}\n[step limit (2) reached before the subagent finished]"
    )
