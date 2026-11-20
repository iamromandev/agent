"""The `task` tool: a read-only child agent whose final report is all the parent keeps."""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar

from src.engine.approval import Decision
from src.engine.events import AgentEvent, ToolFinished, TurnEnded
from src.engine.loop import Agent
from src.engine.provider.base import Done, ProviderError, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.task import REPORT_HEADER
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
