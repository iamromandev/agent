"""web_search inside the loop: no approval, also in plan mode, and deny rules still refuse it."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.engine.approval import Choice, Decision
from src.engine.events import AgentEvent
from src.engine.loop import Agent
from src.engine.mcp.trust import TrustStore
from src.engine.mode import Mode
from src.engine.permissions.setup import Rules, prepare
from src.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.search_web import SearchHit
from src.engine.tools import default_tools
from src.engine.tools.base import ToolContext

from tests.fakes import FakeSearcher, Gate, MemoryRecorder, RecordingApprover, ScriptedProvider


def _make(
    ctx: ToolContext, searcher: FakeSearcher, rules: Rules | None = None
) -> tuple[Agent, ScriptedProvider, RecordingApprover]:
    script: list[list[ProviderEvent | Gate]] = [
        [ToolCallEvent(ToolCall("c1", "web_search", {"query": "textual"})), Done(10, 5)],
        [TextDelta("done"), Done(10, 5)],
    ]
    provider = ScriptedProvider(script, context_window=1000)
    approver = RecordingApprover([Decision(Choice.DENY)])  # a dialog would be refused, and is recorded
    agent = Agent(
        provider,
        default_tools(searcher=searcher),
        approver,
        ctx,
        25,
        today=lambda: date(2026, 11, 14),
        recorder=MemoryRecorder(),
        rules=rules,
    )
    return agent, provider, approver


def _ignore(_: AgentEvent) -> None:
    return None


def _tool_output(agent: Agent) -> str:
    return next(m for m in agent.history if m.role == "tool").content


async def test_plan_mode_offers_web_search_and_runs_it_with_no_approval(ctx: ToolContext) -> None:
    searcher = FakeSearcher([SearchHit("Textual", "https://textual.textualize.io", "TUI framework")])
    agent, provider, approver = _make(ctx, searcher)
    agent.set_mode(Mode.PLAN)

    await agent.run_turn("find it", _ignore)

    assert "web_search" in provider.requests[0][1]
    assert approver.requests == []
    assert searcher.calls == [("textual", 5)]
    assert "https://textual.textualize.io" in _tool_output(agent)


async def test_a_deny_rule_refuses_web_search_without_a_search(ctx: ToolContext, tmp_path: Path) -> None:
    settings = tmp_path / "config" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"permissions": {"deny": ["web_search"]}}))
    rules = prepare(ctx.workspace, settings, TrustStore(tmp_path / "data" / "trust.json")).rules
    searcher = FakeSearcher([SearchHit("t", "https://e.com", "")])
    agent, _, approver = _make(ctx, searcher, rules)

    await agent.run_turn("find it", _ignore)

    assert searcher.calls == [] and approver.requests == []
    assert _tool_output(agent) == "denied by rule web_search (global)"


async def test_a_valued_deny_rule_on_the_query_refuses_that_query(ctx: ToolContext, tmp_path: Path) -> None:
    settings = tmp_path / "config" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"permissions": {"deny": ["web_search(textual)"]}}))
    rules = prepare(ctx.workspace, settings, TrustStore(tmp_path / "data" / "trust.json")).rules
    searcher = FakeSearcher([])
    agent, _, _ = _make(ctx, searcher, rules)

    await agent.run_turn("find it", _ignore)

    assert searcher.calls == []
    assert _tool_output(agent).startswith("denied by rule web_search(textual)")
