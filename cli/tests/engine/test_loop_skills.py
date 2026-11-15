"""skill inside the loop: prompt section, no approval, plan mode, deny rules, the task child, resume."""

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
from src.engine.provider.base import Done, Message, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.skills.loader import Skill, Skills
from src.engine.tools import default_tools, read_only_tools
from src.engine.tools.base import ToolContext

from tests.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider

SKILLS = Skills(
    {
        "release": Skill("release", "Cut a release.", "RELEASE BODY", "project", Path("/w/.agent/skills/release")),
        "review": Skill("review", "House review.", "REVIEW BODY", "global", Path("/g/review")),
    },
    (),
)


def _ignore(_: AgentEvent) -> None:
    return None


def _call(name: str = "release", call_id: str = "c1") -> list[ProviderEvent | Gate]:
    return [ToolCallEvent(ToolCall(call_id, "skill", {"name": name})), Done(10, 5)]


def _say(text: str = "done") -> list[ProviderEvent | Gate]:
    return [TextDelta(text), Done(10, 5)]


def _make(
    ctx: ToolContext,
    script: list[list[ProviderEvent | Gate]],
    skills: Skills | None = SKILLS,
    rules: Rules | None = None,
    task: bool = False,
) -> tuple[Agent, ScriptedProvider, RecordingApprover]:
    provider = ScriptedProvider(script, context_window=100_000)
    approver = RecordingApprover([Decision(Choice.DENY)])  # a dialog would be refused, and is recorded
    agent = Agent(
        provider,
        default_tools(skills=skills),
        approver,
        ctx,
        25,
        today=lambda: date(2026, 11, 14),
        recorder=MemoryRecorder(),
        rules=rules,
        task_tools=read_only_tools(skills=skills) if task else None,
    )
    return agent, provider, approver


def _tool_output(agent: Agent) -> str:
    return next(m for m in agent.history if m.role == "tool").content


async def test_ac7_with_skills_the_prompt_has_the_section_and_the_tool_is_offered(ctx: ToolContext) -> None:
    agent, provider, _ = _make(ctx, [_say()])
    await agent.run_turn("hi", _ignore)

    messages, tools = provider.requests[0]
    assert "skill" in tools
    assert "- release: Cut a release." in messages[0].content
    assert "- review: House review." in messages[0].content


async def test_ac7_without_skills_there_is_no_section_and_no_tool(ctx: ToolContext) -> None:
    agent, provider, _ = _make(ctx, [_say()], skills=None)
    await agent.run_turn("hi", _ignore)

    messages, tools = provider.requests[0]
    assert "skill" not in tools and "Available skills" not in messages[0].content


async def test_the_section_counts_in_the_estimate(ctx: ToolContext) -> None:
    with_skills, _, _ = _make(ctx, [_say()])
    without, _, _ = _make(ctx, [_say()], skills=None)
    assert with_skills.estimate() > without.estimate()


async def test_ac9_plan_mode_runs_skill_with_no_approval(ctx: ToolContext) -> None:
    agent, provider, approver = _make(ctx, [_call(), _say()])
    agent.set_mode(Mode.PLAN)

    await agent.run_turn("go", _ignore)

    assert "skill" in provider.requests[0][1]
    assert approver.requests == []
    assert "RELEASE BODY" in _tool_output(agent)


async def test_ac9_a_deny_rule_refuses_the_skill(ctx: ToolContext, tmp_path: Path) -> None:
    settings = tmp_path / "config" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"permissions": {"deny": ["skill(release)"]}}))
    rules = prepare(ctx.workspace, settings, TrustStore(tmp_path / "data" / "trust.json")).rules
    agent, _, approver = _make(ctx, [_call(), _say()], rules=rules)

    await agent.run_turn("go", _ignore)

    assert approver.requests == []
    assert _tool_output(agent).startswith("denied by rule skill(release)")


async def test_ac12_the_task_child_gets_the_tool_and_the_section(ctx: ToolContext) -> None:
    task_call: list[ProviderEvent | Gate] = [
        ToolCallEvent(ToolCall("t1", "task", {"description": "d", "prompt": "look"})),
        Done(10, 5),
    ]
    agent, provider, _ = _make(ctx, [task_call, _say("report"), _say("final")], task=True)

    await agent.run_turn("go", _ignore)

    child_messages, child_tools = provider.requests[1]
    assert "research subagent" in child_messages[0].content
    assert "skill" in child_tools
    assert "- release: Cut a release." in child_messages[0].content


async def test_ac16_a_resumed_history_with_a_skill_call_works_after_the_skill_is_gone(ctx: ToolContext) -> None:
    history = [
        Message("user", "cut a release"),
        Message("assistant", "", tool_calls=[ToolCall("c0", "skill", {"name": "release"})]),
        Message("tool", "Skill 'release' ...", tool_call_id="c0"),
        Message("assistant", "done"),
    ]
    other = Skills({"other": SKILLS.by_name["review"]}, ())
    agent, provider, _ = _make(ctx, [_call("release", "c1"), _say()], skills=other)
    agent.reset(recorder=MemoryRecorder(), history=history)

    await agent.run_turn("again", _ignore)

    assert provider.requests[0][0][1:5] == history
    assert "Unknown skill 'release'. Available: review" in _tool_output_last(agent)


def _tool_output_last(agent: Agent) -> str:
    return [m for m in agent.history if m.role == "tool"][-1].content
