"""Protected paths inside the loop: a targeted read, glob or grep always opens a dialog (R4, R7)."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pytest
from src.engine.approval import Choice, Decision
from src.engine.events import AgentEvent, ToolFinished
from src.engine.guardrails.setup import prepare as prepare_guard
from src.engine.loop import Agent
from src.engine.mcp.trust import TrustStore
from src.engine.mode import Mode
from src.engine.permissions.setup import Rules, prepare
from src.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.tools import default_tools, read_only_tools
from src.engine.tools.base import ToolContext

from tests.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider

Script = list[list[ProviderEvent | Gate]]


def _call(call_id: str, name: str, **args: object) -> list[ProviderEvent | Gate]:
    return [ToolCallEvent(ToolCall(call_id, name, dict(args))), Done(10, 5)]


def _done() -> list[ProviderEvent | Gate]:
    return [TextDelta("done"), Done(1, 1)]


@pytest.fixture
def home(tmp_path: Path) -> Path:
    folder = tmp_path / "home"
    (folder / ".ssh").mkdir(parents=True)
    (folder / ".ssh" / "config").write_text("Host secret-host\n")
    return folder.resolve()


@pytest.fixture
def gctx(workspace: Path, home: Path) -> ToolContext:
    return ToolContext(workspace, 5.0, guard=prepare_guard(workspace, None, None, {}, "Linux", home=home))


def _agent(
    ctx: ToolContext, script: Script, decisions: list[Decision] | None = None, rules: Rules | None = None, task: bool = False
) -> tuple[Agent, ScriptedProvider, RecordingApprover]:
    provider = ScriptedProvider(script, context_window=100_000)
    approver = RecordingApprover(decisions)
    agent = Agent(
        provider,
        default_tools(),
        approver,
        ctx,
        25,
        today=lambda: date(2026, 11, 14),
        recorder=MemoryRecorder(),
        rules=rules,
        task_tools=read_only_tools() if task else None,
    )
    return agent, provider, approver


async def _run(agent: Agent) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    await agent.run_turn("go", events.append)
    return events


def _tool_message(agent: Agent, call_id: str) -> str:
    return next(m.content for m in agent.history if m.role == "tool" and m.tool_call_id == call_id)


def _allow_read(tmp_path: Path, ctx: ToolContext) -> Rules:
    path = tmp_path / "config" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"permissions": {"allow": ["read"]}}))
    return prepare(ctx.workspace, path, TrustStore(tmp_path / "trust.json")).rules


async def test_ac3_a_protected_read_asks_even_when_read_is_allowed_and_granted(
    gctx: ToolContext, home: Path, tmp_path: Path
) -> None:
    target = str(home / ".ssh" / "config")
    script = [_call("c1", "read", path=target), _call("c2", "read", path=target), _call("c3", "read", path=target), _done()]
    agent, _, approver = _agent(
        gctx, script, [Decision(Choice.DENY), Decision(Choice.ONCE), Decision(Choice.ALWAYS)], _allow_read(tmp_path, gctx)
    )
    agent.grants.add("read", {"path": target})

    await _run(agent)

    assert len(approver.requests) == 3  # a second read asks again, and "Always" is not offered or kept
    first = approver.requests[0]
    assert first.tool == "read"
    assert first.protected is True
    assert first.protected_path == target
    assert first.allow_always is False
    assert first.save_rule is None
    assert first.allow_save is False
    assert _tool_message(agent, "c1").startswith("User denied this call.")
    assert "Host secret-host" in _tool_message(agent, "c2")
    assert "Host secret-host" in _tool_message(agent, "c3")  # a stray Always answer counts as once


async def test_a_protected_read_that_is_denied_does_not_read_the_file(gctx: ToolContext, home: Path) -> None:
    agent, _, _ = _agent(gctx, [_call("c1", "read", path=str(home / ".ssh" / "config")), _done()], [Decision(Choice.DENY)])

    await _run(agent)

    assert "secret-host" not in "".join(m.content for m in agent.history if m.role == "tool")


async def test_a_relative_path_with_dotdot_and_a_symlink_are_matched_after_resolution(
    gctx: ToolContext, home: Path, workspace: Path
) -> None:
    (workspace / "link").symlink_to(home / ".ssh" / "config")
    (workspace / ".env").write_text("A=1\n")
    script = [
        _call("c1", "read", path="link"),
        _call("c2", "read", path="sub/../.env"),
        _done(),
    ]
    (workspace / "sub").mkdir()
    agent, _, approver = _agent(gctx, script, [Decision(Choice.ONCE), Decision(Choice.ONCE)])

    await _run(agent)

    assert [r.protected for r in approver.requests] == [True, True]


async def test_an_ordinary_read_still_needs_no_dialog(gctx: ToolContext, workspace: Path) -> None:
    (workspace / "a.txt").write_text("hello\n")
    agent, _, approver = _agent(gctx, [_call("c1", "read", path="a.txt"), _done()])

    await _run(agent)

    assert approver.requests == []
    assert "hello" in _tool_message(agent, "c1")


async def test_a_protected_path_that_does_not_exist_gets_the_normal_not_found_error(gctx: ToolContext, home: Path) -> None:
    agent, _, approver = _agent(gctx, [_call("c1", "read", path=str(home / ".ssh" / "missing")), _done()])

    await _run(agent)

    assert approver.requests == []
    assert _tool_message(agent, "c1").startswith("No such file")


async def test_a_template_env_file_is_read_without_a_dialog(gctx: ToolContext, workspace: Path) -> None:
    (workspace / ".env.example").write_text("A=\n")
    agent, _, approver = _agent(gctx, [_call("c1", "read", path=".env.example"), _done()])

    await _run(agent)

    assert approver.requests == []


async def test_a_grep_of_one_protected_file_asks_and_a_grep_of_a_folder_does_not(
    gctx: ToolContext, home: Path, workspace: Path
) -> None:
    (workspace / "app.py").write_text("Host in app\n")
    script = [
        _call("c1", "grep", pattern="Host", path=str(home / ".ssh" / "config")),
        _call("c2", "grep", pattern="Host"),
        _call("c3", "grep", pattern="Host", path=str(home / ".ssh")),
        _done(),
    ]
    agent, _, approver = _agent(gctx, script, [Decision(Choice.ONCE)])

    await _run(agent)

    assert [r.label for r in approver.requests] == ["grep Host"]
    assert "secret-host" in _tool_message(agent, "c1")
    assert _tool_message(agent, "c2") == "app.py:1: Host in app"
    assert _tool_message(agent, "c3") == "No matches\n1 protected files skipped"


async def test_a_glob_whose_pattern_names_a_protected_file_asks_and_one_with_wildcards_skips(
    gctx: ToolContext, workspace: Path
) -> None:
    (workspace / ".env").write_text("A=1\n")
    (workspace / "app.py").write_text("x\n")
    script = [_call("c1", "glob", pattern=".env"), _call("c2", "glob", pattern="*"), _done()]
    agent, _, approver = _agent(gctx, script, [Decision(Choice.DENY)])

    await _run(agent)

    assert [r.label for r in approver.requests] == ["glob .env"]
    assert _tool_message(agent, "c1").startswith("User denied this call.")
    assert _tool_message(agent, "c2") == "app.py\n1 protected files skipped"


async def test_ac5_a_deny_rule_refuses_a_protected_read_without_a_dialog(
    gctx: ToolContext, home: Path, tmp_path: Path
) -> None:
    path = tmp_path / "config" / "settings.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"permissions": {"deny": ["read"]}}))
    rules = prepare(gctx.workspace, path, TrustStore(tmp_path / "trust.json")).rules
    agent, _, approver = _agent(gctx, [_call("c1", "read", path=str(home / ".ssh" / "config")), _done()], rules=rules)

    await _run(agent)

    assert approver.requests == []
    assert _tool_message(agent, "c1") == "denied by rule read (global)"


async def test_ac5_plan_mode_still_shows_the_protected_dialog(gctx: ToolContext, home: Path) -> None:
    agent, _, approver = _agent(gctx, [_call("c1", "read", path=str(home / ".ssh" / "config")), _done()], [Decision(Choice.DENY)])
    agent.set_mode(Mode.PLAN)

    await _run(agent)

    assert [r.protected for r in approver.requests] == [True]
    assert approver.requests[0].allow_always is False


async def test_the_finished_event_of_a_denied_protected_read_is_an_error(gctx: ToolContext, home: Path) -> None:
    agent, _, _ = _agent(gctx, [_call("c1", "read", path=str(home / ".ssh" / "config")), _done()], [Decision(Choice.DENY)])

    events = await _run(agent)

    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.result.is_error is True
    assert finished.result.summary == "denied"
    assert finished.applied is False


async def test_ac8_a_task_subagent_reading_a_protected_path_triggers_the_dialog(gctx: ToolContext, home: Path) -> None:
    script: Script = [
        _call("t1", "task", description="look", prompt="read the ssh config"),
        _call("r1", "read", path=str(home / ".ssh" / "config")),  # the child's request
        [TextDelta("refused"), Done(1, 1)],  # the child's report
        _done(),
    ]
    agent, provider, approver = _agent(gctx, script, [Decision(Choice.DENY)], task=True)

    await _run(agent)

    assert [r.protected for r in approver.requests] == [True]
    child_request = provider.requests[2][0]
    assert "secret-host" not in "".join(m.content for m in child_request)


async def test_a_hard_link_to_a_protected_file_in_the_workspace_asks_before_a_read(
    gctx: ToolContext, home: Path, workspace: Path
) -> None:
    os.link(home / ".ssh" / "config", workspace / "notes.txt")
    agent, _, approver = _agent(gctx, [_call("c1", "read", path="notes.txt"), _done()], [Decision(Choice.DENY)])

    await _run(agent)

    assert [r.protected for r in approver.requests] == [True]
    assert "secret-host" not in _tool_message(agent, "c1")
