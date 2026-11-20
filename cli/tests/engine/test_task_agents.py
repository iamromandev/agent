"""A custom subagent with write tools: the parent's undo log, gates and protected folders all apply to it."""

from __future__ import annotations

import asyncio
import os
from datetime import date
from pathlib import Path

import pytest
from src.engine.agents.loader import AgentDef, Agents
from src.engine.approval import Choice, Decision
from src.engine.events import AgentEvent, Notice, TaskProgress
from src.engine.guardrails.setup import prepare as prepare_guardrails
from src.engine.hooks.config import DEFAULT_TIMEOUT, Event, HookEntry
from src.engine.hooks.setup import Hooks
from src.engine.loop import Agent
from src.engine.mcp.trust import TrustStore
from src.engine.mode import Mode
from src.engine.permissions.setup import Rules, prepare
from src.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.task import TaskTool
from src.engine.tools import default_tools, read_only_tools
from src.engine.tools.base import ToolContext
from src.engine.undo import UndoLog

from tests.fakes import FakeRunner, Gate, MemoryRecorder, RecordingApprover, ScriptedProvider, exit_run, ok_run

ONCE = Decision(Choice.ONCE)
ALWAYS = Decision(Choice.ALWAYS)
DENY = Decision(Choice.DENY)
Script = list[list[ProviderEvent | Gate]]


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
    return ToolContext(workspace=workspace, bash_timeout=5.0)


def _call(call_id: str, name: str, /, **args: object) -> list[ProviderEvent | Gate]:
    return [ToolCallEvent(ToolCall(call_id, name, dict(args))), Done(10, 5)]


def _say(text: str) -> list[ProviderEvent | Gate]:
    return [TextDelta(text), Done(1, 1)]


def _def(name: str, tools: tuple[str, ...] | None) -> AgentDef:
    return AgentDef(name, f"{name} agent", "You are the agent.", tools, None, None, "project", Path("/x") / f"{name}.md")


def _delegate(agent: str = "w") -> list[ProviderEvent | Gate]:
    return _call("t1", "task", description="work", prompt="p", agent=agent)


def _make(
    ctx: ToolContext,
    script: Script,
    decisions: list[Decision] | None = None,
    *,
    tools: tuple[str, ...] = ("read", "write", "edit", "bash"),
    mode: Mode = Mode.BUILD,
    rules: Rules | None = None,
    hooks: Hooks | None = None,
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
        task_tools=read_only_tools(None),
        agents=Agents({"w": _def("w", tools)}, ()),
        rules=rules,
        hooks=hooks,
    )
    agent.set_mode(mode)
    return agent, provider, approver


async def _run(agent: Agent, text: str = "go") -> list[AgentEvent]:
    events: list[AgentEvent] = []
    await agent.run_turn(text, events.append)
    return events


def _tool_message(agent: Agent, call_id: str) -> str:
    return next(m.content for m in agent.history if m.role == "tool" and m.tool_call_id == call_id)


def _peeked_prompt(log: UndoLog) -> str | None:
    turn = log.peek()
    return turn.prompt if turn is not None else None


# Step 7: the child's write and edit land in the parent's turn

async def test_ac18_a_child_write_and_edit_are_undone_with_the_parent_turn(ctx: ToolContext, workspace: Path) -> None:
    script = [
        _delegate(),
        _call("w1", "write", path="n.txt", content="made\n"),
        _call("e1", "edit", path="a.py", old_string="hi", new_string="bye"),
        _say("report"),
        _say("done"),
    ]
    agent, _, _ = _make(ctx, script, [ONCE, ONCE])
    await _run(agent, "do it")
    assert (workspace / "n.txt").exists() and "bye" in (workspace / "a.py").read_text()
    turn = agent.undo_log.peek()
    assert turn is not None and turn.prompt == "do it"

    events: list[AgentEvent] = []
    prompt = await agent.undo(events.append)

    assert prompt == "do it"
    assert not (workspace / "n.txt").exists()
    assert (workspace / "a.py").read_text() == "print('hi')\n"
    assert agent.undo_log.peek() is None  # one parent turn, not two


async def test_the_child_does_not_open_a_turn_of_its_own(ctx: ToolContext) -> None:
    agent, _, _ = _make(ctx, [_delegate(), _call("w1", "write", path="n.txt", content="x"), _say("r"), _say("done")], [ONCE])
    await _run(agent)
    log = agent.undo_log
    assert log.pop() is not None
    assert log.pop() is None


async def test_the_default_child_keeps_its_own_undo_log(ctx: ToolContext) -> None:
    agent, provider, _ = _make(ctx, [_call("t1", "task", description="d", prompt="p"), _say("r"), _say("done")])
    await _run(agent)
    assert len(provider.requests) == 3
    assert agent.undo_log.peek() is not None  # the parent's own turn only


async def test_a_child_that_compacts_leaves_the_parents_undo_log_intact(ctx: ToolContext, workspace: Path) -> None:
    log = UndoLog(workspace)
    log.begin(0, "parent turn")
    log.finish()
    provider = ScriptedProvider([_say("one"), _say("summary of it"), _say("summary again")], context_window=100_000)
    child = Agent(
        provider,
        [],
        RecordingApprover(),
        ctx,
        5,
        today=lambda: date(2026, 11, 14),
        recorder=None,
        role="task",
        undo=log,
        subagent="w",
    )
    await child.compact(lambda event: None)  # nothing to compact: the empty-history path
    assert log.peek() is not None

    await child.run_turn("hello", lambda event: None)
    assert _peeked_prompt(log) == "parent turn"
    await child.compact(lambda event: None)  # a forced compaction with history
    assert _peeked_prompt(log) == "parent turn"


# Step 8: the dialog names the subagent

async def test_ac12_a_child_edit_asks_in_build_mode_and_names_the_subagent(ctx: ToolContext, workspace: Path) -> None:
    script = [_delegate(), _call("e1", "edit", path="a.py", old_string="hi", new_string="bye"), _say("r"), _say("done")]
    agent, _, approver = _make(ctx, script, [ONCE])
    await _run(agent)
    assert [r.subagent for r in approver.requests] == ["w"]
    assert "bye" in (workspace / "a.py").read_text()


async def test_the_main_agents_own_request_names_no_subagent(ctx: ToolContext) -> None:
    agent, _, approver = _make(ctx, [_call("e1", "edit", path="a.py", old_string="hi", new_string="bye"), _say("done")], [ONCE])
    await _run(agent)
    assert [r.subagent for r in approver.requests] == [""]


async def test_a_child_protected_read_names_the_subagent(workspace: Path, tmp_path: Path) -> None:
    (workspace / ".env").write_text("K=v\n")
    guard = prepare_guardrails(workspace, None, None, {}, "Linux", home=tmp_path / "home")
    ctx = ToolContext(workspace, 5.0, guard=guard)
    script = [_delegate(), _call("r1", "read", path=".env"), _say("r"), _say("done")]
    agent, _, approver = _make(ctx, script, [DENY])
    await _run(agent)
    assert [(r.protected, r.subagent) for r in approver.requests] == [(True, "w")]


# Step 10: the pool is the parent's own instances

async def test_the_child_uses_the_parents_own_tool_instances(ctx: ToolContext) -> None:
    agent, provider, _ = _make(ctx, [_delegate(), _say("r"), _say("done")])
    await _run(agent)
    assert provider.requests[1][1] == ["read", "write", "edit", "bash"]
    task = next(t for t in agent.tools if isinstance(t, TaskTool))
    pool = task.agent_pool(task.agents.get("w"))  # type: ignore[arg-type]
    parent_by_name = {t.name: t for t in agent.tools}
    assert pool[1] is parent_by_name["write"] and pool[3] is parent_by_name["bash"]


async def test_tools_the_parent_does_not_have_are_dropped_silently(ctx: ToolContext) -> None:
    agent, provider, _ = _make(ctx, [_delegate(), _say("r"), _say("done")], tools=("web_fetch", "grep", "write"))
    await _run(agent)
    assert provider.requests[1][1] == ["grep", "write"]


async def test_ac12_in_plan_mode_the_child_is_not_offered_write_tools_and_a_forced_call_is_refused(
    ctx: ToolContext, workspace: Path
) -> None:
    script = [_delegate(), _call("w1", "write", path="n.txt", content="x"), _say("r"), _say("done")]
    agent, provider, approver = _make(ctx, script, mode=Mode.PLAN)
    await _run(agent)
    assert provider.requests[1][1] == ["read"]
    assert approver.requests == [] and not (workspace / "n.txt").exists()
    child_results = [m for m in provider.requests[2][0] if m.role == "tool"]
    assert child_results and "plan" in child_results[0].content.lower()


async def test_ac12_in_edits_mode_a_child_edit_runs_without_the_dialog(ctx: ToolContext, workspace: Path) -> None:
    script = [_delegate(), _call("e1", "edit", path="a.py", old_string="hi", new_string="bye"), _say("r"), _say("done")]
    agent, _, approver = _make(ctx, script, mode=Mode.EDITS)
    await _run(agent)
    assert approver.requests == []
    assert "bye" in (workspace / "a.py").read_text()


async def test_ac13_a_deny_rule_stops_the_child_bash(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    settings = tmp_path / "config" / "settings.json"
    settings.parent.mkdir()
    settings.write_text('{"permissions": {"deny": ["bash(rm*)"]}}')
    rules = prepare(workspace, settings, TrustStore(tmp_path / "trust.json")).rules
    (workspace / "x").write_text("x")
    script = [_delegate(), _call("b1", "bash", command="rm x"), _say("r"), _say("done")]
    agent, provider, approver = _make(ctx, script, rules=rules)
    await _run(agent)
    assert approver.requests == [] and (workspace / "x").exists()
    tool_results = [m.content for m in provider.requests[2][0] if m.role == "tool"]
    assert tool_results and tool_results[0].startswith("denied by rule")


async def test_ac13_an_always_in_the_child_covers_the_parent(ctx: ToolContext) -> None:
    script = [
        _delegate(),
        _call("b1", "bash", command="ls"),
        _say("r"),
        _call("b2", "bash", command="ls"),
        _say("done"),
    ]
    agent, _, approver = _make(ctx, script, [ALWAYS])
    await _run(agent)
    assert len(approver.requests) == 1 and approver.requests[0].subagent == "w"
    assert agent.grants.exact  # the child shares the parent's grants


async def test_ac14_a_child_bash_is_the_parents_shell_tool_with_the_same_guard(ctx: ToolContext) -> None:
    agent, _, _ = _make(ctx, [])
    task = next(t for t in agent.tools if isinstance(t, TaskTool))
    pool = task.agent_pool(task.agents.get("w"))  # type: ignore[arg-type]
    shell = next(t for t in pool if t.name == "bash")
    assert shell is next(t for t in agent.tools if t.name == "bash")


# Step 9: `.agent/agents` is guarded

async def test_a_child_write_under_agent_agents_asks_without_always(ctx: ToolContext, workspace: Path) -> None:
    script = [_delegate(), _call("w1", "write", path=".agent/agents/x.md", content="x"), _say("r"), _say("done")]
    agent, _, approver = _make(ctx, script, [DENY], mode=Mode.EDITS)
    await _run(agent)
    assert len(approver.requests) == 1
    request = approver.requests[0]
    assert request.allow_always is False and request.save_rule is None and request.subagent == "w"
    assert not (workspace / ".agent" / "agents" / "x.md").exists()


# Live mode: the child reads the parent's mode at every step

async def _flip_mid_child(
    ctx: ToolContext, start: Mode, to: Mode, decisions: list[Decision] | None = None
) -> tuple[Agent, ScriptedProvider, RecordingApprover]:
    gate = Gate()
    script: Script = [
        _delegate(),
        [gate, *_call("r1", "read", path="a.py")],  # step 1 of the child: the mode flips while it is answering
        _call("w1", "write", path="n.txt", content="x"),  # step 2: the forced or the offered write
        _say("r"),
        _say("done"),
    ]
    agent, provider, approver = _make(ctx, script, decisions, mode=start)
    task = asyncio.ensure_future(_run(agent))
    await gate.reached.wait()
    agent.set_mode(to)
    gate.release.set()
    await task
    return agent, provider, approver


async def test_a_tab_to_plan_mode_mid_child_stops_its_next_write(ctx: ToolContext, workspace: Path) -> None:
    _, provider, approver = await _flip_mid_child(ctx, Mode.BUILD, Mode.PLAN)
    assert "write" in provider.requests[1][1]  # step 1 ran in build mode
    assert provider.requests[2][1] == ["read"]  # step 2 is not offered write or bash
    assert approver.requests == [] and not (workspace / "n.txt").exists()
    assert any("plan mode" in m.content for m in provider.requests[3][0] if m.role == "tool")


async def test_a_tab_from_edits_to_build_mid_child_makes_the_next_write_ask(ctx: ToolContext, workspace: Path) -> None:
    _, _, approver = await _flip_mid_child(ctx, Mode.EDITS, Mode.BUILD, [ONCE])
    assert [r.subagent for r in approver.requests] == ["w"]
    assert (workspace / "n.txt").exists()


async def test_a_forced_bash_in_plan_mode_from_a_custom_child_is_refused(ctx: ToolContext) -> None:
    script = [_delegate(), _call("b1", "bash", command="echo hi > n.txt"), _say("r"), _say("done")]
    agent, provider, approver = _make(ctx, script, mode=Mode.PLAN)
    await _run(agent)
    assert approver.requests == [] and not (ctx.workspace / "n.txt").exists()
    results = [m.content for m in provider.requests[2][0] if m.role == "tool"]
    assert results == ["refused: bash is unavailable in plan mode"]


# Hard link to a file under .agent/agents

async def test_a_hard_link_to_an_agent_file_asks_without_always_even_after_a_grant(
    ctx: ToolContext, workspace: Path
) -> None:
    folder = workspace / ".agent" / "agents"
    folder.mkdir(parents=True)
    (folder / "x.md").write_text("old\n")
    os.link(folder / "x.md", workspace / "alias.md")
    script = [
        _call("w0", "write", path="other.txt", content="1"),  # an Always for write...
        _call("w1", "write", path="alias.md", content="2"),  # ...must not cover the alias
        _say("done"),
    ]
    agent, _, approver = _make(ctx, script, [ALWAYS, DENY])
    await _run(agent)
    assert len(approver.requests) == 2
    assert approver.requests[1].allow_always is False and approver.requests[1].save_rule is None
    assert (folder / "x.md").read_text() == "old\n"


# Hooks on a custom child's calls

def _hooks(workspace: Path, runner: FakeRunner, *entries: tuple[Event, str | None]) -> Hooks:
    items = [HookEntry(event, f"cmd-{event}", matcher, DEFAULT_TIMEOUT, "global") for event, matcher in entries]
    return Hooks(items, [], runner, lambda: True, workspace)


async def test_a_pretooluse_block_stops_a_child_bash_and_shows_a_notice(ctx: ToolContext, workspace: Path) -> None:
    runner = FakeRunner([exit_run(2, "nope")])
    hooks = _hooks(workspace, runner, ("PreToolUse", "bash"))
    script = [_delegate(), _call("b1", "bash", command="echo hi > n.txt"), _say("r"), _say("done")]
    agent, provider, approver = _make(ctx, script, hooks=hooks)
    events = await _run(agent)
    results = [m.content for m in provider.requests[2][0] if m.role == "tool"]
    assert results == ["blocked by hook: nope"]
    assert approver.requests == [] and not (workspace / "n.txt").exists()
    assert runner.calls[0][1]["agent"] == "w" and runner.calls[0][1]["tool"] == "bash"
    shown = [e.event.text for e in events if isinstance(e, TaskProgress) and isinstance(e.event, Notice)]
    assert any("blocked: nope" in text for text in shown)  # the app prints these as `task: <text>`


async def test_a_posttooluse_context_is_appended_to_the_child_result(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "a.txt").write_text("body\n")
    runner = FakeRunner([ok_run(stdout='{"context": "remember this"}')])
    hooks = _hooks(workspace, runner, ("PostToolUse", "read"))
    script = [_delegate(), _call("r1", "read", path="a.txt"), _say("r"), _say("done")]
    agent, provider, _ = _make(ctx, script, hooks=hooks)
    await _run(agent)
    results = [m.content for m in provider.requests[2][0] if m.role == "tool"]
    assert results[0].endswith("\n[hook PostToolUse] remember this")
    assert runner.calls[0][1]["agent"] == "w"


async def test_a_stop_hook_does_not_run_at_the_end_of_the_childs_turn(ctx: ToolContext, workspace: Path) -> None:
    runner = FakeRunner()
    hooks = _hooks(workspace, runner, ("Stop", None))
    agent, _, _ = _make(ctx, [_delegate(), _say("r"), _say("done")], hooks=hooks)
    await _run(agent)
    stops = [payload for _, payload, _ in runner.calls if payload["event"] == "Stop"]
    assert len(stops) == 1 and "agent" not in stops[0]  # the parent's own, once


async def test_the_default_child_runs_no_hooks(ctx: ToolContext, workspace: Path) -> None:
    runner = FakeRunner()
    hooks = _hooks(workspace, runner, ("PreToolUse", "read"), ("PostToolUse", "read"))
    (workspace / "a.txt").write_text("x")
    script = [
        _call("t1", "task", description="d", prompt="p"),
        _call("r1", "read", path="a.txt"),
        _say("r"),
        _say("done"),
    ]
    agent, _, _ = _make(ctx, script, hooks=hooks)
    await _run(agent)
    assert runner.calls == []


# Remaining gaps

async def test_a_child_bash_output_is_redacted_and_its_environment_filtered(workspace: Path, tmp_path: Path) -> None:
    guard = prepare_guardrails(workspace, None, None, {"MY_TOKEN": "supersecretvalue99"}, "Linux", home=tmp_path / "home")
    ctx = ToolContext(workspace, 5.0, guard=guard)
    command = "echo ghp_" + "a" * 36 + "; echo token=${MY_TOKEN:-unset}"
    script = [_delegate(), _call("b1", "bash", command=command), _say("r"), _say("done")]
    agent, provider, _ = _make(ctx, script, [ONCE])
    await _run(agent)
    output = next(m.content for m in provider.requests[2][0] if m.role == "tool")
    assert "[redacted:github-token]" in output and "ghp_" not in output
    assert "token=unset" in output  # the child environment filter strips *_TOKEN


async def test_an_always_in_the_parent_covers_the_childs_next_identical_call(ctx: ToolContext) -> None:
    script = [
        _call("b0", "bash", command="ls"),
        _delegate(),
        _call("b1", "bash", command="ls"),
        _say("r"),
        _say("done"),
    ]
    agent, _, approver = _make(ctx, script, [ALWAYS])
    await _run(agent)
    assert len(approver.requests) == 1 and approver.requests[0].subagent == ""
