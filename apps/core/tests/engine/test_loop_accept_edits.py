"""Accept-edits mode inside the loop: write and edit skip the dialog, except config files and `.git`."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import pytest
from core.engine.approval import Choice, Decision
from core.engine.events import AgentEvent, ToolFinished
from core.engine.guardrails.setup import prepare as prepare_guardrails
from core.engine.hooks.config import DEFAULT_TIMEOUT, HookEntry
from core.engine.hooks.setup import Hooks
from core.engine.loop import Agent
from core.engine.mcp.trust import TrustStore
from core.engine.mode import Mode
from core.engine.permissions.setup import Rules, prepare
from core.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from core.engine.terminal.manager import TerminalManager
from core.engine.tools import default_tools
from core.engine.tools.base import Prepared, Tool, ToolContext, ToolResult
from core.testing.fakes import (
    FakeFetcher,
    FakeRunner,
    Gate,
    MemoryRecorder,
    RecordingApprover,
    ScriptedProvider,
    exit_run,
)

Script = list[list[ProviderEvent | Gate]]
DENY = Decision(Choice.DENY)


def _call(call_id: str, name: str, /, **args: object) -> list[ProviderEvent | Gate]:
    return [ToolCallEvent(ToolCall(call_id, name, dict(args))), Done(10, 5)]


def _done() -> list[ProviderEvent | Gate]:
    return [TextDelta("done"), Done(1, 1)]


def _agent(
    ctx: ToolContext,
    script: Script,
    decisions: list[Decision] | None = None,
    *,
    mode: Mode = Mode.EDITS,
    rules: Rules | None = None,
    hooks: Hooks | None = None,
    fetcher: FakeFetcher | None = None,
    terminals: TerminalManager | None = None,
    extra: list[Tool] | None = None,
) -> tuple[Agent, ScriptedProvider, RecordingApprover]:
    provider = ScriptedProvider(script, context_window=100_000)
    approver = RecordingApprover(decisions)
    agent = Agent(
        provider,
        [*default_tools(terminals, fetcher=fetcher), *(extra or [])],
        approver,
        ctx,
        25,
        today=lambda: date(2026, 11, 14),
        recorder=MemoryRecorder(),
        rules=rules,
        hooks=hooks,
    )
    agent.set_mode(mode)
    return agent, provider, approver


async def _run(agent: Agent) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    await agent.run_turn("go", events.append)
    return events


def _tool_message(agent: Agent, call_id: str) -> str:
    return next(m.content for m in agent.history if m.role == "tool" and m.tool_call_id == call_id)


async def test_write_and_edit_run_without_a_dialog_and_undo_reverts_both(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text("old = 1\n")
    script = [
        _call("c1", "write", path="src/new.py", content="x = 1\n"),
        _call("c2", "edit", path="src/app.py", old_string="old = 1", new_string="old = 2"),
        _done(),
    ]
    agent, _, approver = _agent(ctx, script)

    events = await _run(agent)

    assert approver.requests == []
    assert (workspace / "src" / "new.py").read_text() == "x = 1\n"
    assert (workspace / "src" / "app.py").read_text() == "old = 2\n"
    finished = {e.call_id: e for e in events if isinstance(e, ToolFinished)}
    assert finished["c1"].preview is not None and finished["c2"].preview is not None
    assert finished["c2"].preview.kind == "diff"
    await agent.undo(events.append)
    assert not (workspace / "src" / "new.py").exists()
    assert (workspace / "src" / "app.py").read_text() == "old = 1\n"


def _make_git_hooks(workspace: Path) -> str:
    (workspace / ".git" / "hooks").mkdir(parents=True)
    return ".git/hooks/pre-commit"


def _make_upper_git(workspace: Path) -> str:
    (workspace / ".GIT").mkdir()
    return ".GIT/config"


def _make_git_file(workspace: Path) -> str:
    (workspace / "sub").mkdir()
    (workspace / "sub" / ".git").write_text("gitdir: x\n")
    return "sub/.git"


def _make_git_link(workspace: Path) -> str:
    (workspace / ".git").mkdir()
    os.symlink(workspace / ".git", workspace / "link")
    return "link/config"


@pytest.mark.parametrize(
    "path", [".agent/agents/x.md", ".AGENT/Agents/x.md", ".agent/workflows/x.json", ".Agent/WORKFLOWS/x.json"]
)
async def test_the_agent_files_folder_asks_without_always_in_edits_mode(
    ctx: ToolContext, workspace: Path, path: str
) -> None:
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=path, content="x\n"), _done()], [DENY])

    await _run(agent)

    _asks_without_always(approver)
    assert not (workspace / path).exists()


@pytest.mark.parametrize("path", [".agent/settings.json"])
async def test_the_agent_config_files_ask_without_always(ctx: ToolContext, workspace: Path, path: str) -> None:
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=path, content="{}\n"), _done()], [DENY])

    await _run(agent)

    assert len(approver.requests) == 1
    request = approver.requests[0]
    assert request.allow_always is False and request.save_rule is None
    assert not (workspace / path).exists()


# A real call for every built-in tool that needs approval, apart from write and edit. `term_stop` names a terminal
# the test starts first: stopping one that does not exist is refused before the approval step.
APPROVAL_CALLS: dict[str, dict[str, object]] = {
    "bash": {"command": "echo hi"},
    "term_start": {"name": "t", "command": "sleep 5"},
    "term_stop": {"name": "api"},
    "web_fetch": {"url": "https://example.com/"},
}


async def test_a_blocking_pretooluse_hook_still_wins(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "a.txt").write_text("one\n")
    hooks = Hooks(
        [HookEntry("PreToolUse", "x", "edit", DEFAULT_TIMEOUT, "global")],
        [],
        FakeRunner([exit_run(2, "nope")]),
        lambda: True,
        workspace,
    )
    script = [_call("c1", "edit", path="a.txt", old_string="one", new_string="two"), _done()]
    agent, _, approver = _agent(ctx, script, hooks=hooks)

    await _run(agent)

    assert _tool_message(agent, "c1") == "blocked by hook: nope"
    assert approver.requests == [] and (workspace / "a.txt").read_text() == "one\n"


@pytest.mark.parametrize("path", ["../outside.txt"])
async def test_the_tools_own_refusals_stand(ctx: ToolContext, workspace: Path, path: str) -> None:
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=path, content="x\n"), _done()])

    await _run(agent)

    assert approver.requests == []
    assert not (workspace / path).exists()
    assert _tool_message(agent, "c1")


def _allow(workspace: Path, tmp_path: Path, *allow: str) -> Rules:
    settings = tmp_path / "config" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({"permissions": {"allow": list(allow)}}))
    return prepare(workspace, settings, TrustStore(tmp_path / "data" / "trust.json")).rules


def _asks_without_always(approver: RecordingApprover) -> None:
    assert len(approver.requests) == 1
    request = approver.requests[0]
    assert request.allow_always is False and request.save_rule is None


# Finding 1: the config guard compares names case-insensitively (APFS), in every mode. Only the lower-case `.agent`
# folder exists, so on a case-sensitive file system the mixed-case name is another path, and the guard must still ask.
MIXED_CASE_CONFIG = [".agent/MCP.json", ".AGENT/settings.json", ".Agent/Settings.JSON"]


async def test_a_workflow_file_asks_without_always_even_when_a_rule_allows_write(
    ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    """`.agent/workflows` is guarded like `.agent/agents`: no allow rule, grant or mode approves a write there."""
    rules = _allow(workspace, tmp_path, "write")
    script = [_call("c1", "write", path=".agent/workflows/x.json", content="{}\n"), _done()]
    agent, _, approver = _agent(ctx, script, [DENY], mode=Mode.BUILD, rules=rules)

    await _run(agent)

    _asks_without_always(approver)
    assert not (workspace / ".agent" / "workflows" / "x.json").exists()


@pytest.mark.parametrize("path", ["dat/agent/mcp-trust.json", "cfg/agent/workflows/x.json"])
async def test_the_launch_guards_config_data_and_sessions_folders_ask(workspace: Path, path: str) -> None:
    env = {"XDG_CONFIG_HOME": str(workspace / "cfg"), "XDG_DATA_HOME": str(workspace / "dat")}
    guard = prepare_guardrails(workspace, None, None, env, home=workspace, sessions=workspace / "sess")
    ctx = ToolContext(workspace, 5.0, guard=guard)
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=path, content="x\n"), _done()], [DENY])

    await _run(agent)

    assert len(approver.requests) == 1
    assert not (workspace / path).exists()


# Finding 4: the decision uses the path the tool resolved and checked, not the argument resolved again.
class _TargetWrite(Tool):
    """A `write` whose prepared target is fixed, whatever the `path` argument says."""

    name = "write"
    description = "x"
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {"path": {"type": "string"}}}
    mutating = True
    needs_approval = True

    def __init__(self, target: Path | None) -> None:
        self.target = target
        self.ran = False

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        return Prepared(None, self._run, target=self.target)

    async def _run(self) -> ToolResult:
        self.ran = True
        return ToolResult("ok", False, "ok")


def _target_agent(
    ctx: ToolContext, tool: Tool, path: str, *, mode: Mode = Mode.EDITS, rules: Rules | None = None
) -> tuple[Agent, RecordingApprover]:
    approver = RecordingApprover([DENY])
    agent = Agent(
        ScriptedProvider([_call("c1", "write", path=path), _done()], context_window=100_000),
        [tool],
        approver,
        ctx,
        25,
        today=lambda: date(2026, 11, 14),
        recorder=MemoryRecorder(),
        rules=rules,
    )
    agent.set_mode(mode)
    return agent, approver


async def test_accept_edits_decides_from_the_prepared_target(ctx: ToolContext, workspace: Path) -> None:
    tool = _TargetWrite(workspace / ".git" / "config")
    agent, approver = _target_agent(ctx, tool, "ok.txt")

    await _run(agent)

    assert len(approver.requests) == 1 and not tool.ran


async def test_a_write_without_a_prepared_target_asks(ctx: ToolContext) -> None:
    tool = _TargetWrite(None)
    agent, approver = _target_agent(ctx, tool, "ok.txt")

    await _run(agent)

    assert len(approver.requests) == 1 and not tool.ran


# Review: the macOS firmlink spelling `/System/Volumes/Data/...` of the workspace or the home folder is one place.
FIRMLINK = "/System/Volumes/Data"


# Review: a hard link in the workspace is the same file as its other names, which may be in an asking folder.
def _data_guard_ctx(workspace: Path, tmp_path: Path) -> tuple[ToolContext, Path]:
    env = {"XDG_CONFIG_HOME": str(tmp_path / "cfg"), "XDG_DATA_HOME": str(tmp_path / "dat")}
    guard = prepare_guardrails(workspace, None, None, env, home=tmp_path / "home")
    trust = tmp_path / "dat" / "agent" / "permission-trust.json"
    trust.parent.mkdir(parents=True)
    trust.write_text("{}\n")
    return ToolContext(workspace, 5.0, guard=guard), trust
