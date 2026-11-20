"""Accept-edits mode inside the loop: write and edit skip the dialog, except config files and `.git`."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import pytest
from src.engine.approval import Choice, Decision
from src.engine.events import AgentEvent, ToolFinished
from src.engine.guardrails.setup import prepare as prepare_guardrails
from src.engine.hooks.config import DEFAULT_TIMEOUT, HookEntry
from src.engine.hooks.setup import Hooks
from src.engine.loop import Agent
from src.engine.mcp.trust import TrustStore
from src.engine.mode import Mode
from src.engine.permissions.setup import Rules, prepare
from src.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.terminal.manager import TerminalManager
from src.engine.tools import default_tools
from src.engine.tools.base import Prepared, Tool, ToolContext, ToolResult
from src.engine.web import FetchResponse

from tests.fakes import (
    FakeFetcher,
    FakeRunner,
    FakeSearcher,
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


@pytest.mark.parametrize("make", [_make_git_hooks, _make_upper_git, _make_git_file, _make_git_link])
async def test_a_path_in_git_still_asks_for_write_and_edit(ctx: ToolContext, workspace: Path, make: Callable[[Path], str]) -> None:
    path = make(workspace)
    target = workspace / path
    if target.is_file():
        before = target.read_text()
        call = _call("c1", "edit", path=path, old_string=before.split()[0], new_string="zzz")
    else:
        before = None
        call = _call("c1", "write", path=path, content="hacked\n")
    agent, _, approver = _agent(ctx, [call, _done()], [DENY])

    await _run(agent)

    assert len(approver.requests) == 1
    assert (target.read_text() if target.is_file() else None) == before


@pytest.mark.parametrize("path", [".gitignore", ".github/workflows/ci.yml", "src/git/x.py"])
async def test_names_that_only_start_like_git_do_not_ask(ctx: ToolContext, workspace: Path, path: str) -> None:
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=path, content="x\n"), _done()])

    await _run(agent)

    assert approver.requests == []
    assert (workspace / path).read_text() == "x\n"


@pytest.mark.parametrize("path", [".agent/settings.json", ".agent/mcp.json"])
async def test_the_agent_config_files_ask_without_always(ctx: ToolContext, workspace: Path, path: str) -> None:
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=path, content="{}\n"), _done()], [DENY])

    await _run(agent)

    assert len(approver.requests) == 1
    request = approver.requests[0]
    assert request.allow_always is False and request.save_rule is None
    assert not (workspace / path).exists()


async def test_other_approved_tools_still_ask(ctx: ToolContext, workspace: Path, manager: TerminalManager) -> None:
    page = FetchResponse("https://example.com/", 200, "OK", "text/plain", None, b"hi", 2)

    class Extra(Tool):
        name = "mcp__s__t"
        description = "x"
        parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}
        needs_approval = True

        async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
            return Prepared(None, self._run)

        async def _run(self) -> ToolResult:
            return ToolResult("ok", False, "ok")

    calls = [
        ("bash", {"command": "echo hi"}),
        ("web_fetch", {"url": "https://example.com/"}),
        ("term_start", {"name": "t", "command": "sleep 5"}),
        ("term_stop", {"name": "api"}),
        ("mcp__s__t", {}),
    ]
    manager.start("api", "sleep 30")
    for name, args in calls:
        agent, _, approver = _agent(
            ctx, [_call("c1", name, **args), _done()], [DENY], fetcher=FakeFetcher(page), terminals=manager, extra=[Extra()]
        )
        await _run(agent)
        assert [r.tool for r in approver.requests] == [name], name


# A real call for every built-in tool that needs approval, apart from write and edit. `term_stop` names a terminal
# the test starts first: stopping one that does not exist is refused before the approval step.
APPROVAL_CALLS: dict[str, dict[str, object]] = {
    "bash": {"command": "echo hi"},
    "term_start": {"name": "t", "command": "sleep 5"},
    "term_stop": {"name": "api"},
    "web_fetch": {"url": "https://example.com/"},
}


async def test_every_other_built_in_tool_that_needs_approval_asks(
    ctx: ToolContext, workspace: Path, manager: TerminalManager
) -> None:
    page = FetchResponse("https://example.com/", 200, "OK", "text/plain", None, b"hi", 2)
    every = default_tools(manager, fetcher=FakeFetcher(page), searcher=FakeSearcher())
    needing = {t.name for t in every if t.needs_approval} - {"write", "edit"}
    assert needing == set(APPROVAL_CALLS)  # a new tool that needs approval must be added above
    manager.start("api", "sleep 30")
    for name, args in APPROVAL_CALLS.items():
        agent, _, approver = _agent(
            ctx, [_call("c1", name, **args), _done()], [DENY], fetcher=FakeFetcher(page), terminals=manager
        )
        await _run(agent)
        assert [r.tool for r in approver.requests] == [name], name
        assert _tool_message(agent, "c1").startswith("User denied"), name
    assert not (workspace / "t").exists() and manager.get("t") is None and manager.get("api") is not None


async def test_a_deny_rule_still_wins(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    settings = tmp_path / "config" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"permissions": {"deny": ["write"]}}))
    rules = prepare(workspace, settings, TrustStore(tmp_path / "data" / "trust.json")).rules
    agent, _, approver = _agent(ctx, [_call("c1", "write", path="a.txt", content="x"), _done()], rules=rules)

    await _run(agent)

    assert _tool_message(agent, "c1").startswith("denied by rule write (global)")
    assert approver.requests == [] and not (workspace / "a.txt").exists()


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


@pytest.mark.parametrize("path", ["../outside.txt", ".env"])
async def test_the_tools_own_refusals_stand(ctx: ToolContext, workspace: Path, path: str) -> None:
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=path, content="x\n"), _done()])

    await _run(agent)

    assert approver.requests == []
    assert not (workspace / path).exists()
    assert _tool_message(agent, "c1")


async def test_a_mode_change_during_a_step_applies_from_the_next_request(ctx: ToolContext, workspace: Path) -> None:
    gate = Gate()
    script: Script = [
        [gate, *_call("c1", "write", path="a.txt", content="1\n")],
        _call("c2", "write", path="b.txt", content="2\n"),
        _done(),
    ]
    agent, _, approver = _agent(ctx, script, [DENY], mode=Mode.BUILD)
    import asyncio

    task = asyncio.ensure_future(_run(agent))
    await gate.reached.wait()
    agent.set_mode(Mode.EDITS)
    gate.release.set()
    await task

    assert [r.label for r in approver.requests] == ["write a.txt"]
    assert not (workspace / "a.txt").exists()
    assert (workspace / "b.txt").read_text() == "2\n"


async def test_reset_returns_to_build_so_the_next_write_asks(ctx: ToolContext, workspace: Path) -> None:
    agent, _, approver = _agent(ctx, [_call("c1", "write", path="a.txt", content="x"), _done()], [DENY])
    agent.reset(recorder=MemoryRecorder())

    assert agent.mode is Mode.BUILD
    await _run(agent)
    assert len(approver.requests) == 1


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


@pytest.mark.parametrize("path", MIXED_CASE_CONFIG)
async def test_a_mixed_case_agent_config_name_asks_in_edits_mode(ctx: ToolContext, workspace: Path, path: str) -> None:
    (workspace / ".agent").mkdir()
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=path, content="{}\n"), _done()], [DENY])

    await _run(agent)

    _asks_without_always(approver)
    assert list((workspace / ".agent").iterdir()) == []


@pytest.mark.parametrize("path", MIXED_CASE_CONFIG)
async def test_a_mixed_case_agent_config_name_asks_in_build_mode_despite_an_allow_rule(
    ctx: ToolContext, workspace: Path, tmp_path: Path, path: str
) -> None:
    (workspace / ".agent").mkdir()
    rules = _allow(workspace, tmp_path, "write")
    script = [_call("c1", "write", path=path, content="{}\n"), _done()]
    agent, _, approver = _agent(ctx, script, [DENY], mode=Mode.BUILD, rules=rules)

    await _run(agent)

    _asks_without_always(approver)
    assert list((workspace / ".agent").iterdir()) == []


# Finding 2: a workspace that is (or contains) the home folder holds the agent's own folders.
@pytest.mark.parametrize(
    "path",
    [
        ".config/agent/settings.json",
        ".config/agent/mcp.json",
        ".config/agent/commands/x.md",
        ".config/agent/skills/s/SKILL.md",
        ".local/share/agent/permission-trust.json",
        ".local/share/agent/mcp-trust.json",
        ".local/share/agent/checkpoints/x/HEAD",
        ".Config/Agent/settings.json",
    ],
)
async def test_the_agents_own_folders_ask_when_the_workspace_is_the_home_folder(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    monkeypatch.setenv("HOME", str(workspace))
    monkeypatch.delenv("XDG_CONFIG_HOME")
    monkeypatch.delenv("XDG_DATA_HOME")
    ctx = ToolContext(workspace, 5.0)  # no guard: the built-in defaults, from this environment
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=path, content="x\n"), _done()], [DENY])

    await _run(agent)

    assert len(approver.requests) == 1
    assert not (workspace / path).exists()


@pytest.mark.parametrize("path", ["cfg/agent/settings.json", "dat/agent/mcp-trust.json", "sess/abc/1.jsonl", "SESS/x"])
async def test_the_launch_guards_config_data_and_sessions_folders_ask(workspace: Path, path: str) -> None:
    env = {"XDG_CONFIG_HOME": str(workspace / "cfg"), "XDG_DATA_HOME": str(workspace / "dat")}
    guard = prepare_guardrails(workspace, None, None, env, "Linux", home=workspace, sessions=workspace / "sess")
    ctx = ToolContext(workspace, 5.0, guard=guard)
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=path, content="x\n"), _done()], [DENY])

    await _run(agent)

    assert len(approver.requests) == 1
    assert not (workspace / path).exists()


async def test_a_file_beside_the_agents_folders_does_not_ask(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(workspace))
    monkeypatch.delenv("XDG_CONFIG_HOME")
    monkeypatch.delenv("XDG_DATA_HOME")
    ctx = ToolContext(workspace, 5.0)
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=".config/agentx/a.txt", content="x\n"), _done()])

    await _run(agent)

    assert approver.requests == []
    assert (workspace / ".config" / "agentx" / "a.txt").read_text() == "x\n"


# Finding 3: commands and skills are loaded at launch without a trust prompt.
@pytest.mark.parametrize(
    "path", [".agent/commands/x.md", ".agent/skills/s/SKILL.md", ".AGENT/Commands/x.md", ".agent/SKILLS/s/SKILL.md"]
)
async def test_project_commands_and_skills_ask(ctx: ToolContext, workspace: Path, path: str) -> None:
    (workspace / ".agent").mkdir()
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=path, content="x\n"), _done()], [DENY])

    await _run(agent)

    assert len(approver.requests) == 1
    assert list((workspace / ".agent").iterdir()) == []


async def test_other_files_in_the_agent_folder_do_not_ask(ctx: ToolContext, workspace: Path) -> None:
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=".agent/notes.md", content="x\n"), _done()])

    await _run(agent)

    assert approver.requests == []
    assert (workspace / ".agent" / "notes.md").read_text() == "x\n"


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


async def test_a_prepared_target_in_the_workspace_is_accepted_whatever_the_argument(ctx: ToolContext, workspace: Path) -> None:
    tool = _TargetWrite(workspace / "ok.txt")
    agent, approver = _target_agent(ctx, tool, ".git/config")

    await _run(agent)

    assert approver.requests == [] and tool.ran


async def test_a_write_without_a_prepared_target_asks(ctx: ToolContext) -> None:
    tool = _TargetWrite(None)
    agent, approver = _target_agent(ctx, tool, "ok.txt")

    await _run(agent)

    assert len(approver.requests) == 1 and not tool.ran


async def test_the_config_guard_uses_the_prepared_target_too(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    tool = _TargetWrite(workspace / ".agent" / "settings.json")
    agent, approver = _target_agent(ctx, tool, "ok.txt", mode=Mode.BUILD, rules=_allow(workspace, tmp_path, "write"))

    await _run(agent)

    _asks_without_always(approver)
    assert not tool.ran


async def test_write_and_edit_prepare_a_resolved_target(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "src").mkdir()
    (workspace / "src" / "a.txt").write_text("one\n")
    tools = {t.name: t for t in default_tools()}

    written = await tools["write"].prepare({"path": "src/../src/b.txt", "content": "x"}, ctx)
    edited = await tools["edit"].prepare({"path": "src/a.txt", "old_string": "one", "new_string": "two"}, ctx)

    assert isinstance(written, Prepared) and written.target == workspace / "src" / "b.txt"
    assert isinstance(edited, Prepared) and edited.target == workspace / "src" / "a.txt"


# Finding 5: other paths and approvals in accept-edits mode.
async def test_an_absolute_path_inside_the_workspace_does_not_ask(ctx: ToolContext, workspace: Path) -> None:
    path = str(workspace / "a.txt")
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=path, content="x\n"), _done()])

    await _run(agent)

    assert approver.requests == []
    assert (workspace / "a.txt").read_text() == "x\n"


async def test_a_home_relative_path_inside_the_workspace_does_not_ask(
    ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(workspace.parent))
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=f"~/{workspace.name}/a.txt", content="x\n"), _done()])

    await _run(agent)

    assert approver.requests == []
    assert (workspace / "a.txt").read_text() == "x\n"


@pytest.mark.parametrize("path", ["src/../.git/config", "vendor/lib/.git/config"])
async def test_a_git_segment_after_dots_or_nested_asks(ctx: ToolContext, workspace: Path, path: str) -> None:
    (workspace / "src").mkdir()
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=path, content="x\n"), _done()], [DENY])

    await _run(agent)

    assert len(approver.requests) == 1
    assert not (workspace / ".git").exists() and not (workspace / "vendor").exists()


async def test_a_symlink_to_the_project_settings_asks_without_always(ctx: ToolContext, workspace: Path) -> None:
    (workspace / ".agent").mkdir()
    (workspace / ".agent" / "settings.json").write_text("{}\n")
    os.symlink(workspace / ".agent" / "settings.json", workspace / "link.json")
    agent, _, approver = _agent(ctx, [_call("c1", "write", path="link.json", content="{\"x\": 1}\n"), _done()], [DENY])

    await _run(agent)

    _asks_without_always(approver)
    assert (workspace / ".agent" / "settings.json").read_text() == "{}\n"


async def test_an_allow_rule_still_applies_in_edits_mode(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    rules = _allow(workspace, tmp_path, "bash(echo hi)")
    agent, _, approver = _agent(ctx, [_call("c1", "bash", command="echo hi"), _done()], rules=rules)

    await _run(agent)

    assert approver.requests == []
    assert "hi" in _tool_message(agent, "c1")


async def test_a_grant_from_build_mode_still_applies_in_edits_mode(ctx: ToolContext) -> None:
    page = FetchResponse("https://example.com/", 200, "OK", "text/plain", None, b"hi", 2)
    fetcher = FakeFetcher(page)
    agent, _, approver = _agent(ctx, [_call("c1", "web_fetch", url="https://example.com/"), _done()], fetcher=fetcher)
    agent.grants.add("web_fetch", {"url": "https://example.com/"}, "url")

    await _run(agent)

    assert approver.requests == []
    assert fetcher.calls == ["https://example.com/"]


async def test_the_pretooluse_payload_says_edits(ctx: ToolContext, workspace: Path) -> None:
    runner = FakeRunner()
    hooks = Hooks([HookEntry("PreToolUse", "x", None, DEFAULT_TIMEOUT, "global")], [], runner, lambda: True, workspace)
    agent, _, _ = _agent(ctx, [_call("c1", "write", path="a.txt", content="x"), _done()], hooks=hooks)

    await _run(agent)

    [(_, payload, _)] = runner.calls
    assert payload["mode"] == "edits"


# Review: the macOS firmlink spelling `/System/Volumes/Data/...` of the workspace or the home folder is one place.
FIRMLINK = "/System/Volumes/Data"


@pytest.mark.skipif(sys.platform != "darwin", reason="firmlinks are a macOS file system feature")
@pytest.mark.parametrize("spelled", ["workspace", "home"])
@pytest.mark.parametrize("path", [".config/agent/settings.json", ".local/share/agent/permission-trust.json"])
async def test_the_firmlink_spelling_does_not_hide_the_agents_own_folders(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, spelled: str, path: str
) -> None:
    firm = Path(FIRMLINK + str(workspace))
    assert firm.exists() and firm.resolve() == firm  # resolve() keeps the prefix: the comparison must drop it
    monkeypatch.setenv("HOME", str(firm if spelled == "home" else workspace))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    ctx = ToolContext(workspace if spelled == "home" else firm, 5.0)
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=path, content="x\n"), _done()], [DENY])

    await _run(agent)

    assert len(approver.requests) == 1
    assert not (workspace / path).exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="firmlinks are a macOS file system feature")
@pytest.mark.parametrize("path", [".git/config", ".agent/commands/x.md"])
async def test_the_firmlink_spelling_of_the_workspace_still_asks_for_git_and_agent_files(
    workspace: Path, path: str
) -> None:
    (workspace / ".git").mkdir()
    (workspace / ".agent" / "commands").mkdir(parents=True)
    ctx = ToolContext(Path(FIRMLINK + str(workspace)), 5.0)
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=path, content="x\n"), _done()], [DENY])

    await _run(agent)

    assert len(approver.requests) == 1
    assert not (workspace / path).exists()


# Review: a hard link in the workspace is the same file as its other names, which may be in an asking folder.
def _data_guard_ctx(workspace: Path, tmp_path: Path) -> tuple[ToolContext, Path]:
    env = {"XDG_CONFIG_HOME": str(tmp_path / "cfg"), "XDG_DATA_HOME": str(tmp_path / "dat")}
    guard = prepare_guardrails(workspace, None, None, env, "Linux", home=tmp_path / "home")
    trust = tmp_path / "dat" / "agent" / "permission-trust.json"
    trust.parent.mkdir(parents=True)
    trust.write_text("{}\n")
    return ToolContext(workspace, 5.0, guard=guard), trust


@pytest.mark.parametrize("tool", ["write", "edit"])
async def test_a_hard_link_to_a_file_in_the_agents_data_folder_asks(workspace: Path, tmp_path: Path, tool: str) -> None:
    ctx, trust = _data_guard_ctx(workspace, tmp_path)
    os.link(trust, workspace / "notes.json")
    if tool == "write":
        call = _call("c1", "write", path="notes.json", content="hacked\n")
    else:
        call = _call("c1", "edit", path="notes.json", old_string="{}", new_string="hacked")
    agent, _, approver = _agent(ctx, [call, _done()], [DENY])

    await _run(agent)

    assert len(approver.requests) == 1
    assert trust.read_text() == "{}\n"


async def test_an_ordinary_existing_file_still_does_not_ask(workspace: Path, tmp_path: Path) -> None:
    ctx, _ = _data_guard_ctx(workspace, tmp_path)
    (workspace / "notes.json").write_text("{}\n")
    script = [_call("c1", "edit", path="notes.json", old_string="{}", new_string="[]"), _done()]
    agent, _, approver = _agent(ctx, script)

    await _run(agent)

    assert approver.requests == []
    assert (workspace / "notes.json").read_text() == "[]\n"
