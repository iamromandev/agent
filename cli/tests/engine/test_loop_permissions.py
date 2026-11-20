"""Permission rules inside the loop: deny first, then allow, then this launch's "Always", then the dialog."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.engine.approval import Choice, Decision
from src.engine.events import AgentEvent, Notice
from src.engine.loop import Agent
from src.engine.mcp.trust import TrustStore
from src.engine.permissions.config import project_settings_path
from src.engine.permissions.setup import Rules, prepare
from src.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.terminal.manager import TerminalManager
from src.engine.tools import default_tools, read_only_tools
from src.engine.tools.base import ToolContext
from src.engine.web import FetchResponse

from tests.fakes import FakeFetcher, Gate, MemoryRecorder, RecordingApprover, ScriptedProvider

Script = list[list[ProviderEvent | Gate]]
URL = "https://example.com/docs"


def _call(call_id: str, name: str, **args: object) -> list[ProviderEvent | Gate]:
    return [ToolCallEvent(ToolCall(call_id, name, dict(args))), Done(10, 5)]


def _term_start(command: str) -> list[ProviderEvent | Gate]:
    return [ToolCallEvent(ToolCall("c1", "term_start", {"name": "t", "command": command})), Done(10, 5)]


def _done() -> list[ProviderEvent | Gate]:
    return [TextDelta("done"), Done(1, 1)]


class Setup:
    """A workspace with a global and a project settings file, prepared like a launch."""

    def __init__(self, ctx: ToolContext, tmp_path: Path) -> None:
        self.ctx = ctx
        self.global_path = tmp_path / "config" / "settings.json"
        self.store = TrustStore(tmp_path / "data" / "permission-trust.json")
        self.project_path = project_settings_path(ctx.workspace)

    def write(self, path: Path, allow: list[str] | None = None, deny: list[str] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        permissions = {k: v for k, v in (("allow", allow), ("deny", deny)) if v is not None}
        path.write_text(json.dumps({"permissions": permissions}))

    def rules(self, trust: bool = False) -> Rules:
        rules = prepare(self.ctx.workspace, self.global_path, self.store).rules
        if trust:
            rules.trust_project()
        return rules


def _setup(ctx: ToolContext, tmp_path: Path) -> Setup:
    return Setup(ctx, tmp_path)


def _agent(
    ctx: ToolContext,
    script: Script,
    rules: Rules | None,
    decisions: list[Decision] | None = None,
    *,
    fetcher: FakeFetcher | None = None,
    task: bool = False,
    terminals: TerminalManager | None = None,
) -> tuple[Agent, ScriptedProvider, RecordingApprover]:
    provider = ScriptedProvider(script, context_window=100_000)
    approver = RecordingApprover(decisions)
    agent = Agent(
        provider,
        default_tools(terminals, fetcher=fetcher),
        approver,
        ctx,
        25,
        today=lambda: date(2026, 11, 14),
        recorder=MemoryRecorder(),
        rules=rules,
        task_tools=read_only_tools(fetcher) if task else None,
    )
    return agent, provider, approver


async def _run(agent: Agent, text: str = "go") -> list[AgentEvent]:
    events: list[AgentEvent] = []
    await agent.run_turn(text, events.append)
    return events


def _tool_message(agent: Agent, call_id: str) -> str:
    return next(m.content for m in agent.history if m.role == "tool" and m.tool_call_id == call_id)


def _page() -> FetchResponse:
    return FetchResponse(URL, 200, "OK", "text/plain", None, b"hello", 5)


async def test_a_deny_rule_on_read_also_stops_the_task_subagent(ctx: ToolContext, tmp_path: Path) -> None:
    (ctx.workspace / "a.txt").write_text("secret")
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, deny=["read"])
    script: Script = [
        _call("t1", "task", description="look", prompt="read a.txt"),
        _call("r1", "read", path="a.txt"),  # the child's request
        [TextDelta("it was refused"), Done(1, 1)],  # the child's report
        _done(),
    ]
    agent, provider, _ = _agent(ctx, script, setup.rules(), task=True)

    await _run(agent)

    child_request = provider.requests[2][0]
    refused = next(m for m in child_request if m.role == "tool")
    assert refused.content == "denied by rule read (global)"
    assert "secret" not in "".join(m.content for m in child_request)


# "Always here"


def _notices(events: list[AgentEvent]) -> list[Notice]:
    return [e for e in events if isinstance(e, Notice)]


async def test_always_here_for_a_command_with_a_star_runs_it_this_run_only_and_warns(
    ctx: ToolContext, tmp_path: Path
) -> None:
    """`bash(echo x*)` would allow more than the one command that was approved, so nothing is saved."""
    setup = _setup(ctx, tmp_path)
    script = [_call("c1", "bash", command="echo x*"), _call("c2", "bash", command="echo x*"), _done()]
    agent, _, approver = _agent(ctx, script, setup.rules(), [Decision(Choice.PROJECT)])

    events = await _run(agent)

    assert len(approver.requests) == 1  # granted for this run
    assert not setup.project_path.exists()
    (notice,) = _notices(events)
    assert notice.level == "warn"
    assert "allowed for this run only" in notice.text


async def test_a_rule_saved_but_not_remembered_as_trusted_says_exactly_that(ctx: ToolContext, tmp_path: Path) -> None:
    setup = _setup(ctx, tmp_path)
    setup.store = TrustStore(ctx.workspace / "trust.json")  # refused: inside the workspace
    agent, _, _ = _agent(ctx, [_call("c1", "bash", command="echo one"), _done()], setup.rules(), [Decision(Choice.PROJECT)])

    events = await _run(agent)

    (notice,) = _notices(events)
    assert notice.level == "warn"
    assert notice.text.startswith("saved to .agent/settings.json but could not be remembered as trusted")
    assert notice.text.endswith("the next launch will ask")
    assert "allowed for this run only" not in notice.text
    assert json.loads(setup.project_path.read_text())["permissions"]["allow"] == ["bash(echo one)"]


# what the "Always here" option says it saves


# the settings files themselves
