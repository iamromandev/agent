"""Protected paths inside the loop: a targeted read, glob or grep always opens a dialog (R4, R7)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import pytest
from core.engine.approval import Choice, Decision
from core.engine.events import AgentEvent, ToolFinished
from core.engine.guardrails.redact import Redactor
from core.engine.guardrails.setup import Guardrails
from core.engine.guardrails.setup import prepare as prepare_guard
from core.engine.loop import REDACT_MARGIN, Agent
from core.engine.mcp.contract import CallResult, ToolInfo
from core.engine.mcp.tool import McpTool
from core.engine.mcp.trust import TrustStore
from core.engine.permissions.setup import Rules, prepare
from core.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from core.engine.tools import default_tools, read_only_tools
from core.engine.tools.base import Prepared, Tool, ToolContext, ToolResult
from core.engine.tools.text import TOOL_OUTPUT_LIMIT
from core.testing.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider

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
    return ToolContext(workspace, 5.0, guard=prepare_guard(workspace, None, None, {}, home=home))


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


# --- Redaction (R14, R15, R16, NFR4, NFR5) -----------------------------------------------------------------------

GITHUB = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
ENV_SECRET = "hard-coded-deploy-secret-42"
REDACTED_GITHUB = "[redacted:github-token]"


class _Leak(Tool):
    """A tool (standing in for an MCP tool) that returns a fixed text, and a display copy with colors."""

    name: ClassVar[str] = "mcp__srv__leak"
    description: ClassVar[str] = "returns a secret"
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}, "required": []}
    text = f"token {GITHUB}"

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        async def run() -> ToolResult:
            return ToolResult(self.text, False, "ok", display=f"\x1b[31m{self.text}\x1b[0m")

        return Prepared(None, run)


@pytest.fixture(autouse=True)
def _posix_sh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", "/bin/sh")


def _redacting_agent(
    ctx: ToolContext,
    script: Script,
    *,
    recorder: Any = None,
    task: bool = False,
    decisions: list[Decision] | None = None,
) -> Agent:
    provider = ScriptedProvider(script, context_window=100_000)
    return Agent(
        provider,
        default_tools(),
        RecordingApprover(decisions or [Decision(Choice.ONCE)] * 5),
        ctx,
        25,
        today=lambda: date(2026, 11, 14),
        recorder=recorder if recorder is not None else MemoryRecorder(),
        extra_tools=lambda: [_Leak()],
        task_tools=read_only_tools() if task else None,
    )


def _guard(workspace: Path, home: Path, env: dict[str, str] | None = None) -> Guardrails:
    return prepare_guard(workspace, None, None, env or {}, home=home)


async def test_nfr4_a_redactor_that_raises_withholds_the_output(
    gctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: Redactor, text: str) -> tuple[str, int]:
        raise RuntimeError("boom with " + text)

    monkeypatch.setattr(Redactor, "redact", broken)
    agent = _redacting_agent(gctx, [_call("c1", "mcp__srv__leak"), _done()])

    events = await _run(agent)

    assert _tool_message(agent, "c1") == "[redaction failed: output withheld]"
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.result.output == "[redaction failed: output withheld]"
    assert finished.result.is_error is True
    assert finished.result.summary == "redaction failed"
    assert finished.result.display is None
    assert GITHUB not in repr(events)
    assert gctx.guard is not None
    assert gctx.guard.redactions == 0
    assert any(isinstance(e, ToolFinished) for e in events)  # the turn went on
    assert _tool_message(agent, "c1") in "".join(m.content for m in agent.history)


# --- Review fixes: the bounded input, orphan blocks, previews, MCP and task paths ----------------------------------

PEM_LINES = ["-----BEGIN PRIVATE KEY-----", "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7", "Zm9vYmFyYmF6", "-----END PRIVATE KEY-----"]


class _McpCaller:
    def __init__(self, outcome: CallResult | Exception) -> None:
        self.outcome = outcome

    async def call(self, server: str, tool: str, args: dict[str, Any]) -> CallResult:
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _mcp_agent(ctx: ToolContext, outcome: CallResult | Exception) -> Agent:
    tool = McpTool(_McpCaller(outcome), "srv", ToolInfo("leak", "leaks", {"type": "object", "properties": {}}), "mcp__srv__leak")
    return Agent(
        ScriptedProvider([_call("c1", "mcp__srv__leak"), _done()], context_window=100_000),
        default_tools(),
        RecordingApprover([Decision(Choice.ONCE)]),
        ctx,
        25,
        today=lambda: date(2026, 11, 14),
        recorder=MemoryRecorder(),
        extra_tools=lambda: [tool],
    )


# --- second review: the window edge and the truncation note ------------------------------------------------------


def _jwt_line(length: int) -> str:
    """A JWT of `length` characters followed by a newline."""
    body = length - 3 - 2  # `eyJ`, two dots
    first = body // 3
    return "eyJ" + "a" * first + "." + "b" * first + "." + "c" * (body - 2 * first) + "\n"


async def test_the_note_adds_what_the_limit_cut_to_what_the_window_dropped(
    gctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    room = TOOL_OUTPUT_LIMIT + REDACT_MARGIN
    word = "word "
    assert room % len(word) == 0
    monkeypatch.setattr(_Leak, "text", word * 40_000)  # nothing to redact: the window cut and the limit cut both count
    agent = _redacting_agent(gctx, [_call("c1", "mcp__srv__leak"), _done()])

    await _run(agent)

    message = _tool_message(agent, "c1")
    assert message.endswith(f"\n[{40_000 * len(word) - TOOL_OUTPUT_LIMIT} characters truncated]")
