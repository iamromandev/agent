"""Protected paths inside the loop: a targeted read, glob or grep always opens a dialog (R4, R7)."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import pytest
from src.engine.approval import Choice, Decision
from src.engine.events import AgentEvent, TaskProgress, ToolFinished
from src.engine.guardrails.redact import Redactor
from src.engine.guardrails.setup import Guardrails
from src.engine.guardrails.setup import prepare as prepare_guard
from src.engine.loop import REDACT_MARGIN, Agent
from src.engine.mcp.contract import CallResult, McpError, ToolInfo
from src.engine.mcp.tool import McpTool
from src.engine.mcp.trust import TrustStore
from src.engine.mode import Mode
from src.engine.permissions.setup import Rules, prepare
from src.engine.provider.base import Done, Message, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.session import FileRecorder, SessionStore, ToolMeta, workspace_dir
from src.engine.tools import default_tools, read_only_tools
from src.engine.tools.base import Prepared, Tool, ToolContext, ToolResult
from src.engine.tools.text import TOOL_OUTPUT_LIMIT

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
    return prepare_guard(workspace, None, None, env or {}, "Linux", home=home)


async def test_ac14_a_bash_output_is_redacted_in_history_event_and_count(gctx: ToolContext) -> None:
    script = [_call("c1", "bash", command=f"echo {GITHUB}"), _done()]
    agent = _redacting_agent(gctx, script)

    events = await _run(agent)

    assert _tool_message(agent, "c1") == f"{REDACTED_GITHUB}\nexit 0"
    assert GITHUB not in "".join(m.content for m in agent.history)
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert GITHUB not in finished.result.output
    assert gctx.guard is not None
    assert gctx.guard.redactions == 1


async def test_ac14_a_read_result_is_redacted(gctx: ToolContext, workspace: Path) -> None:
    (workspace / "notes.txt").write_text(f"token: {GITHUB}\nok\n")
    agent = _redacting_agent(gctx, [_call("c1", "read", path="notes.txt"), _done()])

    await _run(agent)

    assert REDACTED_GITHUB in _tool_message(agent, "c1")
    assert GITHUB not in _tool_message(agent, "c1")


async def test_ac14_an_mcp_style_result_is_redacted_and_its_colour_copy_dropped(gctx: ToolContext) -> None:
    agent = _redacting_agent(gctx, [_call("c1", "mcp__srv__leak"), _done()])

    events = await _run(agent)

    assert _tool_message(agent, "c1") == f"token {REDACTED_GITHUB}"
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.result.output == f"token {REDACTED_GITHUB}"
    assert finished.result.display is None  # the colour copy of a redacted output is dropped (a secret may be split by codes)
    assert finished.result.summary == "ok"
    assert GITHUB not in repr(finished)


async def test_ac14_a_task_report_is_redacted(gctx: ToolContext) -> None:
    script: Script = [
        _call("t1", "task", description="look", prompt="find the token"),
        [TextDelta(f"found {GITHUB}"), Done(1, 1)],
        _done(),
    ]
    agent = _redacting_agent(gctx, script, task=True)

    await _run(agent)

    assert REDACTED_GITHUB in _tool_message(agent, "t1")
    assert GITHUB not in _tool_message(agent, "t1")


async def test_ac14_the_session_file_and_a_resumed_session_hold_the_placeholder(
    gctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    sessions = tmp_path / "sessions"
    recorder = FileRecorder(workspace_dir(sessions, workspace), workspace)
    agent = _redacting_agent(gctx, [_call("c1", "mcp__srv__leak"), _done()], recorder=recorder)

    await _run(agent)
    agent.close()

    (path,) = workspace_dir(sessions, workspace).glob("*.jsonl")
    assert GITHUB not in path.read_text()
    assert REDACTED_GITHUB in path.read_text()
    loaded = SessionStore(sessions, workspace).load(path.stem)
    assert GITHUB not in "".join(m.content for m in loaded.history)
    assert GITHUB not in "".join(i.text for i in loaded.items)


async def test_ac14_ordinary_output_is_not_touched_and_not_counted(gctx: ToolContext, workspace: Path) -> None:
    text = "commit 4b825dc642cb6eb9a060e54bf8d69288fbee4904 id 123e4567-e89b-12d3-a456-426614174000 sk-short\n"
    (workspace / "log.txt").write_text(text)
    agent = _redacting_agent(gctx, [_call("c1", "read", path="log.txt"), _done()])

    await _run(agent)

    assert text.strip() in _tool_message(agent, "c1")
    assert gctx.guard is not None
    assert gctx.guard.redactions == 0


async def test_ac15_a_removed_variables_value_in_a_script_is_redacted(workspace: Path, home: Path) -> None:
    guard = _guard(workspace, home, {"GITHUB_TOKEN": ENV_SECRET, "PATH": "/usr/bin:/bin"})
    ctx = ToolContext(workspace, 5.0, guard=guard)
    (workspace / "deploy.sh").write_text(f"#!/bin/sh\necho using {ENV_SECRET}\n")
    agent = _redacting_agent(ctx, [_call("c1", "bash", command="sh deploy.sh"), _done()])

    events = await _run(agent)

    assert "using [redacted:env:GITHUB_TOKEN]" in _tool_message(agent, "c1")
    assert ENV_SECRET not in "".join(m.content for m in agent.history)
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert ENV_SECRET not in repr(finished.result)
    assert guard.redactions == 1


async def test_the_limit_is_applied_after_redaction(
    gctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The token straddles the limit: cut first, its first characters would survive; redacted first, none do.
    monkeypatch.setattr(_Leak, "text", "x" * (TOOL_OUTPUT_LIMIT - 11) + " " + GITHUB + "\ntail")
    agent = _redacting_agent(gctx, [_call("c1", "mcp__srv__leak"), _done()])

    await _run(agent)

    assert "ghp_" not in _tool_message(agent, "c1")
    assert len(_tool_message(agent, "c1")) < TOOL_OUTPUT_LIMIT + 100


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


async def test_a_context_without_a_guard_still_redacts(workspace: Path) -> None:
    ctx = ToolContext(workspace, 5.0)
    agent = _redacting_agent(ctx, [_call("c1", "mcp__srv__leak"), _done()])

    await _run(agent)

    assert _tool_message(agent, "c1") == f"token {REDACTED_GITHUB}"


async def test_nfr5_a_session_written_before_redaction_resumes_unchanged(workspace: Path, tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    recorder = FileRecorder(workspace_dir(sessions, workspace), workspace)
    recorder.message(Message("user", "show the config"), None)
    recorder.message(Message("assistant", "", [ToolCall("c1", "bash", {"command": "cat cfg"})]), None)
    recorder.message(Message("tool", f"token {GITHUB}", tool_call_id="c1", name="bash"), ToolMeta("bash cat cfg", "exit 0"))
    recorder.message(Message("assistant", "done"), None)
    recorder.close()
    (path,) = workspace_dir(sessions, workspace).glob("*.jsonl")
    before = path.read_text()

    loaded = SessionStore(sessions, workspace).load(path.stem)
    ctx = ToolContext(workspace, 5.0, guard=_guard(workspace, tmp_path))
    agent = _redacting_agent(ctx, [[TextDelta("ok"), Done(1, 1)]])
    agent.reset(recorder=MemoryRecorder(), history=loaded.history, tool_meta=loaded.tool_meta)
    await _run(agent)

    assert path.read_text() == before  # nothing rewrites an old file
    assert [m.role for m in loaded.history] == ["user", "assistant", "tool", "assistant"]
    assert loaded.history[2].content == f"token {GITHUB}"  # old content is kept as it was written
    assert agent.history[-1].content == "ok"


# --- Review fixes: the bounded input, orphan blocks, previews, MCP and task paths ----------------------------------

PEM_LINES = ["-----BEGIN PRIVATE KEY-----", "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7", "Zm9vYmFyYmF6", "-----END PRIVATE KEY-----"]


async def test_redaction_never_sees_more_than_the_limit_plus_a_margin(
    gctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []
    original = Redactor.redact

    def spy(self: Redactor, text: str) -> tuple[str, int]:
        seen.append(len(text))
        return original(self, text)

    monkeypatch.setattr(Redactor, "redact", spy)
    monkeypatch.setattr(_Leak, "text", "x" * 200_000)
    agent = _redacting_agent(gctx, [_call("c1", "mcp__srv__leak"), _done()])

    events = await _run(agent)

    assert len(seen) == 2  # the output and its display copy
    assert seen[0] <= TOOL_OUTPUT_LIMIT + 10_000
    assert seen[1] <= 4 * (TOOL_OUTPUT_LIMIT + 10_000)  # the display copy has room for its colour codes
    output = _tool_message(agent, "c1")
    assert output.startswith("x" * TOOL_OUTPUT_LIMIT)
    assert output.endswith(f"\n[{200_000 - TOOL_OUTPUT_LIMIT} characters truncated]")  # the note counts what was cut
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.result.display is not None and len(finished.result.display) < TOOL_OUTPUT_LIMIT + 200


async def test_a_huge_adversarial_output_is_redacted_in_a_fraction_of_a_second(
    gctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    monkeypatch.setattr(_Leak, "text", "eyJ" * 100_000)
    agent = _redacting_agent(gctx, [_call("c1", "mcp__srv__leak"), _done()])

    start = time.perf_counter()
    await _run(agent)

    assert time.perf_counter() - start < 2.0


async def test_a_secret_that_starts_before_the_limit_and_ends_after_the_window_is_gone(
    gctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_Leak, "text", "x" * (TOOL_OUTPUT_LIMIT - 5) + " password=" + "s" * 50_000)
    agent = _redacting_agent(gctx, [_call("c1", "mcp__srv__leak"), _done()])

    await _run(agent)

    assert "password=sss" not in _tool_message(agent, "c1")


async def test_a_windowed_read_that_starts_inside_a_pem_block_is_redacted(gctx: ToolContext, workspace: Path) -> None:
    (workspace / "key.txt").write_text("\n".join(["intro", *PEM_LINES, "outro"]) + "\n")
    agent = _redacting_agent(gctx, [_call("c1", "read", path="key.txt", offset=4, limit=2), _done()])

    await _run(agent)

    message = _tool_message(agent, "c1")
    assert "Zm9vYmFy" not in message
    assert "[redacted:private-key]" in message


async def test_a_task_childs_tool_results_are_redacted_before_the_progress_event(gctx: ToolContext, workspace: Path) -> None:
    (workspace / "notes.txt").write_text(f"token: {GITHUB}\n")
    script: Script = [
        _call("t1", "task", description="look", prompt="read notes"),
        _call("r1", "read", path="notes.txt"),
        [TextDelta("done reading"), Done(1, 1)],
        _done(),
    ]
    agent = _redacting_agent(gctx, script, task=True)

    events = await _run(agent)

    child = [e.event for e in events if isinstance(e, TaskProgress) and isinstance(e.event, ToolFinished)]
    assert len(child) == 1
    assert REDACTED_GITHUB in child[0].result.output
    assert GITHUB not in repr(events)
    assert GITHUB not in "".join(m.content for m in agent.history)


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


async def test_a_real_mcp_tools_error_text_is_redacted(gctx: ToolContext) -> None:
    agent = _mcp_agent(gctx, McpError(f"server said: bad token {GITHUB}", "error"))

    events = await _run(agent)

    assert _tool_message(agent, "c1") == f"server said: bad token {REDACTED_GITHUB}"
    assert GITHUB not in repr(events)


async def test_a_real_mcp_tools_result_text_is_redacted(gctx: ToolContext) -> None:
    agent = _mcp_agent(gctx, CallResult([f"a {GITHUB}", "password=hunter2"], False))

    await _run(agent)

    assert _tool_message(agent, "c1") == f"a {REDACTED_GITHUB}\npassword=[redacted:password]"


async def test_an_edit_previews_a_redacted_diff_in_the_dialog_and_the_event_but_writes_the_real_file(
    gctx: ToolContext, workspace: Path
) -> None:
    target = workspace / "app.env.txt"
    target.write_text(f"keep\ntoken={GITHUB}\nlast\n")
    script = [_call("c1", "edit", path="app.env.txt", old_string="keep", new_string="kept"), _done()]
    agent = _redacting_agent(gctx, script)

    events = await _run(agent)

    approver = agent.approver
    assert isinstance(approver, RecordingApprover)
    request = approver.requests[0]
    assert request.preview is not None
    assert GITHUB not in request.preview.body
    assert f"token={GITHUB}" in target.read_text()  # only the display changed
    assert target.read_text() == f"kept\ntoken={GITHUB}\nlast\n"
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.preview is not None
    assert GITHUB not in finished.preview.body
    assert "[redacted:github-token]" in finished.preview.body
    assert gctx.guard is not None
    assert gctx.guard.redactions == 0  # a preview is not a result: nothing counted


async def test_a_display_only_secret_drops_the_display_and_is_not_counted(gctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_Leak, "text", "clean output")

    async def prepare(self: _Leak, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        async def run() -> ToolResult:
            return ToolResult("clean", False, "ok", display=f"shown {GITHUB}")

        return Prepared(None, run)

    monkeypatch.setattr(_Leak, "prepare", prepare)
    agent = _redacting_agent(gctx, [_call("c1", "mcp__srv__leak"), _done()])

    events = await _run(agent)

    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.result.display is None
    assert gctx.guard is not None
    assert gctx.guard.redactions == 0


# --- second review: the window edge and the truncation note ------------------------------------------------------


def _jwt_line(length: int) -> str:
    """A JWT of `length` characters followed by a newline."""
    body = length - 3 - 2  # `eyJ`, two dots
    first = body // 3
    return "eyJ" + "a" * first + "." + "b" * first + "." + "c" * (body - 2 * first) + "\n"


async def test_a_token_that_straddles_the_window_edge_leaves_no_prefix_when_redaction_shrinks_the_text(
    gctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    room = TOOL_OUTPUT_LIMIT + REDACT_MARGIN
    lines = _jwt_line(600) * 60
    lead = lines[: room - 20 - 1] + " "  # the token starts 20 characters before the cut
    assert len(lead) == room - 20
    monkeypatch.setattr(_Leak, "text", lead + GITHUB + "\n" + _jwt_line(600) * 60)
    agent = _redacting_agent(gctx, [_call("c1", "mcp__srv__leak"), _done()])

    await _run(agent)

    message = _tool_message(agent, "c1")
    assert "ghp_" not in message
    assert len(message) < TOOL_OUTPUT_LIMIT  # the 60 JWTs shrank to their placeholders


async def test_the_note_counts_what_the_window_dropped_when_redaction_shrank_the_rest(
    gctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    room = TOOL_OUTPUT_LIMIT + REDACT_MARGIN
    line = _jwt_line(679)  # 680 characters with its newline: the window ends exactly after a line
    assert room % len(line) == 0
    text = line * 150
    monkeypatch.setattr(_Leak, "text", text)
    agent = _redacting_agent(gctx, [_call("c1", "mcp__srv__leak"), _done()])

    await _run(agent)

    message = _tool_message(agent, "c1")
    assert message == "[redacted:jwt]\n" * (room // len(line)) + f"\n[{len(text) - room} characters truncated]"


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
