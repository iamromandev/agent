"""PreToolUse and PostToolUse inside the loop: order against the dialogs, redaction, limits, resume, no cost without hooks."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import date
from pathlib import Path

import pytest
from src.engine.approval import Choice, Decision
from src.engine.events import AgentEvent, Notice, ToolFinished, TurnEnded
from src.engine.guardrails.setup import prepare as prepare_guard
from src.engine.hooks.config import DEFAULT_TIMEOUT, Event, HookEntry
from src.engine.hooks.runner import HookRun
from src.engine.hooks.setup import Hooks
from src.engine.loop import INTERRUPTED, Agent
from src.engine.mcp.trust import TrustStore
from src.engine.mode import Mode
from src.engine.permissions.rules import Source
from src.engine.permissions.setup import prepare as prepare_rules
from src.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.session import FileRecorder, Recorder, SessionStore, workspace_dir
from src.engine.tools import default_tools, read_only_tools
from src.engine.tools.base import ToolContext
from src.engine.tools.text import TOOL_OUTPUT_LIMIT

from tests.fakes import FakeRunner, Gate, MemoryRecorder, RecordingApprover, ScriptedProvider, exit_run, ok_run

Script = list[list[ProviderEvent | Gate]]
GITHUB = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"


def call(call_id: str, name: str, **args: object) -> list[ProviderEvent | Gate]:
    return [ToolCallEvent(ToolCall(call_id, name, dict(args))), Done(10, 5)]


def done() -> list[ProviderEvent | Gate]:
    return [TextDelta("done"), Done(1, 1)]


@pytest.fixture(autouse=True)
def _posix_sh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", "/bin/sh")


@pytest.fixture
def home(tmp_path: Path) -> Path:
    folder = tmp_path / "home"
    (folder / ".ssh").mkdir(parents=True)
    (folder / ".ssh" / "config").write_text("Host secret-host\n")
    return folder.resolve()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("hello\n")
    return root.resolve()


@pytest.fixture
def ctx(workspace: Path, home: Path) -> ToolContext:
    return ToolContext(workspace, 5.0, guard=prepare_guard(workspace, None, None, {}, "Linux", home=home))


def entry(event: Event, command: str = "x", matcher: str | None = None, source: Source = "global") -> HookEntry:
    return HookEntry(event, command, matcher, DEFAULT_TIMEOUT, source)


def make(
    ctx: ToolContext,
    script: Script,
    runner: FakeRunner,
    *entries: HookEntry,
    decisions: list[Decision] | None = None,
    recorder: Recorder | None = None,
    task: bool = False,
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
        recorder=recorder or MemoryRecorder(),
        hooks=Hooks(entries, [], runner, lambda: True, ctx.workspace),
        task_tools=read_only_tools() if task else None,
    )
    return agent, provider, approver


async def run(agent: Agent, text: str = "go") -> list[AgentEvent]:
    events: list[AgentEvent] = []
    await agent.run_turn(text, events.append)
    return events


def tool_message(agent: Agent, call_id: str) -> str:
    return next(m.content for m in agent.history if m.role == "tool" and m.tool_call_id == call_id)


async def test_a_blocking_pretooluse_hook_refuses_the_call_before_anything_runs_or_asks(
    ctx: ToolContext, workspace: Path
) -> None:
    runner = FakeRunner([exit_run(2, "no shell here")])
    agent, _, approver = make(ctx, [call("c1", "bash", command="touch ran"), done()], runner, entry("PreToolUse", matcher="bash"))

    events = await run(agent)

    assert tool_message(agent, "c1") == "blocked by hook: no shell here"
    assert not (workspace / "ran").exists() and approver.requests == []
    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert finished[0].result.is_error and finished[0].result.summary == "blocked by hook"
    assert Notice("info", "hook PreToolUse (global) blocked: no shell here") in events


async def test_a_token_in_a_hooks_stderr_is_redacted_in_the_transcript_line(ctx: ToolContext) -> None:
    runner = FakeRunner([exit_run(2, f"leaked {GITHUB}")])
    agent, _, _ = make(ctx, [call("c1", "read", path="a.txt"), done()], runner, entry("PreToolUse"))

    events = await run(agent)

    notices = [e.text for e in events if isinstance(e, Notice) and e.text.startswith("hook PreToolUse")]
    assert len(notices) == 1 and GITHUB not in notices[0] and notices[0].startswith("hook PreToolUse (global) blocked: leaked ")
    assert GITHUB not in tool_message(agent, "c1")


async def test_a_token_in_a_posttooluse_warning_is_redacted_in_the_transcript_line(ctx: ToolContext) -> None:
    runner = FakeRunner([exit_run(1, f"leaked {GITHUB}")])
    agent, _, _ = make(ctx, [call("c1", "read", path="a.txt"), done()], runner, entry("PostToolUse"))

    events = await run(agent)

    notices = [e.text for e in events if isinstance(e, Notice) and e.text.startswith("hook PostToolUse")]
    assert len(notices) == 1 and GITHUB not in notices[0] and "leaked" in notices[0]


async def test_the_pretooluse_payload_has_tool_args_session_cwd_and_mode(ctx: ToolContext, workspace: Path) -> None:
    runner = FakeRunner()
    agent, _, _ = make(ctx, [call("c1", "read", path="a.txt"), done()], runner, entry("PreToolUse"))

    await run(agent)

    [(_, payload, env)] = runner.calls
    assert payload == {
        "event": "PreToolUse",
        "tool": "read",
        "args": {"path": "a.txt"},
        "session_id": "",
        "cwd": str(workspace),
        "mode": "build",
    }
    assert env["AGENT_HOOK_EVENT"] == "PreToolUse"


@pytest.mark.parametrize("mode", [Mode.BUILD, Mode.PLAN])
async def test_ac7_a_block_on_a_protected_read_comes_before_the_protected_dialog(
    ctx: ToolContext, home: Path, mode: Mode
) -> None:
    target = str(home / ".ssh" / "config")
    agent, _, approver = make(ctx, [call("c1", "read", path=target), done()], FakeRunner([exit_run(2, "keep out")]), entry("PreToolUse"))
    agent.set_mode(mode)

    await run(agent)

    assert approver.requests == []
    assert tool_message(agent, "c1") == "blocked by hook: keep out"
    assert "secret-host" not in tool_message(agent, "c1")


async def test_ac7_decision_allow_does_not_skip_the_approval_dialog(ctx: ToolContext, workspace: Path) -> None:
    runner = FakeRunner([ok_run(json.dumps({"decision": "allow"}))])
    agent, _, approver = make(
        ctx, [call("c1", "bash", command="touch ran"), done()], runner, entry("PreToolUse"), decisions=[Decision(Choice.DENY)]
    )

    await run(agent)

    assert [r.tool for r in approver.requests] == ["bash"]
    assert tool_message(agent, "c1").startswith("User denied")
    assert not (workspace / "ran").exists()


async def test_a_hook_does_not_run_for_a_call_a_deny_rule_or_plan_mode_already_refused(ctx: ToolContext) -> None:
    runner = FakeRunner()
    agent, _, _ = make(ctx, [call("c1", "write", path="x.txt", content="x"), call("c2", "nonexistent"), done()], runner, entry("PreToolUse"))
    agent.set_mode(Mode.PLAN)

    await run(agent)

    assert "plan mode" in tool_message(agent, "c1") and runner.calls == []


async def test_a_hook_that_warns_does_not_block_and_shows_a_warning(ctx: ToolContext) -> None:
    agent, _, _ = make(ctx, [call("c1", "read", path="a.txt"), done()], FakeRunner([exit_run(1, "lint broke")]), entry("PreToolUse"))

    events = await run(agent)

    assert "hello" in tool_message(agent, "c1")
    assert Notice("warn", "hook PreToolUse (global) warning: exit 1: lint broke") in events


async def test_ac8_posttooluse_context_is_appended_to_the_result(ctx: ToolContext) -> None:
    runner = FakeRunner([ok_run(json.dumps({"context": "lint: E1"}))])
    agent, _, _ = make(ctx, [call("c1", "read", path="a.txt"), done()], runner, entry("PostToolUse", matcher="read"))

    events = await run(agent)

    assert tool_message(agent, "c1") == "     1\thello\n[hook PostToolUse] lint: E1"
    [(_, payload, _)] = runner.calls
    assert payload["event"] == "PostToolUse" and payload["tool"] == "read" and payload["is_error"] is False
    assert "hello" in payload["result"] and "hook PostToolUse" not in payload["result"]
    assert Notice("info", "hook PostToolUse (global) context: lint: E1") in events
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.result.output.endswith("lint: E1") and finished.result.display is None


async def test_ac8_a_token_in_the_context_is_redacted(ctx: ToolContext) -> None:
    runner = FakeRunner([ok_run(json.dumps({"context": f"found {GITHUB}"}))])
    agent, _, _ = make(ctx, [call("c1", "read", path="a.txt"), done()], runner, entry("PostToolUse"))

    await run(agent)

    assert GITHUB not in tool_message(agent, "c1") and "[hook PostToolUse] found " in tool_message(agent, "c1")
    assert ctx.guard is not None and ctx.guard.redactions == 1


async def test_ac8_posttooluse_stdin_carries_the_redacted_result(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "key.txt").write_text(f"token {GITHUB}\n")
    runner = FakeRunner()
    agent, _, _ = make(ctx, [call("c1", "read", path="key.txt"), done()], runner, entry("PostToolUse"))

    await run(agent)

    assert GITHUB not in runner.calls[0][1]["result"]


async def test_ac8_the_result_stays_within_the_output_limit(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "big.txt").write_text("y" * (TOOL_OUTPUT_LIMIT - 100))
    runner = FakeRunner([ok_run(json.dumps({"context": "z" * 5000}))])
    agent, _, _ = make(ctx, [call("c1", "read", path="big.txt"), done()], runner, entry("PostToolUse"))

    await run(agent)

    output = tool_message(agent, "c1")
    assert len(output) <= TOOL_OUTPUT_LIMIT
    assert "yyyy" in output[:30] and "[hook PostToolUse] z" in output


async def test_a_tool_output_already_at_the_limit_keeps_all_of_it(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "big.txt").write_text("y" * (TOOL_OUTPUT_LIMIT * 2))
    with_hook, _, _ = make(
        ctx, [call("c1", "read", path="big.txt"), done()], FakeRunner([ok_run(json.dumps({"context": "note"}))]), entry("PostToolUse")
    )
    without, _, _ = make(ctx, [call("c1", "read", path="big.txt"), done()], FakeRunner(), entry("PostToolUse"))

    await run(with_hook)
    await run(without)

    assert tool_message(with_hook, "c1").startswith(tool_message(without, "c1")[: TOOL_OUTPUT_LIMIT - 50])


async def test_posttooluse_runs_for_an_error_result_and_exit_2_stderr_becomes_context(ctx: ToolContext) -> None:
    runner = FakeRunner([exit_run(2, "E2 on it")])
    agent, _, _ = make(ctx, [call("c1", "read", path="missing.txt"), done()], runner, entry("PostToolUse"))

    await run(agent)

    assert runner.calls[0][1]["is_error"] is True
    assert tool_message(agent, "c1").endswith("\n[hook PostToolUse] E2 on it")


async def test_posttooluse_does_not_run_after_a_hook_blocked_the_call(ctx: ToolContext) -> None:
    runner = FakeRunner(lambda command, payload: exit_run(2, "no") if payload["event"] == "PreToolUse" else ok_run())
    agent, _, _ = make(
        ctx, [call("c1", "read", path="a.txt"), done()], runner, entry("PreToolUse"), entry("PostToolUse")
    )

    await run(agent)

    assert [c[1]["event"] for c in runner.calls] == ["PreToolUse"]


def post_calls(runner: FakeRunner) -> list[str]:
    return [c[1]["tool"] for c in runner.calls if c[1]["event"] == "PostToolUse"]


async def test_posttooluse_does_not_run_after_a_user_denial(ctx: ToolContext) -> None:
    runner = FakeRunner()
    agent, _, _ = make(
        ctx, [call("c1", "bash", command="touch ran"), done()], runner, entry("PostToolUse"), decisions=[Decision(Choice.DENY)]
    )

    await run(agent)

    assert tool_message(agent, "c1").startswith("User denied") and runner.calls == []


async def test_posttooluse_does_not_run_for_a_call_plan_mode_refused(ctx: ToolContext) -> None:
    runner = FakeRunner()
    agent, _, _ = make(ctx, [call("c1", "write", path="x.txt", content="x"), done()], runner, entry("PostToolUse"))
    agent.set_mode(Mode.PLAN)

    await run(agent)

    assert "plan mode" in tool_message(agent, "c1") and runner.calls == []


async def test_posttooluse_does_not_run_for_a_call_a_deny_rule_refused(ctx: ToolContext, tmp_path: Path) -> None:
    settings = tmp_path / "config" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"permissions": {"deny": ["read"]}}))
    runner = FakeRunner()
    agent, _, _ = make(ctx, [call("c1", "read", path="a.txt"), done()], runner, entry("PostToolUse"))
    agent.rules = prepare_rules(ctx.workspace, settings, TrustStore(tmp_path / "data" / "trust.json")).rules

    await run(agent)

    assert tool_message(agent, "c1").startswith("denied by rule") and runner.calls == []


async def test_posttooluse_does_not_run_for_a_protected_read_the_user_refused(ctx: ToolContext, home: Path) -> None:
    runner = FakeRunner()
    agent, _, approver = make(
        ctx,
        [call("c1", "read", path=str(home / ".ssh" / "config")), done()],
        runner,
        entry("PostToolUse"),
        decisions=[Decision(Choice.DENY)],
    )

    await run(agent)

    assert [r.protected for r in approver.requests] == [True]
    assert tool_message(agent, "c1").startswith("User denied") and runner.calls == []


async def test_posttooluse_does_not_run_for_invalid_arguments(ctx: ToolContext) -> None:
    runner = FakeRunner()
    agent, _, _ = make(ctx, [call("c1", "read"), done()], runner, entry("PostToolUse"))

    await run(agent)

    assert "path" in tool_message(agent, "c1") and runner.calls == []


async def test_posttooluse_runs_for_a_call_that_was_approved_and_ran(ctx: ToolContext, workspace: Path) -> None:
    runner = FakeRunner()
    agent, _, _ = make(
        ctx, [call("c1", "write", path="out.txt", content="d"), done()], runner, entry("PostToolUse"), decisions=[Decision(Choice.ONCE)]
    )

    await run(agent)

    assert (workspace / "out.txt").exists() and post_calls(runner) == ["write"]


async def test_posttooluse_is_skipped_when_the_cancel_landed_during_a_write(
    ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, release = threading.Event(), threading.Event()
    real = Path.write_bytes

    def slow(path: Path, data: bytes) -> int:
        if path.name == "out.txt":
            started.set()
            assert release.wait(5.0)
        return real(path, data)

    monkeypatch.setattr(Path, "write_bytes", slow)
    runner = FakeRunner([ok_run(json.dumps({"context": "late"}))])
    agent, _, _ = make(
        ctx, [call("c1", "write", path="out.txt", content="data"), done()], runner, entry("PostToolUse"), decisions=[Decision(Choice.ONCE)]
    )
    events: list[AgentEvent] = []
    task = asyncio.ensure_future(agent.run_turn("go", events.append))
    assert await asyncio.to_thread(started.wait, 5.0)

    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert runner.calls == []
    assert (workspace / "out.txt").read_text() == "data"
    assert "late" not in tool_message(agent, "c1") and next(e for e in events if isinstance(e, ToolFinished)).applied


class RaisingEmit:
    """An event sink that fails for the hook's notices, as a broken UI callback might."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        if isinstance(event, Notice) and event.text.startswith("hook PostToolUse") and "context" in event.text:
            raise RuntimeError("sink broke")
        self.events.append(event)


async def test_a_failure_in_the_posttooluse_step_keeps_the_result_and_warns(ctx: ToolContext) -> None:
    runner = FakeRunner([ok_run(json.dumps({"context": "lint: E1"}))])
    agent, _, _ = make(ctx, [call("c1", "read", path="a.txt"), done()], runner, entry("PostToolUse"))
    sink = RaisingEmit()

    await agent.run_turn("go", sink)

    assert tool_message(agent, "c1") == "     1\thello"
    assert any(isinstance(e, Notice) and e.level == "warn" and "PostToolUse" in e.text and "sink broke" in e.text for e in sink.events)
    assert TurnEnded("done") in sink.events


async def test_a_blocked_call_is_an_ordinary_error_in_the_session_and_the_session_resumes(
    ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    root = tmp_path / "sessions"
    recorder = FileRecorder(workspace_dir(root, workspace), workspace)
    agent, _, _ = make(
        ctx, [call("c1", "read", path="a.txt"), done()], FakeRunner([exit_run(2, "nope")]), entry("PreToolUse"), recorder=recorder
    )

    await run(agent)
    recorder.close()
    loaded = SessionStore(root, workspace).load(recorder.session_id)

    assert recorder.session_id and loaded.skipped_lines == 0
    assert [m.content for m in loaded.history if m.role == "tool"] == ["blocked by hook: nope"]
    assert loaded.tool_meta["c1"].summary == "blocked by hook"
    assert recorder.path is not None
    resumed, provider, _ = make(ctx, [done()], FakeRunner(), recorder=FileRecorder.resume(recorder.path, workspace))
    resumed.reset(recorder=FileRecorder.resume(recorder.path, workspace), history=loaded.history, tool_meta=loaded.tool_meta)
    await run(resumed, "again")
    assert any(m.content == "blocked by hook: nope" for m in provider.requests[0][0])


async def test_the_session_id_in_the_payload_is_the_recorders(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    recorder = FileRecorder(workspace_dir(tmp_path / "sessions", workspace), workspace)
    runner = FakeRunner()
    agent, _, _ = make(ctx, [call("c1", "read", path="a.txt"), done()], runner, entry("PreToolUse"), recorder=recorder)

    await run(agent)
    recorder.close()

    assert recorder.session_id != "" and runner.calls[0][1]["session_id"] == recorder.session_id


async def test_without_hooks_nothing_is_spawned_and_the_overhead_is_tiny(ctx: ToolContext) -> None:
    runner = FakeRunner()
    agent, _, _ = make(ctx, [call("c1", "read", path="a.txt"), done()], runner)  # no entries at all

    await run(agent)

    assert runner.calls == []
    started = time.perf_counter()
    for _ in range(1000):
        agent.hooks.has("PreToolUse", "read")
        agent.hooks.has("PostToolUse", "read")
    assert (time.perf_counter() - started) / 1000 <= 0.001


async def test_a_default_agent_has_no_hooks(ctx: ToolContext) -> None:
    agent = Agent(ScriptedProvider([[TextDelta("x"), Done(1, 1)]]), default_tools(), RecordingApprover(), ctx, 5)

    assert agent.hooks.entries == () and not agent.hooks.has("PreToolUse", "read")


async def test_the_task_childs_own_calls_are_not_hooked_but_the_parents_task_call_is(ctx: ToolContext) -> None:
    runner = FakeRunner()
    child: Script = [call("k1", "read", path="a.txt"), [TextDelta("report"), Done(1, 1)]]
    agent, _, _ = make(
        ctx, [call("c1", "task", description="look", prompt="read a.txt"), *child, done()], runner, entry("PreToolUse"), task=True
    )

    await run(agent)

    assert [c[1]["tool"] for c in runner.calls] == ["task"]


class HangingRunner:
    """A runner whose hooks never finish: the test cancels the turn while one runs."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False

    async def run(self, command: str, stdin: bytes, timeout: float, env_extra: object) -> HookRun:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")


async def test_esc_during_a_pretooluse_hook_kills_it_and_interrupts_the_call(ctx: ToolContext) -> None:
    runner = HangingRunner()
    agent, _, _ = make(ctx, [call("c1", "read", path="a.txt"), done()], FakeRunner(), entry("PreToolUse"))
    agent.hooks = Hooks([entry("PreToolUse")], [], runner, lambda: True, ctx.workspace)
    events: list[AgentEvent] = []
    task = asyncio.ensure_future(agent.run_turn("go", events.append))
    await asyncio.wait_for(runner.started.wait(), 5)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert runner.cancelled
    assert tool_message(agent, "c1") == INTERRUPTED
    assert TurnEnded("interrupted") in events


async def test_esc_during_a_posttooluse_hook_keeps_the_finished_result_and_interrupts(
    ctx: ToolContext, workspace: Path
) -> None:
    runner = HangingRunner()
    agent, _, _ = make(ctx, [call("c1", "write", path="out.txt", content="data"), done()], FakeRunner(), decisions=[Decision(Choice.ONCE)])
    agent.hooks = Hooks([entry("PostToolUse")], [], runner, lambda: True, workspace)
    events: list[AgentEvent] = []
    task = asyncio.ensure_future(agent.run_turn("go", events.append))
    await asyncio.wait_for(runner.started.wait(), 5)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert runner.cancelled
    assert (workspace / "out.txt").read_text() == "data"
    assert tool_message(agent, "c1") != INTERRUPTED and "out.txt" in tool_message(agent, "c1")
    assert next(e for e in events if isinstance(e, ToolFinished)).applied
    assert TurnEnded("interrupted") in events

