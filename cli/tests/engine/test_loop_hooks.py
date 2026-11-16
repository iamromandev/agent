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
from src.engine.compaction import turn_starts
from src.engine.events import AgentEvent, Notice, ToolFinished, TurnEnded
from src.engine.guardrails.setup import prepare as prepare_guard
from src.engine.hooks.config import DEFAULT_TIMEOUT, Event, HookEntry
from src.engine.hooks.runner import HookRun
from src.engine.hooks.setup import Hooks, hook_failed
from src.engine.loop import HOOK_CONTEXT_CUT, HOOK_CONTEXT_LIMIT, INTERRUPTED, Agent
from src.engine.mcp.trust import TrustStore
from src.engine.mode import Mode
from src.engine.permissions.rules import Source
from src.engine.permissions.setup import prepare as prepare_rules
from src.engine.provider.base import Done, Message, ProviderError, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
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



# UserPromptSubmit, Stop and SessionStart


def reply(text: str = "ok") -> list[ProviderEvent | Gate]:
    return [TextDelta(text), Done(1, 1)]


def system_texts(messages: list) -> list[str]:
    return [m.content for m in messages[1:] if m.role == "system"]


async def test_ac9_a_blocking_prompt_hook_is_reported_and_the_history_is_left_alone(ctx: ToolContext) -> None:
    runner = FakeRunner([exit_run(2, "no secrets in prompts")])
    agent, provider, _ = make(ctx, [], runner, entry("UserPromptSubmit"))
    events: list[AgentEvent] = []

    decision = await agent.prompt_hooks("my prompt", events.append)

    assert decision.blocked and decision.reason == "no secrets in prompts"
    assert agent.history == [] and provider.requests == []
    assert events == [Notice("info", "hook UserPromptSubmit (global) blocked: no secrets in prompts")]


async def test_the_prompt_hook_gets_the_prompt_session_and_cwd(ctx: ToolContext, workspace: Path) -> None:
    runner = FakeRunner()
    agent, _, _ = make(ctx, [], runner, entry("UserPromptSubmit"))

    await agent.prompt_hooks("my prompt", lambda e: None)

    _, payload, env = runner.calls[0]
    assert payload == {"event": "UserPromptSubmit", "prompt": "my prompt", "session_id": "", "cwd": str(workspace)}
    assert env["AGENT_HOOK_EVENT"] == "UserPromptSubmit"


async def test_without_a_prompt_hook_nothing_is_run(ctx: ToolContext) -> None:
    runner = FakeRunner()
    agent, _, _ = make(ctx, [], runner, entry("Stop"))
    events: list[AgentEvent] = []

    decision = await agent.prompt_hooks("hi", events.append)

    assert not decision.blocked and decision.context == [] and runner.calls == [] and events == []


async def test_ac9_prompt_context_is_a_system_note_in_the_first_request_only(ctx: ToolContext) -> None:
    runner = FakeRunner([ok_run(json.dumps({"context": "branch is dev"}))])
    agent, provider, _ = make(ctx, [call("c1", "read", path="a.txt"), done()], runner, entry("UserPromptSubmit"))
    events: list[AgentEvent] = []
    decision = await agent.prompt_hooks("go", events.append)

    await agent.run_turn("go", events.append, context=decision.context)

    first, second = provider.requests[0][0], provider.requests[1][0]
    assert system_texts(first) == ["[hook context]\nbranch is dev"]
    assert first[-1].content == "go" and first[-2].role == "system"  # just before the latest user message
    assert system_texts(second) == []  # cleared after sending
    assert all(m.role != "system" for m in agent.history)  # never in the history
    assert Notice("info", "hook UserPromptSubmit (global) context: branch is dev") in events


async def test_ac9_a_token_in_prompt_context_is_redacted_before_the_model_sees_it(ctx: ToolContext) -> None:
    agent, provider, _ = make(ctx, [done()], FakeRunner())

    await agent.run_turn("go", lambda e: None, context=[f"token {GITHUB}"])

    note = system_texts(provider.requests[0][0])[0]
    assert GITHUB not in note and note.startswith("[hook context]\ntoken ")


async def test_the_hook_note_and_the_todo_note_share_one_system_message(ctx: ToolContext) -> None:
    agent, provider, _ = make(ctx, [call("t1", "todo", items=[{"content": "step", "status": "pending"}]), done()], FakeRunner())

    await agent.run_turn("go", lambda e: None, context=["ctx one"])
    await agent.run_turn("again", lambda e: None, context=["ctx two"])

    third = provider.requests[2][0]
    notes = system_texts(third)
    assert len(notes) == 1 and "ctx two" in notes[0] and "step" in notes[0]
    assert notes[0].index("ctx two") < notes[0].index("step")


async def test_pending_hook_context_counts_in_the_estimate_and_reset_drops_it(ctx: ToolContext) -> None:
    agent, _, _ = make(ctx, [], FakeRunner([ok_run(json.dumps({"context": "x" * 4000}))]), entry("SessionStart"))
    before = agent.estimate()
    await agent.session_started("startup", lambda e: None)

    assert agent.estimate() >= before + 900

    agent.reset(recorder=MemoryRecorder())

    assert agent.estimate() == before


async def test_a_blank_prompt_keeps_no_context(ctx: ToolContext) -> None:
    agent, provider, _ = make(ctx, [done()], FakeRunner())

    await agent.run_turn("  ", lambda e: None, context=["stale"])
    await agent.run_turn("real", lambda e: None)

    assert system_texts(provider.requests[0][0]) == []


def user_notes(agent: Agent) -> list[str]:
    return [m.content for m in agent.history if m.role == "user" and m.content.startswith("[hook Stop]")]


async def test_ac10_a_stop_hook_that_always_blocks_continues_three_times_then_warns(ctx: ToolContext) -> None:
    runner = FakeRunner([exit_run(2, "run the tests")])
    agent, provider, _ = make(ctx, [reply(f"r{i}") for i in range(4)], runner, entry("Stop"))

    events = await run(agent)

    assert user_notes(agent) == ["[hook Stop] run the tests"] * 3
    assert [c[1]["continuations"] for c in runner.calls] == [0, 1, 2]
    assert len(provider.requests) == 4
    assert Notice("warn", "Stop hook limit reached") in events
    assert events[-1] == TurnEnded("done")
    assert sum(1 for e in events if isinstance(e, Notice) and e.text.startswith("hook Stop (global) blocked")) == 3
    assert [m.role for m in agent.history] == ["user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant"]


async def test_a_stop_hook_that_allows_ends_the_turn_after_one_run(ctx: ToolContext) -> None:
    runner = FakeRunner()
    agent, provider, _ = make(ctx, [reply()], runner, entry("Stop"))

    events = await run(agent)

    assert len(runner.calls) == 1 and len(provider.requests) == 1
    payload = runner.calls[0][1]
    assert payload["event"] == "Stop" and payload["continuations"] == 0 and payload["cwd"] == str(ctx.workspace)
    assert events[-1] == TurnEnded("done") and not any(isinstance(e, Notice) for e in events)


async def test_the_continuation_counter_is_per_user_turn(ctx: ToolContext) -> None:
    runner = FakeRunner([exit_run(2, "again"), exit_run(0), exit_run(0)])
    agent, _, _ = make(ctx, [reply(), reply(), reply()], runner, entry("Stop"))

    await run(agent, "one")
    await run(agent, "two")

    assert [c[1]["continuations"] for c in runner.calls] == [0, 1, 0]


async def test_a_token_in_a_stop_reason_is_redacted_in_the_note_and_the_line(ctx: ToolContext) -> None:
    runner = FakeRunner([exit_run(2, f"leaked {GITHUB}"), exit_run(0)])
    agent, _, _ = make(ctx, [reply(), reply()], runner, entry("Stop"))

    events = await run(agent)

    assert GITHUB not in "".join(user_notes(agent)) and user_notes(agent)[0].startswith("[hook Stop] leaked ")
    assert not any(isinstance(e, Notice) and GITHUB in e.text for e in events)


async def test_stop_continuations_count_toward_the_step_limit_and_the_hook_is_not_run_on_the_last_step(
    ctx: ToolContext,
) -> None:
    runner = FakeRunner([exit_run(2, "more")])
    agent, _, _ = make(ctx, [reply(), reply()], runner, entry("Stop"))
    agent.max_steps = 2

    events = await run(agent)

    assert [c[1]["continuations"] for c in runner.calls] == [0]  # the second reply used the last step
    assert user_notes(agent) == ["[hook Stop] more"] and events[-1] == TurnEnded("done")


async def test_stop_hooks_do_not_run_when_the_turn_ends_at_the_step_limit(ctx: ToolContext) -> None:
    runner = FakeRunner()
    agent, _, _ = make(ctx, [call(f"c{i}", "read", path="a.txt") for i in range(2)], runner, entry("Stop"))
    agent.max_steps = 2

    events = await run(agent)

    assert runner.calls == [] and events[-1] == TurnEnded("step_limit")


async def test_stop_hooks_do_not_run_after_a_provider_error(ctx: ToolContext) -> None:
    runner = FakeRunner()
    agent, _, _ = make(ctx, [[ProviderError("boom")]], runner, entry("Stop"))

    events = await run(agent)

    assert runner.calls == [] and events[-1] == TurnEnded("error")


async def test_ac10_stop_hooks_do_not_run_after_esc(ctx: ToolContext) -> None:
    runner = FakeRunner()
    gate = Gate()
    agent, _, _ = make(ctx, [[gate, *reply()]], runner, entry("Stop"))
    events: list[AgentEvent] = []
    task = asyncio.ensure_future(agent.run_turn("go", events.append))
    await asyncio.wait_for(gate.reached.wait(), 5)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert runner.calls == [] and TurnEnded("interrupted") in events


async def test_esc_during_a_stop_hook_kills_it_and_interrupts_the_turn(ctx: ToolContext) -> None:
    runner = HangingRunner()
    agent, _, _ = make(ctx, [reply()], FakeRunner())
    agent.hooks = Hooks([entry("Stop")], [], runner, lambda: True, ctx.workspace)
    events: list[AgentEvent] = []
    task = asyncio.ensure_future(agent.run_turn("go", events.append))
    await asyncio.wait_for(runner.started.wait(), 5)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert runner.cancelled and TurnEnded("interrupted") in events and not agent.busy


async def test_a_warning_stop_hook_does_not_continue(ctx: ToolContext) -> None:
    runner = FakeRunner([exit_run(1, "oops")])
    agent, provider, _ = make(ctx, [reply()], runner, entry("Stop"))

    events = await run(agent)

    assert len(provider.requests) == 1 and user_notes(agent) == []
    assert Notice("warn", "hook Stop (global) warning: exit 1: oops") in events


async def test_stop_context_is_unused_and_says_so(ctx: ToolContext) -> None:
    runner = FakeRunner([ok_run(json.dumps({"context": "hello"}))])
    agent, provider, _ = make(ctx, [reply()], runner, entry("Stop"))

    events = await run(agent)

    assert len(provider.requests) == 1
    assert any(isinstance(e, Notice) and e.level == "warn" and "Stop context is unused" in e.text for e in events)


async def test_ac10_stop_notes_are_user_messages_in_the_session_and_it_resumes(
    ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    root = tmp_path / "sessions"
    recorder = FileRecorder(workspace_dir(root, workspace), workspace)
    runner = FakeRunner([exit_run(2, "check again"), exit_run(0)])
    agent, _, _ = make(ctx, [reply("one"), reply("two")], runner, entry("Stop"), recorder=recorder)

    await run(agent)
    recorder.close()
    loaded = SessionStore(root, workspace).load(recorder.session_id)

    assert [(m.role, m.content) for m in loaded.history] == [
        ("user", "go"),
        ("assistant", "one"),
        ("user", "[hook Stop] check again"),
        ("assistant", "two"),
    ]
    assert loaded.skipped_lines == 0
    assert recorder.path is not None
    resumed, provider, _ = make(ctx, [reply("three")], FakeRunner(), recorder=FileRecorder.resume(recorder.path, workspace))
    resumed.reset(recorder=FileRecorder.resume(recorder.path, workspace), history=loaded.history, tool_meta=loaded.tool_meta)
    await run(resumed, "next")
    assert any(m.content == "[hook Stop] check again" for m in provider.requests[0][0])


async def test_undo_removes_the_continuations_with_their_turn(ctx: ToolContext) -> None:
    agent, _, _ = make(ctx, [reply(), reply()], FakeRunner([exit_run(2, "more"), exit_run(0)]), entry("Stop"))
    await run(agent)

    prompt = await agent.undo(lambda e: None)

    assert prompt == "go" and agent.history == []


async def test_ac11_session_start_gets_the_source_and_its_context_reaches_the_next_request(ctx: ToolContext, workspace: Path) -> None:
    for source in ("startup", "resume", "new", "switch"):
        runner = FakeRunner([ok_run(json.dumps({"context": f"from {source}"}))])
        agent, provider, _ = make(ctx, [done()], runner, entry("SessionStart"))
        events: list[AgentEvent] = []

        await agent.session_started(source, events.append)
        await run(agent)

        payload = runner.calls[0][1]
        assert payload["event"] == "SessionStart" and payload["source"] == source and payload["cwd"] == str(workspace)
        assert system_texts(provider.requests[0][0]) == [f"[hook context]\nfrom {source}"]
        assert Notice("info", f"hook SessionStart (global) context: from {source}") in events


async def test_session_start_gets_the_recorders_session_id(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    recorder = FileRecorder(workspace_dir(tmp_path / "sessions", workspace), workspace)
    recorder.message(Message("user", "hi"), None)
    runner = FakeRunner()
    agent, _, _ = make(ctx, [], runner, entry("SessionStart"), recorder=recorder)

    await agent.session_started("resume", lambda e: None)
    recorder.close()

    assert runner.calls[0][1]["session_id"] == recorder.session_id != ""


async def test_reset_drops_context_a_session_start_added_for_the_old_session(ctx: ToolContext) -> None:
    runner = FakeRunner([ok_run(json.dumps({"context": "old"})), ok_run(json.dumps({"context": "new"}))])
    agent, provider, _ = make(ctx, [done()], runner, entry("SessionStart"))
    await agent.session_started("startup", lambda e: None)

    agent.reset(recorder=MemoryRecorder())
    await agent.session_started("new", lambda e: None)
    await run(agent)

    assert system_texts(provider.requests[0][0]) == ["[hook context]\nnew"]


async def test_a_session_start_hook_cannot_block_and_a_block_is_a_warning(ctx: ToolContext) -> None:
    runner = FakeRunner([exit_run(2, "no")])
    agent, provider, _ = make(ctx, [done()], runner, entry("SessionStart"))
    events: list[AgentEvent] = []

    await agent.session_started("startup", events.append)
    await run(agent)

    assert any(isinstance(e, Notice) and e.level == "warn" and "SessionStart" in e.text for e in events)
    assert system_texts(provider.requests[0][0]) == []


async def test_a_token_in_session_start_context_is_redacted(ctx: ToolContext) -> None:
    runner = FakeRunner([ok_run(json.dumps({"context": f"key {GITHUB}"}))])
    agent, provider, _ = make(ctx, [done()], runner, entry("SessionStart"))
    events: list[AgentEvent] = []

    await agent.session_started("startup", events.append)
    await run(agent)

    assert GITHUB not in system_texts(provider.requests[0][0])[0]
    assert not any(isinstance(e, Notice) and GITHUB in e.text for e in events)


async def test_session_start_without_hooks_runs_nothing(ctx: ToolContext) -> None:
    runner = FakeRunner()
    agent, _, _ = make(ctx, [], runner)

    await agent.session_started("startup", lambda e: None)

    assert runner.calls == []


async def test_a_failing_sink_during_session_start_never_raises(ctx: ToolContext) -> None:
    runner = FakeRunner([ok_run(json.dumps({"context": "c"}))])
    agent, _, _ = make(ctx, [], runner, entry("SessionStart"))

    def broken(event: AgentEvent) -> None:
        raise RuntimeError("sink broke")

    await agent.session_started("startup", broken)  # must not raise


async def test_hook_context_survives_a_provider_error_on_the_first_request(ctx: ToolContext) -> None:
    agent, provider, _ = make(ctx, [[ProviderError("boom")], done()], FakeRunner())

    await agent.run_turn("one", lambda e: None, context=["keep me"])
    await run(agent, "two")

    assert system_texts(provider.requests[1][0]) == ["[hook context]\nkeep me"]


async def _cancel_at(agent: Agent, gate: Gate, text: str, context: list[str]) -> None:
    task = asyncio.ensure_future(agent.run_turn(text, lambda e: None, context=context))
    await asyncio.wait_for(gate.reached.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_cancelled_prompts_context_is_not_sent_with_the_next_prompt(ctx: ToolContext) -> None:
    gate = Gate()
    agent, provider, _ = make(ctx, [[gate, *reply()], done()], FakeRunner())

    await _cancel_at(agent, gate, "one", ["for the cancelled prompt"])
    await run(agent, "two")

    assert system_texts(provider.requests[1][0]) == []


async def test_session_start_context_survives_a_cancel_before_a_request_is_answered(ctx: ToolContext) -> None:
    gate = Gate()
    runner = FakeRunner([ok_run(json.dumps({"context": "welcome"}))])
    agent, provider, _ = make(ctx, [[gate, *reply()], done()], runner, entry("SessionStart"))
    await agent.session_started("startup", lambda e: None)

    await _cancel_at(agent, gate, "one", ["prompt ctx"])
    await run(agent, "two")

    assert system_texts(provider.requests[1][0]) == ["[hook context]\nwelcome"]


async def test_session_start_context_is_spent_once_the_model_started_answering(ctx: ToolContext) -> None:
    gate = Gate()
    runner = FakeRunner([ok_run(json.dumps({"context": "welcome"}))])
    agent, provider, _ = make(ctx, [[TextDelta("par"), gate, *reply()], done()], runner, entry("SessionStart"))
    await agent.session_started("startup", lambda e: None)

    await _cancel_at(agent, gate, "one", [])
    await run(agent, "two")

    assert system_texts(provider.requests[1][0]) == []


async def test_stop_notes_do_not_make_compaction_see_extra_turns(ctx: ToolContext) -> None:
    runner = FakeRunner([exit_run(2, "again")])
    agent, _, _ = make(ctx, [reply() for _ in range(4)], runner, entry("Stop"))

    await run(agent, "one")

    assert len(turn_starts(agent.history)) == 1


async def test_undo_removes_stop_notes_with_their_turn(ctx: ToolContext) -> None:
    runner = FakeRunner([exit_run(2, "again")])
    agent, _, _ = make(ctx, [reply() for _ in range(4)], runner, entry("Stop"))

    await run(agent, "one")
    prompt = await agent.undo(lambda e: None)

    assert prompt == "one" and agent.history == []


async def test_an_oversized_hook_context_is_cut_with_a_note(ctx: ToolContext) -> None:
    agent, provider, _ = make(ctx, [done()], FakeRunner())

    await agent.run_turn("go", lambda e: None, context=["x" * (HOOK_CONTEXT_LIMIT * 3)])

    note = system_texts(provider.requests[0][0])[0]
    assert len(note) < HOOK_CONTEXT_LIMIT + 200 and note.endswith(HOOK_CONTEXT_CUT)


def test_hook_exception_warnings_are_cut_and_escaped() -> None:
    notice = hook_failed("Stop", RuntimeError("bad\x1b[31m" + "x" * 500))

    assert notice.level == "warn" and notice.text.startswith("hook Stop failed: bad\\x1b[31mxxx")
    assert len(notice.text) < 260


async def test_skipping_stop_hooks_on_the_last_step_says_so(ctx: ToolContext) -> None:
    agent, _, _ = make(ctx, [reply()], FakeRunner(), entry("Stop"))
    agent.max_steps = 1

    events = await run(agent)

    assert Notice("info", "Stop hooks skipped: no step left") in events and events[-1] == TurnEnded("done")
