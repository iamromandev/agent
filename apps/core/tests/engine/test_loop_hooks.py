"""PreToolUse and PostToolUse inside the loop: order against the dialogs, redaction, limits, resume, no cost without hooks."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path

import pytest
from core.engine.approval import Choice, Decision
from core.engine.events import AgentEvent, Notice, ToolFinished, TurnEnded
from core.engine.guardrails.setup import prepare as prepare_guard
from core.engine.hooks.config import DEFAULT_TIMEOUT, Event, HookEntry
from core.engine.hooks.runner import HookRun
from core.engine.hooks.setup import Hooks
from core.engine.loop import HOOK_CONTEXT_CUT, HOOK_CONTEXT_LIMIT, INTERRUPTED, Agent
from core.engine.permissions.rules import Source
from core.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from core.engine.session import Recorder
from core.engine.tools import default_tools, read_only_tools
from core.engine.tools.base import ToolContext
from core.engine.tools.text import TOOL_OUTPUT_LIMIT
from core.testing.fakes import FakeRunner, Gate, MemoryRecorder, RecordingApprover, ScriptedProvider, exit_run, ok_run

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
    return ToolContext(workspace, 5.0, guard=prepare_guard(workspace, None, None, {}, home=home))


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


async def test_ac7_decision_allow_does_not_skip_the_approval_dialog(ctx: ToolContext, workspace: Path) -> None:
    runner = FakeRunner([ok_run(json.dumps({"decision": "allow"}))])
    agent, _, approver = make(
        ctx, [call("c1", "bash", command="touch ran"), done()], runner, entry("PreToolUse"), decisions=[Decision(Choice.DENY)]
    )

    await run(agent)

    assert [r.tool for r in approver.requests] == ["bash"]
    assert tool_message(agent, "c1").startswith("User denied")
    assert not (workspace / "ran").exists()


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


def post_calls(runner: FakeRunner) -> list[str]:
    return [c[1]["tool"] for c in runner.calls if c[1]["event"] == "PostToolUse"]


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


class HangingRunner:
    """A runner whose hooks never finish: the test cancels the turn while one runs."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False

    async def run(self, command: str, stdin: bytes, timeout: float, env_extra: object, *, event: str = "") -> HookRun:
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


async def test_pending_hook_context_counts_in_the_estimate_and_reset_drops_it(ctx: ToolContext) -> None:
    agent, _, _ = make(ctx, [], FakeRunner([ok_run(json.dumps({"context": "x" * 4000}))]), entry("SessionStart"))
    before = agent.estimate()
    await agent.session_started("startup", lambda e: None)

    assert agent.estimate() >= before + 900

    agent.reset(recorder=MemoryRecorder())

    assert agent.estimate() == before


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


async def test_stop_continuations_count_toward_the_step_limit_and_the_hook_is_not_run_on_the_last_step(
    ctx: ToolContext,
) -> None:
    runner = FakeRunner([exit_run(2, "more")])
    agent, _, _ = make(ctx, [reply(), reply()], runner, entry("Stop"))
    agent.max_steps = 2

    events = await run(agent)

    assert [c[1]["continuations"] for c in runner.calls] == [0]  # the second reply used the last step
    assert user_notes(agent) == ["[hook Stop] more"] and events[-1] == TurnEnded("done")


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


async def test_a_failing_sink_during_session_start_never_raises(ctx: ToolContext) -> None:
    runner = FakeRunner([ok_run(json.dumps({"context": "c"}))])
    agent, _, _ = make(ctx, [], runner, entry("SessionStart"))

    def broken(event: AgentEvent) -> None:
        raise RuntimeError("sink broke")

    await agent.session_started("startup", broken)  # must not raise


async def _cancel_at(agent: Agent, gate: Gate, text: str, context: list[str]) -> None:
    task = asyncio.ensure_future(agent.run_turn(text, lambda e: None, context=context))
    await asyncio.wait_for(gate.reached.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_an_oversized_hook_context_is_cut_with_a_note(ctx: ToolContext) -> None:
    agent, provider, _ = make(ctx, [done()], FakeRunner())

    await agent.run_turn("go", lambda e: None, context=["x" * (HOOK_CONTEXT_LIMIT * 3)])

    note = system_texts(provider.requests[0][0])[0]
    assert len(note) < HOOK_CONTEXT_LIMIT + 200 and note.endswith(HOOK_CONTEXT_CUT)
