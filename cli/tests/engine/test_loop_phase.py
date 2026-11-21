"""`Phase` events: what the main agent reports while a turn runs (display only, never recorded)."""

from __future__ import annotations

from datetime import date

from src.engine.approval import Choice, Decision
from src.engine.events import AgentEvent, Phase, TaskProgress, TurnEnded
from src.engine.hooks.config import DEFAULT_TIMEOUT, Event, HookEntry
from src.engine.hooks.setup import Hooks
from src.engine.loop import Agent
from src.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.tools import default_tools, read_only_tools
from src.engine.tools.base import ToolContext

from tests.fakes import FakeRunner, Gate, MemoryRecorder, RecordingApprover, ScriptedProvider, exit_run

Script = list[list[ProviderEvent | Gate]]


def call(call_id: str, name: str, **args: object) -> list[ProviderEvent | Gate]:
    return [ToolCallEvent(ToolCall(call_id, name, dict(args))), Done(10, 5)]


def say(text: str = "done") -> list[ProviderEvent | Gate]:
    return [TextDelta(text), Done(1, 1)]


def make(
    ctx: ToolContext,
    script: Script,
    *,
    decisions: list[Decision] | None = None,
    hooks: Hooks | None = None,
    task: bool = False,
    max_steps: int = 25,
    recorder: MemoryRecorder | None = None,
) -> Agent:
    return Agent(
        ScriptedProvider(script, context_window=100_000),
        default_tools(),
        RecordingApprover(decisions),
        ctx,
        max_steps,
        today=lambda: date(2026, 11, 14),
        recorder=recorder or MemoryRecorder(),
        hooks=hooks,
        task_tools=read_only_tools() if task else None,
    )


def hook(event: Event, runner: FakeRunner, ctx: ToolContext) -> Hooks:
    entries = [HookEntry(event, "x", None, DEFAULT_TIMEOUT, "global")]
    return Hooks(entries, [], runner, lambda: True, ctx.workspace)


async def run(agent: Agent) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    await agent.run_turn("go", events.append)
    return events


def phases(events: list[AgentEvent]) -> list[tuple[str, int, str]]:
    return [(e.kind, e.step, e.detail) for e in events if isinstance(e, Phase)]


async def test_a_text_only_turn_waits_then_writes(ctx: ToolContext) -> None:
    events = await run(make(ctx, [say("a")]))

    assert phases(events) == [("waiting", 1, ""), ("writing", 1, "")]
    assert next(e for e in events if isinstance(e, Phase)).max_steps == 25


async def test_writing_is_reported_once_per_request(ctx: ToolContext) -> None:
    events = await run(make(ctx, [[TextDelta("a"), TextDelta("b"), Done(1, 1)]]))

    assert phases(events) == [("waiting", 1, ""), ("writing", 1, "")]


async def test_a_tool_then_text_counts_two_steps(ctx: ToolContext) -> None:
    (ctx.workspace / "a.txt").write_text("a")
    events = await run(make(ctx, [call("c1", "read", path="a.txt"), say()]))

    assert phases(events) == [("waiting", 1, ""), ("running", 1, "read"), ("waiting", 2, ""), ("writing", 2, "")]


async def test_the_step_limit_is_reported(ctx: ToolContext) -> None:
    events = await run(make(ctx, [call("c1", "read", path="nope")], max_steps=1))

    assert next(e for e in events if isinstance(e, Phase)).max_steps == 1


async def test_an_approved_call_goes_running_approval_running(ctx: ToolContext) -> None:
    events = await run(make(ctx, [call("c1", "write", path="o.txt", content="x"), say()], decisions=[Decision(Choice.ONCE)]))

    assert phases(events)[:4] == [
        ("waiting", 1, ""),
        ("running", 1, "write"),
        ("approval", 1, "write"),
        ("running", 1, "write"),
    ]


async def test_a_denied_call_goes_straight_to_the_next_request(ctx: ToolContext) -> None:
    events = await run(make(ctx, [call("c1", "write", path="o.txt", content="x"), say()], decisions=[Decision(Choice.DENY)]))

    assert phases(events) == [
        ("waiting", 1, ""),
        ("running", 1, "write"),
        ("approval", 1, "write"),
        ("waiting", 2, ""),
        ("writing", 2, ""),
    ]


async def test_a_protected_path_dialog_is_an_approval_phase(ctx: ToolContext) -> None:
    (ctx.workspace / ".env").write_text("A=1\n")
    events = await run(make(ctx, [call("c1", "read", path=".env"), say()], decisions=[Decision(Choice.ONCE)]))

    assert ("approval", 1, "read") in phases(events)
    assert phases(events).count(("running", 1, "read")) == 2


async def test_a_refused_call_never_shows_running(ctx: ToolContext) -> None:
    events = await run(make(ctx, [call("c1", "nope"), call("c2", "read"), say()]))  # unknown tool, invalid arguments

    assert phases(events) == [("waiting", 1, ""), ("waiting", 2, ""), ("waiting", 3, ""), ("writing", 3, "")]


async def test_a_post_hook_is_a_hook_phase(ctx: ToolContext) -> None:
    (ctx.workspace / "a.txt").write_text("a")
    agent = make(ctx, [call("c1", "read", path="a.txt"), say()], hooks=hook("PostToolUse", FakeRunner(), ctx))
    events = await run(agent)

    assert phases(events)[:4] == [("waiting", 1, ""), ("running", 1, "read"), ("hook", 1, "PostToolUse"), ("waiting", 2, "")]


async def test_a_pre_hook_is_followed_by_running_again(ctx: ToolContext) -> None:
    (ctx.workspace / "a.txt").write_text("a")
    agent = make(ctx, [call("c1", "read", path="a.txt"), say()], hooks=hook("PreToolUse", FakeRunner(), ctx))
    events = await run(agent)

    assert phases(events)[:3] == [("waiting", 1, ""), ("hook", 1, "PreToolUse"), ("running", 1, "read")]


async def test_a_call_a_pre_hook_blocks_never_shows_running(ctx: ToolContext) -> None:
    (ctx.workspace / "a.txt").write_text("a")
    agent = make(ctx, [call("c1", "read", path="a.txt"), say()], hooks=hook("PreToolUse", FakeRunner([exit_run(2, "no")]), ctx))
    events = await run(agent)

    assert phases(events) == [("waiting", 1, ""), ("hook", 1, "PreToolUse"), ("waiting", 2, ""), ("writing", 2, "")]


async def test_a_stop_hook_block_continues_with_the_next_step(ctx: ToolContext) -> None:
    runner = FakeRunner([exit_run(2, "more"), exit_run(0)])
    events = await run(make(ctx, [say("a"), say("b")], hooks=hook("Stop", runner, ctx)))

    assert phases(events) == [
        ("waiting", 1, ""),
        ("writing", 1, ""),
        ("hook", 1, "Stop"),
        ("waiting", 2, ""),
        ("writing", 2, ""),
        ("hook", 2, "Stop"),
    ]


async def test_the_task_child_sends_no_phase(ctx: ToolContext) -> None:
    (ctx.workspace / "a.txt").write_text("a")
    script = [call("t1", "task", description="d", prompt="p"), call("r", "read", path="a.txt"), say("report"), say()]
    events = await run(make(ctx, script, task=True))

    inner = [e.event for e in events if isinstance(e, TaskProgress)]
    assert inner
    assert not any(isinstance(e, Phase) for e in inner)
    assert phases(events) == [("waiting", 1, ""), ("running", 1, "task"), ("waiting", 2, ""), ("writing", 2, "")]


async def test_a_sink_that_raises_on_phase_leaves_the_turn_unchanged(ctx: ToolContext) -> None:
    (ctx.workspace / "a.txt").write_text("a")
    script = [call("c1", "read", path="a.txt"), say()]
    normal = make(ctx, list(script))
    await run(normal)
    broken = make(ctx, list(script))
    seen: list[AgentEvent] = []

    def sink(event: AgentEvent) -> None:
        if isinstance(event, Phase):
            raise RuntimeError("boom")
        seen.append(event)

    await broken.run_turn("go", sink)

    assert [e for e in seen if isinstance(e, TurnEnded)] == [TurnEnded("done")]
    assert broken.history == normal.history


async def test_no_phase_reaches_the_history_or_the_recorder(ctx: ToolContext) -> None:
    recorder = MemoryRecorder()
    agent = make(ctx, [say()], recorder=recorder)
    await run(agent)

    assert "Phase" not in repr(agent.history)
    assert "Phase" not in repr(recorder.records)
    assert [m.role for m, _ in recorder.messages] == ["user", "assistant"]

