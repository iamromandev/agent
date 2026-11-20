"""Usage records in the loop (R1, R3, R4, AC1, AC3, AC11): one per finished main-agent model call."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from src.engine.events import AgentEvent, Notice, TurnEnded
from src.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.session import NullRecorder, SessionWriteError
from src.engine.tools.base import ToolContext
from src.engine.usage import UsageRecord, UsageTotals

from tests.engine.test_compaction_loop import big_text_history, make_agent, seed
from tests.engine.test_task import _make as make_task_agent
from tests.engine.test_task import _say, _task
from tests.fakes import Gate, MemoryRecorder


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.py").write_text("print('hi')\n")
    return root.resolve()


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, bash_timeout=5.0)


def call_step() -> list[ProviderEvent | Gate]:
    return [ToolCallEvent(ToolCall("c1", "read", {"path": "a.py"})), Done(100, 20)]


def reply_step(text: str = "ok") -> list[ProviderEvent | Gate]:
    return [TextDelta(text), Done(100, 20)]


async def run(agent, text: str = "go") -> list[AgentEvent]:
    events: list[AgentEvent] = []
    await agent.run_turn(text, events.append)
    return events


async def test_two_calls_write_two_records_with_model_and_tokens(ctx: ToolContext) -> None:
    from src.engine.tools import default_tools

    agent, provider, recorder = make_agent(ctx, [call_step(), reply_step()], tools=default_tools())
    provider.model = "qwen"
    await run(agent)

    records = recorder.usage_records
    assert [(r.model, r.input_tokens, r.output_tokens, r.compaction) for r in records] == [("qwen", 100, 20, False)] * 2
    assert all(r.duration_ms >= 0 and r.at.endswith("Z") for r in records)
    assert agent.usage.total.input_tokens == 200 and agent.usage.calls == 2


async def test_duration_covers_the_whole_stream(ctx: ToolContext) -> None:
    agent, provider, recorder = make_agent(ctx, [reply_step()])
    provider.delay = 0.05  # per event: two events
    await run(agent)
    assert recorder.usage_records[0].duration_ms >= 90


async def test_a_cancelled_call_writes_no_record(ctx: ToolContext) -> None:
    gate = Gate()
    agent, _, recorder = make_agent(ctx, [[TextDelta("a"), gate, Done(1, 1)]])
    task = asyncio.create_task(run(agent))
    await gate.reached.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert recorder.usage_records == [] and agent.usage.calls == 0


async def test_a_failed_call_writes_no_record(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [])  # script exhausted: a provider error
    await run(agent)
    assert recorder.usage_records == []


async def test_the_task_child_writes_nothing(ctx: ToolContext) -> None:
    recorder = MemoryRecorder()
    agent, _, _ = make_task_agent(ctx, [_task(), _say("report"), _say("final")], recorder=recorder)
    await run(agent)

    assert len(recorder.usage_records) == 2  # the parent's two calls, not the child's
    assert agent.usage.calls == 2


async def test_a_compaction_summary_writes_a_record_marked_compaction(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [reply_step("gist"), reply_step("ok")])
    seed(agent, big_text_history())
    await run(agent, "u4")

    assert [r.compaction for r in recorder.usage_records] == [True, False]
    assert agent.usage.total.compactions == 1


async def test_reset_seeds_the_totals_and_a_turn_adds_to_them(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [reply_step()])
    seeded = UsageTotals().add(UsageRecord("m", 5, 6, 7, False, "t"))
    agent.reset(recorder=recorder, usage=seeded, unrecorded=True)
    assert agent.usage == seeded and agent.usage_unrecorded is True
    await run(agent)
    assert agent.usage.total.input_tokens == 105


async def test_reset_with_defaults_gives_zero(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply_step()])
    await run(agent)
    agent.reset(recorder=NullRecorder())
    assert agent.usage == UsageTotals() and agent.usage_unrecorded is False


async def test_undo_keeps_the_totals(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply_step()])
    await run(agent)
    before = agent.usage
    await agent.undo(lambda event: None)
    assert agent.usage == before


class FailingUsageRecorder(MemoryRecorder):
    def usage(self, record: UsageRecord) -> None:
        raise SessionWriteError("disk full")


async def test_a_failing_recorder_gives_the_save_error_and_the_turn_completes(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply_step()], recorder=FailingUsageRecorder())
    events = await run(agent)

    assert events[-1] == TurnEnded("done")
    errors = [e for e in events if isinstance(e, Notice) and e.level == "error"]
    assert errors and "Could not save the session" in errors[0].text
    assert agent.usage.calls == 1
