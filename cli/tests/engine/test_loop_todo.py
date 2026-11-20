"""The todo list in the loop: the note on each request, events, undo, compaction and reset."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
from src.engine.approval import Choice, Decision
from src.engine.events import AgentEvent
from src.engine.loop import Agent
from src.engine.provider.base import Done, Message, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.session import NullRecorder
from src.engine.todo import TodoItem
from src.engine.tools import default_tools
from src.engine.tools.base import ToolContext

from tests.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path.resolve(), bash_timeout=5.0)


def todo_call(call_id: str, *pairs: tuple[str, str]) -> ToolCallEvent:
    items = [{"content": c, "status": s} for c, s in pairs]
    return ToolCallEvent(ToolCall(call_id, "todo", {"items": items}))


def reply(*events: ProviderEvent | Gate) -> list[ProviderEvent | Gate]:
    return [*events, Done(10, 5)]


def make(
    ctx: ToolContext, script: list[list[ProviderEvent | Gate]], recorder: Any = None, window: int = 100_000, **options: Any
) -> tuple[Agent, ScriptedProvider, list[AgentEvent]]:
    provider = ScriptedProvider(script, context_window=window)
    agent = Agent(
        provider,
        default_tools(),
        RecordingApprover([Decision(Choice.DENY)]),  # todo must never ask
        ctx,
        25,
        today=lambda: date(2026, 11, 15),
        recorder=recorder if recorder is not None else MemoryRecorder(),
        **options,
    )
    return agent, provider, []


def notes(messages: list[Message]) -> list[tuple[int, str]]:
    return [(i, m.content) for i, m in enumerate(messages) if m.role == "system" and i > 0]


TWO = (("Find uses", "in_progress"), ("Rename", "pending"))
LIST_TWO = (TodoItem("Find uses", "in_progress"), TodoItem("Rename", "pending"))


async def test_a_todo_call_that_never_ran_leaves_the_list_unchanged(ctx: ToolContext) -> None:
    import asyncio

    bash = ToolCallEvent(ToolCall("b1", "bash", {"command": "sleep 30"}))
    script = [reply(todo_call("t1", *TWO)), reply(bash, todo_call("t2", ("Other", "pending"))), reply(TextDelta("ok"))]
    provider = ScriptedProvider(script, context_window=100_000)
    agent = Agent(
        provider,
        default_tools(),
        RecordingApprover([Decision(Choice.ONCE)]),
        ctx,
        25,
        recorder=MemoryRecorder(),
    )
    task = asyncio.create_task(agent.run_turn("go", lambda e: None))
    for _ in range(500):
        if agent._inflight == "b1":
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert agent.todos == LIST_TWO
    replayed, _, _ = make(ctx, [])
    replayed.reset(recorder=NullRecorder(), history=list(agent.history))
    assert replayed.todos == LIST_TWO
