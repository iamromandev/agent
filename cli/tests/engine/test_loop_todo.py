"""The todo list in the loop: the note on each request, events, undo, compaction and reset."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
from src.engine.approval import Choice, Decision
from src.engine.events import AgentEvent, TodosChanged
from src.engine.loop import Agent
from src.engine.mode import Mode
from src.engine.provider.base import Done, Message, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.session import NullRecorder
from src.engine.todo import TodoItem, render_note
from src.engine.tools import default_tools, read_only_tools
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


async def test_after_a_todo_call_the_next_request_has_the_note_before_the_last_user_message(ctx: ToolContext) -> None:
    agent, provider, events = make(ctx, [reply(todo_call("t1", *TWO)), reply(TextDelta("ok"))])
    await agent.run_turn("do it", events.append)

    first, second = provider.requests
    assert notes(first[0]) == []
    assert notes(second[0]) == [(1, render_note(LIST_TWO))]
    assert second[0][2] == Message("user", "do it")


async def test_the_note_goes_before_the_latest_user_message_of_a_later_turn(ctx: ToolContext) -> None:
    agent, provider, events = make(ctx, [reply(todo_call("t1", *TWO)), reply(TextDelta("ok")), reply(TextDelta("again"))])
    await agent.run_turn("one", events.append)
    await agent.run_turn("two", events.append)

    messages = provider.requests[-1][0]
    [(index, _)] = notes(messages)
    assert messages[index + 1] == Message("user", "two")
    assert messages[index - 1].content == "ok"


async def test_the_note_is_not_in_the_history_or_the_saved_session(ctx: ToolContext) -> None:
    recorder = MemoryRecorder()
    agent, _, events = make(ctx, [reply(todo_call("t1", *TWO)), reply(TextDelta("ok"))], recorder)
    await agent.run_turn("do it", events.append)

    assert all(m.role != "system" for m in agent.history)
    assert all(m.role != "system" and "Current todo list" not in m.content for m, _ in recorder.messages)
    assert agent.todos == LIST_TWO


async def test_without_a_todo_call_there_is_no_note(ctx: ToolContext) -> None:
    agent, provider, events = make(ctx, [reply(TextDelta("hi"))])
    await agent.run_turn("hello", events.append)

    assert [m.role for m in provider.requests[0][0]] == ["system", "user"]


async def test_the_note_counts_in_the_estimate(ctx: ToolContext) -> None:
    from src.engine.compaction import message_tokens

    history = [Message("user", "q"), Message("assistant", "a")]
    with_list, _, _ = make(ctx, [])
    without, _, _ = make(ctx, [])
    with_list.reset(recorder=NullRecorder(), history=history, todos=LIST_TWO)
    without.reset(recorder=NullRecorder(), history=history)

    assert with_list.estimate() - without.estimate() == message_tokens(Message("system", render_note(LIST_TWO)))


async def test_a_list_that_is_all_done_gets_no_note_on_the_next_turn(ctx: ToolContext) -> None:
    script = [
        reply(todo_call("t1", ("A", "pending"))),
        reply(todo_call("t2", ("A", "done"))),
        reply(TextDelta("finished")),
        reply(TextDelta("next")),
    ]
    agent, provider, events = make(ctx, script)
    await agent.run_turn("one", events.append)
    assert notes(provider.requests[1][0]) != []
    assert notes(provider.requests[2][0]) == []  # all done: no note even in the same turn

    await agent.run_turn("two", events.append)
    assert notes(provider.requests[3][0]) == []


async def test_an_empty_list_clears_it(ctx: ToolContext) -> None:
    agent, provider, events = make(
        ctx, [reply(todo_call("t1", *TWO)), reply(ToolCallEvent(ToolCall("t2", "todo", {"items": []}))), reply(TextDelta("x"))]
    )
    await agent.run_turn("go", events.append)

    assert agent.todos == ()
    assert notes(provider.requests[2][0]) == []


async def test_todo_runs_in_plan_mode_without_asking(ctx: ToolContext) -> None:
    provider = ScriptedProvider([reply(todo_call("t1", *TWO)), reply(TextDelta("ok"))], context_window=100_000)
    approver = RecordingApprover()
    agent = Agent(provider, default_tools(), approver, ctx, 25, recorder=MemoryRecorder())
    agent.set_mode(Mode.PLAN)

    await agent.run_turn("plan", lambda e: None)

    assert "todo" in provider.requests[0][1]
    assert approver.requests == []
    assert agent.todos == LIST_TWO


async def test_a_successful_call_emits_the_list_and_a_bad_one_does_not(ctx: ToolContext) -> None:
    bad = ToolCallEvent(ToolCall("t2", "todo", {"items": [{"content": "x", "status": "doing"}]}))
    agent, _, events = make(ctx, [reply(todo_call("t1", *TWO)), reply(bad), reply(TextDelta("ok"))])
    await agent.run_turn("go", events.append)

    assert [e for e in events if isinstance(e, TodosChanged)] == [TodosChanged(LIST_TWO)]
    assert agent.todos == LIST_TWO


async def test_undo_restores_the_list_from_before_the_turn_and_says_so(ctx: ToolContext) -> None:
    script = [
        reply(todo_call("t1", ("First", "pending"))),
        reply(TextDelta("a")),
        reply(todo_call("t2", *TWO)),
        reply(TextDelta("b")),
        reply(TextDelta("c")),
    ]
    agent, provider, events = make(ctx, script)
    await agent.run_turn("one", events.append)
    await agent.run_turn("two", events.append)
    assert agent.todos == LIST_TWO
    events.clear()

    await agent.undo(events.append)

    first = (TodoItem("First", "pending"),)
    assert agent.todos == first
    assert [e for e in events if isinstance(e, TodosChanged)] == [TodosChanged(first)]
    await agent.run_turn("again", events.append)
    assert notes(provider.requests[-1][0])[0][1] == render_note(first)


async def test_undo_of_the_only_todo_turn_clears_the_list(ctx: ToolContext) -> None:
    agent, _, events = make(ctx, [reply(todo_call("t1", *TWO)), reply(TextDelta("a"))])
    await agent.run_turn("one", events.append)
    events.clear()

    await agent.undo(events.append)

    assert agent.todos == ()
    assert [e for e in events if isinstance(e, TodosChanged)] == [TodosChanged(())]


async def test_compaction_keeps_the_list_and_the_note(ctx: ToolContext) -> None:
    script = [
        reply(todo_call("t1", *TWO)),
        reply(TextDelta("a")),
        reply(TextDelta("b")),
        reply(TextDelta("c")),
        reply(TextDelta("summary of the start")),
        reply(TextDelta("d")),
    ]
    agent, provider, events = make(ctx, script)
    for text in ("one", "two", "three"):
        await agent.run_turn(text, events.append)
    assert any(c.name == "todo" for m in agent.history for c in m.tool_calls)

    await agent.compact(events.append)

    assert not any(c.name == "todo" for m in agent.history for c in m.tool_calls)  # the call was summarized away
    assert agent.todos == LIST_TWO
    await agent.run_turn("four", events.append)
    assert notes(provider.requests[-1][0])[0][1] == render_note(LIST_TWO)


async def test_reset_takes_the_given_list_and_clears_it_otherwise(ctx: ToolContext) -> None:
    agent, _, _ = make(ctx, [])
    agent.reset(recorder=NullRecorder(), todos=LIST_TWO)
    assert agent.todos == LIST_TWO

    agent.reset(recorder=NullRecorder())
    assert agent.todos == ()


async def test_the_task_child_has_no_todo_and_no_note(ctx: ToolContext) -> None:
    script = [
        reply(todo_call("t1", *TWO)),
        reply(ToolCallEvent(ToolCall("k1", "task", {"description": "look", "prompt": "find it"}))),
        reply(TextDelta("found")),
        reply(TextDelta("done")),
    ]
    provider = ScriptedProvider(script, context_window=100_000)
    agent = Agent(
        provider, default_tools(), RecordingApprover(), ctx, 25, recorder=MemoryRecorder(), task_tools=read_only_tools()
    )
    await agent.run_turn("go", lambda e: None)

    child_messages, child_tools = provider.requests[2]
    assert "todo" not in child_tools
    assert notes(child_messages) == []
    assert "todo" in provider.requests[1][1]


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


async def test_the_estimate_after_a_report_does_not_count_the_note_twice(ctx: ToolContext) -> None:
    from src.engine.compaction import message_tokens

    agent, _, _ = make(ctx, [reply(todo_call("t1", *TWO)), reply(TextDelta("ok"))])
    await agent.run_turn("go", lambda e: None)
    assert agent._reported is not None
    tail = sum(message_tokens(m) for m in agent.history[agent._reported.history_len :])

    assert agent.estimate() == agent._reported.total + tail


async def test_a_request_without_todo_calls_has_exactly_the_plain_messages_and_tools(ctx: ToolContext) -> None:
    agent, provider, events = make(ctx, [reply(TextDelta("hi"))])
    await agent.run_turn("hello", events.append)

    messages, tool_names = provider.requests[0]
    assert [m.role for m in messages] == ["system", "user"]
    assert "Current todo list" not in messages[0].content
    assert list(tool_names) == ["read", "glob", "grep", "write", "edit", "bash", "todo"]
