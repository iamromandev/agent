"""Compaction in the loop (R12-R18, AC10-AC15): when it runs, what it keeps, how it fails, what it records."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from src.engine.compaction import (
    is_summary,
)
from src.engine.events import AgentEvent, Compacting, Notice, TurnEnded
from src.engine.loop import Agent, AgentBusy
from src.engine.provider.base import Done, Message, ProviderError, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.session import FileRecorder, SessionStore, ToolMeta, workspace_dir
from src.engine.tools.base import Tool, ToolContext

from tests.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider

TODAY = date(2026, 11, 12)
STILL_FULL = "Context is still full — start a new session with /new"
NOT_SUMMARIZED = "Could not summarize the earlier conversation; continuing with pruned history"
WINDOW = 1000  # limit 800; the system prompt alone is about 140 tokens, the default tool specs about 780 more


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.py").write_text("print('hi')\n")
    return root.resolve()


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, bash_timeout=5.0)


class Events:
    def __init__(self) -> None:
        self.items: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.items.append(event)

    def of(self, kind: type) -> list[Any]:
        return [e for e in self.items if isinstance(e, kind)]

    def notices(self) -> list[tuple[str, str]]:
        return [(n.level, n.text) for n in self.of(Notice)]


def reply(*events: ProviderEvent | Gate) -> list[ProviderEvent | Gate]:
    return [*events, Done(10, 5)]


def make_agent(
    ctx: ToolContext,
    script: list[list[ProviderEvent | Gate]],
    *,
    window: int = WINDOW,
    tools: Sequence[Tool] = (),
    recorder: Any = None,
    max_steps: int = 25,
    compact_at: float = 0.8,
    keep_turns: int = 2,
) -> tuple[Agent, ScriptedProvider, MemoryRecorder]:
    provider = ScriptedProvider(script, context_window=window)
    recorder = recorder if recorder is not None else MemoryRecorder()
    agent = Agent(
        provider,
        tools,
        RecordingApprover(),
        ctx,
        max_steps,
        today=lambda: TODAY,
        recorder=recorder,
        compact_at=compact_at,
        compact_keep_turns=keep_turns,
    )
    return agent, provider, recorder


def tool_turn(n: int, output_chars: int, chatter: int = 0) -> tuple[list[Message], dict[str, ToolMeta]]:
    call = ToolCall(f"c{n}", "read", {"path": f"src/{n}.py"})
    messages = [
        Message("user", f"u{n}"),
        Message("assistant", "", [call]),
        Message("tool", f"{n}" * output_chars, tool_call_id=f"c{n}", name="read"),
        Message("assistant", f"a{n}" + "." * chatter),  # chatter: text that pruning leaves, so a summary pays
    ]
    return messages, {f"c{n}": ToolMeta(f"read src/{n}.py", f"{n}0 lines")}


def text_turn(n: int, reply_chars: int = 10) -> list[Message]:
    return [Message("user", f"u{n}"), Message("assistant", f"a{n}" + "." * reply_chars)]


def seed(agent: Agent, messages: list[Message], meta: dict[str, ToolMeta] | None = None) -> None:
    """Put a finished conversation in the agent (and in the recorder, as a real run would have)."""
    for m in messages:
        agent._record(m, (meta or {}).get(m.tool_call_id or ""))


def tool_history(outputs: Sequence[int], chatter: int = 0) -> tuple[list[Message], dict[str, ToolMeta]]:
    messages: list[Message] = []
    meta: dict[str, ToolMeta] = {}
    for n, chars in enumerate(outputs, start=1):
        turn, turn_meta = tool_turn(n, chars, chatter)
        messages += turn
        meta |= turn_meta
    return messages, meta


def sent_history(provider: ScriptedProvider, index: int) -> list[Message]:
    return provider.requests[index][0][1:]  # without the system message


def stub(n: int) -> str:
    return f"[output pruned: read src/{n}.py · {n}0 lines]"


# --- settings of the agent ---------------------------------------------------------------------


@pytest.mark.parametrize("bad", [1.5])
def test_compact_at_must_be_above_zero_and_at_most_one(ctx: ToolContext, bad: float) -> None:
    with pytest.raises(ValueError, match="compact_at"):
        make_agent(ctx, [], compact_at=bad)


def test_keep_turns_must_be_at_least_one(ctx: ToolContext) -> None:
    with pytest.raises(ValueError, match="compact_keep_turns"):
        make_agent(ctx, [], keep_turns=0)
    assert make_agent(ctx, [], keep_turns=1)[0].compact_keep_turns == 1


# --- AC10: pruning ----------------------------------------------------------------------------


# --- AC11: summarizing ------------------------------------------------------------------------


def big_text_history() -> list[Message]:
    return [*text_turn(1, 1000), *text_turn(2, 1000), *text_turn(3, 800)]


async def test_the_summary_request_does_not_use_up_a_step(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("gist")), reply(TextDelta("ok"))], max_steps=1)
    seed(agent, big_text_history())
    events = Events()

    await agent.run_turn("u4", events)

    assert len(provider.requests) == 2
    assert events.of(TurnEnded) == [TurnEnded("done")]
    assert [n for n in events.notices() if "Step limit" in n[1]] == []


# --- AC12: failures ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "summary_reply",
    [
        pytest.param([ToolCallEvent(ToolCall("x", "read", {})), Done(10, 0)], id="only-a-tool-call"),
    ],
)
async def test_a_failed_summary_keeps_the_pruned_history_warns_and_the_turn_goes_on(
    ctx: ToolContext, summary_reply: list[ProviderEvent | Gate]
) -> None:
    agent, provider, recorder = make_agent(ctx, [summary_reply, reply(TextDelta("ok"))])
    history, meta = tool_history([1200, 1200, 1200, 2000], chatter=300)
    seed(agent, history, meta)
    events = Events()

    await agent.run_turn("u5", events)

    assert ("warn", NOT_SUMMARIZED) in events.notices()
    assert events.of(TurnEnded) == [TurnEnded("done")]
    assert len(provider.requests) == 2
    sent = sent_history(provider, 1)
    assert [m.content for m in sent if m.role == "tool"] == [stub(1), stub(2), stub(3), "4" * 2000]
    assert not is_summary(sent, 0)
    [record] = recorder.compactions  # the pruning is still saved
    assert (record.pruned, record.summarized) == (3, 0)
    assert record.history == agent.history[:-1]  # all but the assistant reply


async def test_a_summary_stream_that_raises_is_a_failed_summary_too(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("ok"))])
    seed(agent, big_text_history())

    async def broken(messages: Sequence[Message], tools: Any) -> Any:
        if not tools and messages[0].content.startswith("Summarize"):
            raise RuntimeError("socket closed")
        yield TextDelta("ok")
        yield Done(1, 1)

    provider.stream = broken  # ty: ignore[invalid-assignment]
    events = Events()

    await agent.run_turn("u4", events)

    assert ("warn", NOT_SUMMARIZED) in events.notices()
    assert events.of(TurnEnded) == [TurnEnded("done")]


# --- every request -----------------------------------------------------------------------------


# --- the invariant with a real file -----------------------------------------------------------


# --- write failures ---------------------------------------------------------------------------


async def test_a_compaction_write_failure_and_a_message_failure_make_one_streak(ctx: ToolContext) -> None:
    recorder = MemoryRecorder()
    agent, _, _ = make_agent(ctx, [reply(TextDelta("ok"))], recorder=recorder)
    history, meta = tool_history([1200, 1200, 40])
    seed(agent, history, meta)
    recorder.fail = 2  # the compaction, then the assistant reply
    events = Events()

    await agent.run_turn("u4", events)

    assert len([n for n in events.of(Notice) if n.level == "error"]) == 1
    assert len(recorder.compactions) == 1  # nothing lost: both were written once the disk came back
    assert recorder.messages[-1][0].content == "ok"


# --- one failed summary per turn --------------------------------------------------------------


def summary_requests(provider: ScriptedProvider) -> list[tuple[list[Message], list[str]]]:
    return [r for r in provider.requests if r[0][0].content.startswith("Summarize")]


def call_reply(call_id: str) -> list[ProviderEvent | Gate]:
    """A step that asks for an unknown tool and reports a count that keeps the estimate over the limit."""
    return [ToolCallEvent(ToolCall(call_id, "nope", {})), Done(900, 50)]


# --- the notices and the record ---------------------------------------------------------------


async def test_a_sink_that_raises_on_the_failed_summary_warning_does_not_lose_the_pruned_record(
    ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    root = tmp_path / "sessions"
    agent, _, _ = make_agent(
        ctx, [[ProviderError("down")]], recorder=FileRecorder(workspace_dir(root, workspace), workspace)
    )
    history, meta = tool_history([1200, 1200, 1200, 2000], chatter=300)
    seed(agent, history, meta)

    def sink(event: AgentEvent) -> None:
        if event == Notice("warn", NOT_SUMMARIZED):
            raise RuntimeError("sink broke")

    await agent.run_turn("u5", sink)

    assert [m.content for m in agent.history if m.role == "tool"] == [stub(1), stub(2), stub(3), "4" * 2000]
    store = SessionStore(root, workspace)
    loaded = store.load(store.list()[0].id)
    assert loaded.history == agent.history
    assert loaded.tool_meta == agent.tool_meta


async def test_a_sink_that_raises_does_not_replace_the_cancel_of_a_summary(ctx: ToolContext) -> None:
    gate = Gate()
    agent, _, recorder = make_agent(ctx, [reply(gate, TextDelta("never"))])
    history, meta = tool_history([100, 100, 100, 100])
    seed(agent, history, meta)

    def sink(event: AgentEvent) -> None:
        raise RuntimeError("sink broke")

    running = asyncio.create_task(agent.compact(sink))
    await gate.reached.wait()

    await cancel(running)  # still the cancel, not the sink's error

    assert [(r.pruned, r.summarized) for r in recorder.compactions] == [(2, 0)]
    assert not agent.busy


# --- a summary has to pay for itself ----------------------------------------------------------


# --- the still-full warning, once --------------------------------------------------------------


def over(call_id: str | None = None) -> list[ProviderEvent | Gate]:
    """A step whose reported count is over the window, with a tool call unless it is the last one."""
    calls: list[ProviderEvent | Gate] = [ToolCallEvent(ToolCall(call_id, "nope", {}))] if call_id else [TextDelta("ok")]
    return [*calls, Done(WINDOW, 100)]


async def test_the_still_full_warning_is_shown_once_per_turn_not_at_every_step(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [over("t1"), over("t2"), over(), over()])
    events = Events()

    await agent.run_turn("x" * 4400, events)  # three requests, all over the window

    assert len(provider.requests) == 3
    assert events.notices().count(("warn", STILL_FULL)) == 1

    await agent.run_turn("more", events)  # a new turn warns again

    assert events.notices().count(("warn", STILL_FULL)) == 2


# --- AC13: Agent.compact() --------------------------------------------------------------------


async def test_compact_is_refused_while_a_turn_is_running(ctx: ToolContext) -> None:
    gate = Gate()
    agent, _, _ = make_agent(ctx, [reply(gate, TextDelta("late"))])
    turn = asyncio.create_task(agent.run_turn("hi", Events()))
    await gate.reached.wait()

    with pytest.raises(AgentBusy):
        await agent.compact(Events())

    assert agent.busy  # the refused call did not clear the running turn's busy flag
    await cancel(turn)


# --- AC12: interrupting the summary -----------------------------------------------------------


async def cancel(task: asyncio.Task[None]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def unanswered(history: Sequence[Message]) -> set[str]:
    asked = {c.id for m in history for c in m.tool_calls}
    return asked - {m.tool_call_id for m in history if m.role == "tool" and m.tool_call_id}


async def test_cancelling_a_summary_with_nothing_pruned_records_nothing(ctx: ToolContext) -> None:
    gate = Gate()
    agent, _, recorder = make_agent(ctx, [reply(gate, TextDelta("never"))])
    seed(agent, big_text_history())
    running = asyncio.create_task(agent.compact(Events()))
    await gate.reached.wait()

    await cancel(running)

    assert recorder.compactions == []
    assert agent.history == big_text_history()


async def test_esc_during_the_summary_of_a_turn_ends_it_as_interrupted_with_the_pruned_history(
    ctx: ToolContext,
) -> None:
    gate = Gate()
    agent, provider, recorder = make_agent(ctx, [reply(gate, TextDelta("never")), reply(TextDelta("gist")), reply(TextDelta("ok"))])
    history, meta = tool_history([1200, 1200, 1200, 2000], chatter=300)
    seed(agent, history, meta)
    events = Events()
    turn = asyncio.create_task(agent.run_turn("u5", events))
    await gate.reached.wait()

    await cancel(turn)

    assert events.of(TurnEnded) == [TurnEnded("interrupted")]
    assert [m.content for m in agent.history if m.role == "tool"] == [stub(1), stub(2), stub(3), "4" * 2000]
    assert agent.history[-1] == Message("user", "u5")
    assert unanswered(agent.history) == set()
    assert len(provider.requests) == 1 and len(summary_requests(provider)) == 1  # only the summary request was made
    assert [(r.pruned, r.summarized) for r in recorder.compactions] == [(3, 0)]
    assert not agent.busy


# --- the "compacting" signal for the UI: a summary takes about 10 s on a local model ---------------


class OrderedEvents(Events):
    """Notes how many provider requests had been made when each event arrived."""

    def __init__(self, provider: ScriptedProvider) -> None:
        super().__init__()
        self.provider = provider
        self.at: list[tuple[AgentEvent, int]] = []

    def __call__(self, event: AgentEvent) -> None:
        super().__call__(event)
        self.at.append((event, len(self.provider.requests)))


async def test_nothing_to_summarize_does_not_signal_compacting(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [])
    events = Events()

    await agent.compact(events)

    assert events.of(Compacting) == []
