"""Compaction in the loop (R12-R18, AC10-AC15): when it runs, what it keeps, how it fails, what it records."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from src.agent.compaction import (
    OMITTED_NOTE,
    SUMMARIZER_INSTRUCTIONS,
    SUMMARY_PREFIX,
    SUMMARY_TRUNCATED,
    Reported,
    estimate_tokens,
    is_summary,
    message_tokens,
)
from src.agent.events import AgentEvent, Compacting, Notice, TurnEnded, UsageUpdate
from src.agent.loop import Agent, AgentBusy
from src.agent.provider.base import Done, Message, ProviderError, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.agent.session import FileRecorder, SessionStore, ToolMeta, workspace_dir
from src.agent.tools import default_tools
from src.agent.tools.base import Tool, ToolContext

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


def test_the_defaults_are_the_documented_ones(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [])
    plain = Agent(ScriptedProvider([]), [], RecordingApprover(), ctx, 5)

    assert (plain.compact_at, plain.compact_keep_turns) == (0.8, 2)
    assert (agent.compact_at, agent.compact_keep_turns) == (0.8, 2)


@pytest.mark.parametrize("bad", [0, -0.1, 1.5])
def test_compact_at_must_be_above_zero_and_at_most_one(ctx: ToolContext, bad: float) -> None:
    with pytest.raises(ValueError, match="compact_at"):
        make_agent(ctx, [], compact_at=bad)


def test_compact_at_one_is_accepted(ctx: ToolContext) -> None:
    assert make_agent(ctx, [], compact_at=1)[0].compact_at == 1


def test_keep_turns_must_be_at_least_one(ctx: ToolContext) -> None:
    with pytest.raises(ValueError, match="compact_keep_turns"):
        make_agent(ctx, [], keep_turns=0)
    assert make_agent(ctx, [], keep_turns=1)[0].compact_keep_turns == 1


# --- AC10: pruning ----------------------------------------------------------------------------


async def test_old_tool_results_are_pruned_before_the_request_and_no_summary_is_asked(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("ok"))])
    history, meta = tool_history([1200, 1200, 40])
    seed(agent, history, meta)
    assert 800 <= agent.estimate() < 1000  # "about 900" of AC10

    await agent.run_turn("u4", Events())

    assert len(provider.requests) == 1
    sent = sent_history(provider, 0)
    assert sent[2].content == stub(1)
    assert sent[6].content == stub(2)
    assert sent[8:12] == history[8:12]  # turn 3 untouched
    assert sent[8].content == "u3" and sent[-1].content == "u4"


async def test_the_pruned_history_is_the_agents_history_and_kept_turns_are_byte_identical(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(TextDelta("ok"))])
    history, meta = tool_history([1200, 1200, 40])
    seed(agent, history, meta)
    kept = list(history[8:])

    await agent.run_turn("u4", Events())

    assert agent.history[2].content == stub(1)
    assert agent.history[8:12] == kept
    assert all(a is b for a, b in zip(agent.history[8:12], kept, strict=True))


async def test_nothing_happens_below_the_threshold(ctx: ToolContext) -> None:
    agent, provider, recorder = make_agent(ctx, [reply(TextDelta("ok"))])
    history, meta = tool_history([40, 40, 40])
    seed(agent, history, meta)
    assert agent.estimate() < 800

    events = Events()
    await agent.run_turn("u4", events)

    assert len(provider.requests) == 1
    assert sent_history(provider, 0)[:12] == history
    assert recorder.compactions == []
    assert events.notices() == []
    assert events.of(UsageUpdate) == [UsageUpdate(15, WINDOW)]  # only the model's own report


async def test_a_compaction_notice_and_an_estimated_usage_update_follow_pruning(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(TextDelta("ok"))])
    history, meta = tool_history([1200, 1200, 40])
    seed(agent, history, meta)
    before = agent.estimate() + message_tokens(Message("user", "u4"))  # the message about to be added
    events = Events()

    await agent.run_turn("u4", events)

    [notice] = [n for n in events.of(Notice) if n.text.startswith("Context compacted")]
    after = events.of(UsageUpdate)[0].used
    assert notice == Notice("info", f"Context compacted: pruned 2 tool outputs (ctx ~{before} → ~{after})")
    assert events.of(UsageUpdate)[0] == UsageUpdate(after, WINDOW, estimated=True)
    assert after < 800 <= before


async def test_pruning_is_recorded_with_the_full_compacted_history(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [reply(TextDelta("ok"))])
    history, meta = tool_history([1200, 1200, 40])
    seed(agent, history, meta)
    events = Events()

    await agent.run_turn("u4", events)

    [record] = recorder.compactions
    assert record.history == [*agent.history[:13]]  # taken before the assistant reply was added
    assert record.history[-1] == Message("user", "u4")
    assert (record.pruned, record.summarized) == (2, 0)
    assert record.tool_meta == agent.tool_meta
    notice = next(n for n in events.of(Notice) if n.text.startswith("Context compacted"))
    assert f"(ctx ~{record.before} → ~{record.after})" in notice.text


async def test_the_original_messages_in_the_recorder_are_not_changed_by_pruning(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [reply(TextDelta("ok"))])
    history, meta = tool_history([1200, 1200, 40])
    seed(agent, history, meta)

    await agent.run_turn("u4", Events())

    originals = [m for m, _ in recorder.messages if m.role == "tool"]
    assert [m.content[:3] for m in originals] == ["111", "222", "333"]


async def test_pruning_again_straight_away_changes_nothing(ctx: ToolContext) -> None:
    agent, provider, recorder = make_agent(ctx, [reply(TextDelta("ok"))])  # nothing left for a summary request
    history, meta = tool_history([1200, 1200, 40])
    seed(agent, history, meta)
    await agent.run_turn("u4", Events())
    requests, recorded = len(provider.requests), len(recorder.compactions)
    assert [m.content for m in agent.history if m.role == "tool"][:2] == [stub(1), stub(2)]

    again = Events()
    await agent._compact(again, force=True)  # a forced compaction really runs: it asks for a summary

    assert len(provider.requests) == requests + 1  # so the check below is not a skipped one
    assert ("info", "Context compacted: pruned 0 tool outputs") in [
        (level, text.split(" (ctx")[0]) for level, text in again.notices()
    ]
    assert len(recorder.compactions) == recorded  # nothing was pruned again, so nothing is recorded


# --- AC11: summarizing ------------------------------------------------------------------------


def big_text_history() -> list[Message]:
    return [*text_turn(1, 1000), *text_turn(2, 1000), *text_turn(3, 800)]


async def test_a_history_still_too_big_after_pruning_is_summarized_with_one_extra_request(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("The "), TextDelta("gist")), reply(TextDelta("ok"))])
    seed(agent, big_text_history())

    await agent.run_turn("u4", Events())

    assert len(provider.requests) == 2
    summary_request, tools = provider.requests[0]
    assert tools == []
    assert summary_request[0] == Message("system", SUMMARIZER_INSTRUCTIONS)
    assert summary_request[1].role == "user"
    assert summary_request[1].content.startswith("USER: u1\n\nASSISTANT: a1")
    assert "USER: u3" not in summary_request[1].content  # only what is older than the kept turns


async def test_the_next_request_holds_the_summary_then_the_kept_turns(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("The "), TextDelta("gist")), reply(TextDelta("ok"))])
    history = big_text_history()
    seed(agent, history)

    await agent.run_turn("u4", Events())

    sent = sent_history(provider, 1)
    assert sent == [Message("user", SUMMARY_PREFIX + "The gist"), *history[4:], Message("user", "u4")]
    assert is_summary(sent, 0)
    assert agent.history[:-1] == sent


async def test_the_summary_notice_counts_the_summarized_messages(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [reply(TextDelta("gist")), reply(TextDelta("ok"))])
    seed(agent, big_text_history())
    events = Events()

    await agent.run_turn("u4", events)

    [record] = recorder.compactions
    assert (record.pruned, record.summarized) == (0, 4)
    assert (
        "info",
        f"Context compacted: pruned 0 tool outputs, summarized 4 messages (ctx ~{record.before} → ~{record.after})",
    ) in events.notices()
    assert events.of(UsageUpdate)[0] == UsageUpdate(record.after, WINDOW, estimated=True)
    assert record.after < record.before


async def test_pruning_and_summarizing_in_one_compaction_are_both_counted(ctx: ToolContext) -> None:
    agent, provider, recorder = make_agent(ctx, [reply(TextDelta("gist")), reply(TextDelta("ok"))])
    history, meta = tool_history([1200, 1200, 1200, 2000], chatter=300)
    seed(agent, history, meta)

    await agent.run_turn("u5", Events())

    assert (recorder.compactions[0].pruned, recorder.compactions[0].summarized) == (3, 12)
    # the summary was written from the pruned messages: stubs, not the 1200 characters
    assert stub(1) in provider.requests[0][0][1].content
    assert "1" * 100 not in provider.requests[0][0][1].content


async def test_the_summary_request_does_not_use_up_a_step(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("gist")), reply(TextDelta("ok"))], max_steps=1)
    seed(agent, big_text_history())
    events = Events()

    await agent.run_turn("u4", events)

    assert len(provider.requests) == 2
    assert events.of(TurnEnded) == [TurnEnded("done")]
    assert [n for n in events.notices() if "Step limit" in n[1]] == []


async def test_after_a_compaction_the_estimate_is_a_guess_not_a_stale_or_summary_count(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [[TextDelta("gist"), Done(5000, 5000)]])
    seed(agent, big_text_history())
    agent._reported = Reported(900, len(agent.history))  # the model's count for the history as it was

    await agent._compact(Events(), force=True)

    guess = estimate_tokens(agent._system(agent.mode), agent.history, None)
    assert agent.estimate() == guess < 1000


async def test_the_compaction_record_equals_the_history_after_a_summary(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [reply(TextDelta("gist")), reply(TextDelta("ok"))])
    seed(agent, big_text_history())

    await agent.run_turn("u4", Events())

    [record] = recorder.compactions
    assert record.history == agent.history[:-1]
    assert record.history[0] == Message("user", SUMMARY_PREFIX + "gist")


async def test_an_earlier_summary_is_folded_into_the_next_one(ctx: ToolContext) -> None:
    agent, provider, recorder = make_agent(ctx, [reply(TextDelta("gist two")), reply(TextDelta("ok"))])
    seed(agent, [Message("user", SUMMARY_PREFIX + "OLD GIST"), *big_text_history()])

    await agent.run_turn("u4", Events())

    assert "OLD GIST" in provider.requests[0][0][1].content
    assert sent_history(provider, 1)[0] == Message("user", SUMMARY_PREFIX + "gist two")
    assert recorder.compactions[0].summarized == 4  # the old summary is not one of the messages summarized


async def test_an_oversized_older_part_is_cut_to_the_summary_budget(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("gist")), reply(TextDelta("ok"))])
    seed(agent, [*text_turn(1, 6000), *text_turn(2, 100), *text_turn(3, 6000), *text_turn(4)])  # 600 tokens of budget

    await agent.run_turn("u5", Events())

    summary_text = provider.requests[0][0][1].content
    assert provider.requests[0][0][0].content == SUMMARIZER_INSTRUCTIONS + OMITTED_NOTE
    assert "USER: u1" not in summary_text


# --- AC12: failures ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "summary_reply",
    [
        pytest.param([ProviderError("model unreachable")], id="provider-error"),
        pytest.param([Done(10, 0)], id="no-text"),
        pytest.param([TextDelta(""), TextDelta(""), Done(10, 0)], id="empty-text"),
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


async def test_a_failed_summary_with_nothing_to_prune_records_nothing(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [[ProviderError("down")], reply(TextDelta("ok"))])
    seed(agent, big_text_history())
    events = Events()

    await agent.run_turn("u4", events)

    assert recorder.compactions == []
    assert [n for n in events.notices() if n[1].startswith("Context compacted")] == []
    assert ("warn", NOT_SUMMARIZED) in events.notices()


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


async def test_the_still_full_warning_shows_when_compaction_cannot_get_under_the_window(ctx: ToolContext) -> None:
    agent, provider, recorder = make_agent(ctx, [reply(TextDelta("ok"))])
    events = Events()

    await agent.run_turn("x" * 4400, events)  # one huge turn: nothing is older, nothing can be cut

    assert ("warn", STILL_FULL) in events.notices()
    assert len(provider.requests) == 1  # and the request still goes out
    assert events.of(TurnEnded) == [TurnEnded("done")]
    assert recorder.compactions == []
    assert [n for n in events.notices() if n[1].startswith("Context compacted")] == []


async def test_the_still_full_warning_also_follows_a_compaction_that_was_not_enough(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [reply(TextDelta("gist")), reply(TextDelta("ok"))])
    seed(agent, big_text_history())
    events = Events()

    await agent.run_turn("u4" + "x" * 4400, events)  # the kept turn alone is over the window

    assert ("warn", STILL_FULL) in events.notices()
    assert recorder.compactions[0].after >= WINDOW


async def test_no_warning_when_the_estimate_ends_under_the_window(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(TextDelta("gist")), reply(TextDelta("ok"))])
    seed(agent, big_text_history())
    events = Events()

    await agent.run_turn("u4", events)

    assert ("warn", STILL_FULL) not in events.notices()


# --- every request -----------------------------------------------------------------------------


async def test_compaction_is_checked_before_every_request_of_a_turn(ctx: ToolContext) -> None:
    window = 4000  # the default tool specs are part of the estimate
    first: list[ProviderEvent | Gate] = [ToolCallEvent(ToolCall("t1", "read", {"path": "a.py"})), Done(3250, 50)]
    agent, provider, recorder = make_agent(
        ctx, [first, reply(TextDelta("done"))], window=window, tools=default_tools()
    )
    history, meta = tool_history([2000, 2000, 40])
    seed(agent, history, meta)
    assert agent.estimate() < 0.8 * window  # nothing to do before the first request

    events = Events()
    await agent.run_turn("u4", events)

    assert [m.content for m in sent_history(provider, 0) if m.role == "tool"][:2] == ["1" * 2000, "2" * 2000]
    second = sent_history(provider, 1)
    assert [m.content for m in second if m.role == "tool"][:2] == [stub(1), stub(2)]
    assert len(recorder.compactions) == 1
    assert second[-1].role == "tool" and second[-1].tool_call_id == "t1"  # compaction came before the request, after the tool
    assert events.of(TurnEnded) == [TurnEnded("done")]


async def test_a_pending_tool_call_is_never_cut_by_compaction(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(
        ctx,
        [[ToolCallEvent(ToolCall("t1", "read", {"path": "a.py"})), Done(900, 50)], reply(TextDelta("done"))],
        tools=default_tools(),
        window=2000,
    )
    seed(agent, [*text_turn(1, 3000), *text_turn(2, 3000), *text_turn(3)])

    await agent.run_turn("u4", Events())

    for index in range(len(provider.requests)):
        sent = sent_history(provider, index)
        answered = {m.tool_call_id for m in sent if m.role == "tool"}
        asked = {c.id for m in sent for c in m.tool_calls}
        assert asked <= answered


# --- the invariant with a real file -----------------------------------------------------------


async def test_what_loads_from_the_session_file_after_a_compaction_equals_the_agent(
    ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    root = tmp_path / "sessions"
    recorder = FileRecorder(workspace_dir(root, workspace), workspace)
    agent, _, _ = make_agent(ctx, [reply(TextDelta("gist")), reply(TextDelta("ok")), reply(TextDelta("more"))], recorder=recorder)
    history, meta = tool_history([1200, 1200, 1200, 2000], chatter=300)
    seed(agent, history, meta)

    await agent.run_turn("u5", Events())
    store = SessionStore(root, workspace)
    [info] = store.list()
    loaded = store.load(info.id)

    assert agent.history[0].content.startswith(SUMMARY_PREFIX)
    assert loaded.history == agent.history
    assert loaded.tool_meta == agent.tool_meta

    await agent.run_turn("u6", Events())  # and it keeps being true as the session goes on
    again = store.load(info.id)
    assert again.history == agent.history
    assert again.tool_meta == agent.tool_meta


async def test_the_replayed_notice_says_what_the_live_notice_said(
    ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    root = tmp_path / "sessions"
    agent, _, _ = make_agent(
        ctx,
        [reply(TextDelta("gist")), reply(TextDelta("ok"))],
        recorder=FileRecorder(workspace_dir(root, workspace), workspace),
    )
    history, meta = tool_history([1200, 1200, 1200, 2000], chatter=300)
    seed(agent, history, meta)
    events = Events()

    await agent.run_turn("u5", events)

    store = SessionStore(root, workspace)
    loaded = store.load(store.list()[0].id)
    [live] = [n.text for n in events.of(Notice) if n.text.startswith("Context compacted")]
    assert [i.text for i in loaded.items if i.kind == "notice"] == [live]


async def test_a_pruning_only_compaction_loads_back_equal_to_the_agent(
    ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    root = tmp_path / "sessions"
    agent, _, _ = make_agent(
        ctx, [reply(TextDelta("ok"))], recorder=FileRecorder(workspace_dir(root, workspace), workspace)
    )
    history, meta = tool_history([1200, 1200, 40])
    seed(agent, history, meta)

    await agent.run_turn("u4", Events())

    store = SessionStore(root, workspace)
    loaded = store.load(store.list()[0].id)
    assert loaded.history == agent.history
    assert loaded.tool_meta == agent.tool_meta


# --- write failures ---------------------------------------------------------------------------


async def test_a_failing_compaction_write_gives_one_error_notice_and_the_turn_completes(ctx: ToolContext) -> None:
    recorder = MemoryRecorder()
    agent, _, _ = make_agent(ctx, [reply(TextDelta("ok"))], recorder=recorder)
    history, meta = tool_history([1200, 1200, 40])
    seed(agent, history, meta)
    recorder.fail = 1  # the compaction is the next write
    events = Events()

    await agent.run_turn("u4", events)

    errors = [n for n in events.of(Notice) if n.level == "error"]
    assert len(errors) == 1 and errors[0].text.startswith("Could not save the session")
    assert events.of(TurnEnded) == [TurnEnded("done")]
    assert agent.history[2].content == stub(1)  # the history is swapped anyway
    assert [r for r in recorder.records if not isinstance(r, tuple)] != []  # and the record was retried and written
    assert recorder.compactions[0].history[2].content == stub(1)


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


async def test_a_failed_summary_is_tried_once_per_turn_not_at_every_step(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(
        ctx,
        [
            [Done(10, 0)],  # the summary: no text
            call_reply("t1"),
            call_reply("t2"),
            [TextDelta("done"), Done(900, 50)],
            [Done(10, 0)],  # the next turn's summary: no text either
            [TextDelta("again"), Done(10, 5)],
        ],
    )
    seed(agent, big_text_history())
    events = Events()

    await agent.run_turn("u4", events)

    assert len(provider.requests) == 4  # one summary request and three steps
    assert len(summary_requests(provider)) == 1
    assert events.notices().count(("warn", NOT_SUMMARIZED)) == 1
    assert events.of(TurnEnded) == [TurnEnded("done")]

    await agent.run_turn("u5", events)  # the next turn tries again, once

    assert len(summary_requests(provider)) == 2
    assert events.notices().count(("warn", NOT_SUMMARIZED)) == 2


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


async def test_a_cancelled_summary_still_shows_the_compaction_notice_the_file_replays(
    ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    gate = Gate()
    root = tmp_path / "sessions"
    agent, provider, _ = make_agent(
        ctx,
        [reply(gate, TextDelta("never"))],
        recorder=FileRecorder(workspace_dir(root, workspace), workspace),
    )
    history, meta = tool_history([1200, 1200, 1200, 2000], chatter=300)
    seed(agent, history, meta)
    events = Events()
    turn = asyncio.create_task(agent.run_turn("u5", events))
    await gate.reached.wait()

    await cancel(turn)

    assert len(summary_requests(provider)) == 1  # it was the summary that was cancelled
    store = SessionStore(root, workspace)
    replayed = [i.text for i in store.load(store.list()[0].id).items if i.kind == "notice"]
    live = [n.text for n in events.of(Notice) if n.text.startswith("Context compacted")]
    assert len(live) == 1 and "pruned 3 tool outputs" in live[0]
    assert live == replayed
    assert events.of(UsageUpdate) == [UsageUpdate(agent.estimate(), WINDOW, estimated=True)]
    assert events.of(TurnEnded) == [TurnEnded("interrupted")]


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


async def test_an_automatic_summary_is_skipped_when_the_older_part_is_too_small_to_pay_for_it(
    ctx: ToolContext,
) -> None:
    agent, provider, recorder = make_agent(ctx, [reply(TextDelta("ok"))])
    seed(agent, [*text_turn(1), *text_turn(2), *text_turn(3, 2800)])  # a big kept turn, a tiny older part
    assert 800 <= agent.estimate() < 1000
    assert sum(message_tokens(m) for m in agent.history[:4]) < 0.15 * WINDOW
    events = Events()

    await agent.run_turn("u4", events)

    assert len(provider.requests) == 1  # no summary request
    assert recorder.compactions == []
    assert events.notices() == []


async def test_compact_still_summarizes_a_tiny_older_part(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("gist"))])
    seed(agent, [*text_turn(1), *text_turn(2), *text_turn(3)])
    assert sum(message_tokens(m) for m in agent.history[:2]) < 0.15 * WINDOW

    await agent.compact(Events())

    assert len(provider.requests) == 1
    assert agent.history[0] == Message("user", SUMMARY_PREFIX + "gist")


async def test_an_oversized_summary_is_cut_at_a_word_and_marked(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(TextDelta("word " * 1000)), reply(TextDelta("ok"))])
    seed(agent, big_text_history())  # the cap for a window of 1000 is 800 characters

    await agent.run_turn("u4", Events())

    content = agent.history[0].content
    assert content.startswith(SUMMARY_PREFIX) and content.endswith(SUMMARY_TRUNCATED)
    body = content[len(SUMMARY_PREFIX) : -len(SUMMARY_TRUNCATED)]
    assert 0 < len(body) <= 800
    assert set(body.split()) == {"word"}


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


async def test_compact_warns_again_when_the_context_is_still_full(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [over()])
    events = Events()
    await agent.run_turn("x" * 4400, events)
    assert events.notices().count(("warn", STILL_FULL)) == 1

    await agent.compact(events)

    assert events.notices().count(("warn", STILL_FULL)) == 2


# --- AC13: Agent.compact() --------------------------------------------------------------------


async def test_compact_prunes_and_summarizes_even_below_the_threshold(ctx: ToolContext) -> None:
    agent, provider, recorder = make_agent(ctx, [reply(TextDelta("gist"))])
    history, meta = tool_history([100, 100, 100, 100])
    seed(agent, history, meta)
    before = agent.estimate()
    assert before < 800
    events = Events()

    await agent.compact(events)

    after = agent.estimate()
    assert after < before
    assert len(provider.requests) == 1 and provider.requests[0][1] == []
    assert agent.history[0] == Message("user", SUMMARY_PREFIX + "gist")
    assert agent.history[1:] == history[8:]  # the two kept turns, unchanged (turns 3 and 4: not pruned either)
    assert events.of(Notice) == [
        Notice("info", f"Context compacted: pruned 2 tool outputs, summarized 8 messages (ctx ~{before} → ~{after})")
    ]
    assert events.of(UsageUpdate) == [UsageUpdate(after, WINDOW, estimated=True)]
    [record] = recorder.compactions
    assert record.history == agent.history and (record.pruned, record.summarized) == (2, 8)


async def test_compact_on_a_history_with_nothing_to_cut_makes_no_request_and_no_record(ctx: ToolContext) -> None:
    agent, provider, recorder = make_agent(ctx, [])
    seed(agent, [*text_turn(1), *text_turn(2)])
    events = Events()

    await agent.compact(events)

    estimate = agent.estimate()
    assert provider.requests == []
    assert recorder.compactions == []
    assert events.notices() == [("info", f"Context compacted: pruned 0 tool outputs (ctx ~{estimate} → ~{estimate})")]


async def test_compact_with_a_failing_summary_keeps_the_pruned_history_and_returns(ctx: ToolContext) -> None:
    agent, _, recorder = make_agent(ctx, [[ProviderError("down")]])
    history, meta = tool_history([100, 100, 100, 100])
    seed(agent, history, meta)
    events = Events()

    await agent.compact(events)

    assert ("warn", NOT_SUMMARIZED) in events.notices()
    assert [m.content for m in agent.history if m.role == "tool"] == [stub(1), stub(2), "3" * 100, "4" * 100]
    assert not agent.busy
    assert (recorder.compactions[0].pruned, recorder.compactions[0].summarized) == (2, 0)


async def test_compact_is_refused_while_a_turn_is_running(ctx: ToolContext) -> None:
    gate = Gate()
    agent, _, _ = make_agent(ctx, [reply(gate, TextDelta("late"))])
    turn = asyncio.create_task(agent.run_turn("hi", Events()))
    await gate.reached.wait()

    with pytest.raises(AgentBusy):
        await agent.compact(Events())

    assert agent.busy  # the refused call did not clear the running turn's busy flag
    await cancel(turn)


async def test_a_turn_is_refused_while_compact_runs_and_busy_is_cleared_afterwards(ctx: ToolContext) -> None:
    gate = Gate()
    agent, _, _ = make_agent(ctx, [reply(gate, TextDelta("gist"))])
    history, meta = tool_history([100, 100, 100, 100])
    seed(agent, history, meta)
    running = asyncio.create_task(agent.compact(Events()))
    await gate.reached.wait()

    assert agent.busy
    with pytest.raises(AgentBusy):
        await agent.run_turn("hi", Events())

    gate.release.set()
    await running
    assert not agent.busy


# --- AC12: interrupting the summary -----------------------------------------------------------


async def cancel(task: asyncio.Task[None]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def unanswered(history: Sequence[Message]) -> set[str]:
    asked = {c.id for m in history for c in m.tool_calls}
    return asked - {m.tool_call_id for m in history if m.role == "tool" and m.tool_call_id}


async def test_cancelling_compact_during_the_summary_keeps_the_pruned_history_and_records_it(ctx: ToolContext) -> None:
    gate = Gate()
    agent, _, recorder = make_agent(ctx, [reply(gate, TextDelta("never"))])
    history, meta = tool_history([100, 100, 100, 100])
    seed(agent, history, meta)
    events = Events()
    running = asyncio.create_task(agent.compact(events))
    await gate.reached.wait()

    await cancel(running)  # the cancel propagates

    assert [m.content for m in agent.history if m.role == "tool"] == [stub(1), stub(2), "3" * 100, "4" * 100]
    assert not is_summary(agent.history, 0)
    assert unanswered(agent.history) == set()
    [record] = recorder.compactions  # saved before the cancel went on
    assert record.history == agent.history and (record.pruned, record.summarized) == (2, 0)
    assert events.notices() == [
        ("info", f"Context compacted: pruned 2 tool outputs (ctx ~{record.before} \u2192 ~{record.after})"),
        ("warn", "Compaction interrupted"),
    ]
    assert not agent.busy


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


async def test_a_turn_after_an_interrupted_summary_works(ctx: ToolContext) -> None:
    gate = Gate()
    agent, provider, _ = make_agent(ctx, [reply(gate, TextDelta("never")), reply(TextDelta("ok"))])
    history, meta = tool_history([1200, 1200, 1200, 2000], chatter=300)
    seed(agent, history, meta)
    turn = asyncio.create_task(agent.run_turn("u5", Events()))
    await gate.reached.wait()
    await cancel(turn)

    events = Events()
    await agent.run_turn("again", events)

    assert events.of(TurnEnded) == [TurnEnded("done")]
    assert agent.history[-1] == Message("assistant", "ok")
    assert unanswered(agent.history) == set()
    # the 4th turn is now older than the kept ones, so pruning alone was enough this time
    assert [m.content for m in sent_history(provider, 1) if m.role == "tool"] == [stub(1), stub(2), stub(3), stub(4)]
    assert len(provider.requests) == 2


async def test_what_loads_from_the_file_after_an_interrupted_summary_equals_the_agent(
    ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    gate = Gate()
    root = tmp_path / "sessions"
    agent, _, _ = make_agent(
        ctx,
        [reply(gate, TextDelta("never"))],
        recorder=FileRecorder(workspace_dir(root, workspace), workspace),
    )
    history, meta = tool_history([1200, 1200, 1200, 2000], chatter=300)
    seed(agent, history, meta)
    turn = asyncio.create_task(agent.run_turn("u5", Events()))
    await gate.reached.wait()

    await cancel(turn)

    store = SessionStore(root, workspace)
    loaded = store.load(store.list()[0].id)
    assert loaded.history == agent.history
    assert loaded.tool_meta == agent.tool_meta


async def test_the_summary_stream_is_closed_when_the_summary_is_given_up(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("ok"))])
    seed(agent, big_text_history())
    closed: list[bool] = []

    async def stream(messages: Sequence[Message], tools: Any) -> Any:
        try:
            if not tools and messages[0].content.startswith("Summarize"):
                yield ProviderError("down")
                yield TextDelta("never read")
            else:
                yield TextDelta("ok")
                yield Done(1, 1)
        finally:
            closed.append(True)

    provider.stream = stream  # ty: ignore[invalid-assignment]

    await agent.compact(Events())

    assert closed == [True]


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


async def test_the_summary_request_is_announced_before_it_starts_and_ended_before_the_notices(
    ctx: ToolContext,
) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("The gist")), reply(TextDelta("ok"))])
    seed(agent, big_text_history())
    events = OrderedEvents(provider)

    await agent.run_turn("u4", events)

    kinds = [type(e).__name__ for e, _ in events.at]
    start, end = kinds.index("Compacting"), kinds.index("Compacting", kinds.index("Compacting") + 1)
    assert [e for e, _ in events.at if isinstance(e, Compacting)] == [Compacting(True), Compacting(False)]
    assert events.at[start][1] == 0  # no request yet
    assert events.at[end][1] == 1  # the summary request has been made
    assert start < end < kinds.index("Notice")  # the compaction notice comes after the status ended


async def test_a_failed_summary_still_ends_the_compacting_signal(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(ProviderError("down")), reply(TextDelta("ok"))])
    seed(agent, big_text_history())
    events = Events()

    await agent.run_turn("u4", events)

    assert events.of(Compacting) == [Compacting(True), Compacting(False)]


async def test_a_cancelled_summary_ends_the_compacting_signal_before_the_cancel_goes_on(ctx: ToolContext) -> None:
    gate = Gate()
    agent, _, _ = make_agent(ctx, [reply(gate, TextDelta("never"))])
    seed(agent, big_text_history())
    events = Events()
    running = asyncio.create_task(agent.compact(events))
    await gate.reached.wait()

    await cancel(running)

    assert events.of(Compacting) == [Compacting(True), Compacting(False)]


async def test_a_sink_that_raises_on_the_end_signal_does_not_replace_the_cancel(ctx: ToolContext) -> None:
    gate = Gate()
    agent, _, _ = make_agent(ctx, [reply(gate, TextDelta("never"))])
    seed(agent, big_text_history())

    def sink(event: AgentEvent) -> None:
        if event == Compacting(False):
            raise RuntimeError("sink broke")

    running = asyncio.create_task(agent.compact(sink))
    await gate.reached.wait()

    await cancel(running)  # still a CancelledError, not RuntimeError


async def test_pruning_alone_does_not_signal_compacting(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("ok"))])
    history, meta = tool_history([1200, 1200, 40])
    seed(agent, history, meta)
    events = Events()

    await agent.run_turn("u4", events)

    assert len(provider.requests) == 1  # pruned, nothing summarized
    assert events.of(Compacting) == []


async def test_nothing_to_summarize_does_not_signal_compacting(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [])
    events = Events()

    await agent.compact(events)

    assert events.of(Compacting) == []
