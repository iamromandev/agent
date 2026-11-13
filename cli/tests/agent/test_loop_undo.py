"""Agent.undo: files, history, prompt, session record, and when a turn is not undoable."""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from src.agent.approval import Choice, Decision
from src.agent.events import AgentEvent, Notice, TurnEnded
from src.agent.loop import Agent, AgentBusy
from src.agent.provider.base import Done, Message, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.agent.tools import default_tools
from src.agent.tools.base import ToolContext
from src.agent.undo import NOTHING_TO_UNDO

from tests.fakes import BlockingApprover, Gate, MemoryRecorder, RecordedUndo, RecordingApprover, ScriptedProvider

ONCE = Decision(Choice.ONCE)


@pytest.fixture(autouse=True)
def _posix_sh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", "/bin/sh")


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

    def notices(self) -> list[tuple[str, str]]:
        return [(e.level, e.text) for e in self.items if isinstance(e, Notice)]


def call(name: str, call_id: str = "c1", **arguments: Any) -> ToolCallEvent:
    return ToolCallEvent(ToolCall(call_id, name, arguments))


def reply(*events: ProviderEvent | Gate) -> list[ProviderEvent | Gate]:
    return [*events, Done(10, 5)]


def make_agent(
    ctx: ToolContext,
    script: list[list[ProviderEvent | Gate]],
    decisions: list[Decision] | None = None,
    **options: Any,
) -> tuple[Agent, MemoryRecorder]:
    recorder = MemoryRecorder()
    window = options.pop("window", 1000)
    approver = options.pop("approver", None) or RecordingApprover(decisions)
    agent = Agent(
        ScriptedProvider(script, context_window=window),
        default_tools(),
        approver,
        ctx,
        25,
        today=lambda: date(2026, 11, 12),
        recorder=recorder,
        **options,
    )
    return agent, recorder


EDIT_AND_CREATE = [
    reply(
        call("edit", "c1", path="a.py", old_string="hi", new_string="bye"),
        call("write", "c2", path="new/n.txt", content="made\n"),
    ),
    reply(TextDelta("done")),
]


async def test_undo_restores_files_cuts_history_returns_the_prompt_and_records_it(
    ctx: ToolContext, workspace: Path
) -> None:
    agent, recorder = make_agent(ctx, EDIT_AND_CREATE, [ONCE, ONCE])
    await agent.run_turn("change things", Events())
    assert (workspace / "a.py").read_text() == "print('bye')\n"
    assert (workspace / "new" / "n.txt").exists()

    events = Events()
    prompt = await agent.undo(events)

    assert prompt == "change things"
    assert (workspace / "a.py").read_text() == "print('hi')\n"
    assert not (workspace / "new").exists()
    assert agent.history == []
    assert agent.tool_meta == {}
    assert recorder.undos == [RecordedUndo(0)]
    assert events.notices() == [("info", "Undid the last turn: restored a.py · deleted new/n.txt")]
    assert not agent.busy


async def test_three_edits_of_one_file_in_a_turn_undo_to_the_state_before_the_first(
    ctx: ToolContext, workspace: Path
) -> None:
    script = [
        reply(
            call("edit", "c1", path="a.py", old_string="hi", new_string="one"),
            call("edit", "c2", path="a.py", old_string="one", new_string="two"),
            call("edit", "c3", path="a.py", old_string="two", new_string="three"),
        ),
        reply(TextDelta("done")),
    ]
    agent, _ = make_agent(ctx, script, [ONCE, ONCE, ONCE])
    await agent.run_turn("go", Events())

    await agent.undo(Events())

    assert (workspace / "a.py").read_text() == "print('hi')\n"


async def test_a_denied_write_leaves_nothing_to_list(ctx: ToolContext, workspace: Path) -> None:
    script = [reply(call("write", path="n.txt", content="x")), reply(TextDelta("ok"))]
    agent, _ = make_agent(ctx, script, [Decision(Choice.DENY)])
    await agent.run_turn("write it", Events())

    events = Events()
    prompt = await agent.undo(events)

    assert prompt == "write it"
    assert events.notices() == [("info", "Undid the last turn: no files changed")]


async def test_a_text_only_turn_is_undoable(ctx: ToolContext) -> None:
    agent, _ = make_agent(ctx, [reply(TextDelta("hello"))])
    await agent.run_turn("hi", Events())

    events = Events()
    prompt = await agent.undo(events)

    assert prompt == "hi"
    assert agent.history == []
    assert events.notices() == [("info", "Undid the last turn: no files changed")]


async def test_turns_undo_newest_first_then_nothing_to_undo(ctx: ToolContext) -> None:
    agent, recorder = make_agent(ctx, [reply(TextDelta("1")), reply(TextDelta("2")), reply(TextDelta("3"))])
    for text in ("one", "two", "three"):
        await agent.run_turn(text, Events())

    prompts = [await agent.undo(Events()) for _ in range(3)]
    events = Events()
    fourth = await agent.undo(events)

    assert prompts == ["three", "two", "one"]
    assert fourth is None
    assert events.notices() == [("info", NOTHING_TO_UNDO)]
    assert recorder.undos == [RecordedUndo(4), RecordedUndo(2), RecordedUndo(0)]


async def test_a_resumed_session_keeps_its_earlier_turns(ctx: ToolContext) -> None:
    agent, _ = make_agent(ctx, [reply(TextDelta("new"))])
    agent.reset(
        recorder=MemoryRecorder(),
        history=[Message("user", "old1"), Message("assistant", "r1"), Message("user", "old2"), Message("assistant", "r2")],
    )
    await agent.run_turn("fresh", Events())

    assert await agent.undo(Events()) == "fresh"
    events = Events()
    assert await agent.undo(events) is None

    assert len(agent.history) == 4
    assert events.notices() == [("info", NOTHING_TO_UNDO)]


async def test_two_bash_commands_are_reported_not_undone(ctx: ToolContext) -> None:
    script = [
        reply(call("bash", "c1", command="true"), call("bash", "c2", command="false")),
        reply(TextDelta("done")),
    ]
    agent, _ = make_agent(ctx, script, [ONCE, ONCE])
    await agent.run_turn("run", Events())

    events = Events()
    await agent.undo(events)

    assert events.notices() == [
        ("info", "Undid the last turn: no files changed"),
        ("warn", "That turn ran 2 bash commands; its effects were not undone."),
    ]


async def test_a_denied_bash_is_not_counted(ctx: ToolContext) -> None:
    script = [reply(call("bash", command="true")), reply(TextDelta("done"))]
    agent, _ = make_agent(ctx, script, [Decision(Choice.DENY)])
    await agent.run_turn("run", Events())

    events = Events()
    await agent.undo(events)

    assert events.notices() == [("info", "Undid the last turn: no files changed")]


async def test_the_draft_replaces_the_stored_prompt(ctx: ToolContext) -> None:
    agent, _ = make_agent(ctx, [reply(TextDelta("done"))])
    await agent.run_turn("long expanded instructions", Events(), draft="/init")

    assert await agent.undo(Events()) == "/init"


async def test_undo_while_a_turn_runs_raises_busy_and_changes_nothing(ctx: ToolContext, workspace: Path) -> None:
    approver = BlockingApprover()
    script = [reply(call("write", path="n.txt", content="x")), reply(TextDelta("done"))]
    agent, recorder = make_agent(ctx, script, approver=approver)
    task = asyncio.create_task(agent.run_turn("write", Events()))
    await approver.asked.wait()

    with pytest.raises(AgentBusy):
        await agent.undo(Events())
    approver.release.set()
    await task

    assert (workspace / "n.txt").read_text() == "x"
    assert recorder.undos == []
    assert len(agent.history) == 4
    assert await agent.undo(Events()) == "write"  # the turn was kept undoable


async def test_a_forced_compact_makes_earlier_turns_not_undoable_but_later_ones_are(ctx: ToolContext) -> None:
    script = [reply(TextDelta("1")), reply(TextDelta("2")), reply(TextDelta("sum")), reply(TextDelta("3"))]
    agent, _ = make_agent(ctx, script)
    await agent.run_turn("one", Events())
    await agent.run_turn("two", Events())
    await agent.compact(Events())

    events = Events()
    assert await agent.undo(events) is None
    assert events.notices() == [("info", NOTHING_TO_UNDO)]

    await agent.run_turn("three", Events())
    assert await agent.undo(Events()) == "three"
    assert await agent.undo(Events()) is None


async def test_a_forced_compact_that_changed_nothing_still_clears(ctx: ToolContext) -> None:
    agent, _ = make_agent(ctx, [reply(TextDelta("1"))])
    await agent.run_turn("one", Events())

    await agent.compact(Events())  # one short turn: nothing to prune or cut

    assert await agent.undo(Events()) is None


async def test_automatic_compaction_during_a_turn_makes_that_turn_not_undoable(
    ctx: ToolContext, workspace: Path
) -> None:
    window = 4000
    first: list[ProviderEvent | Gate] = [call("write", path="n.txt", content="x"), Done(3250, 50)]
    agent, _ = make_agent(ctx, [first, reply(TextDelta("done"))], [ONCE], window=window)
    for n, size in enumerate((2000, 2000, 40), start=1):
        agent.history += [
            Message("user", f"u{n}"),
            Message("assistant", "", [ToolCall(f"r{n}", "read", {"path": "a.py"})]),
            Message("tool", f"{n}" * size, tool_call_id=f"r{n}", name="read"),
            Message("assistant", f"a{n}"),
        ]

    await agent.run_turn("u4", Events())

    assert (workspace / "n.txt").exists()
    events = Events()
    assert await agent.undo(events) is None
    assert (workspace / "n.txt").exists()  # not undone
    assert events.notices() == [("info", NOTHING_TO_UNDO)]


async def test_reset_discards_the_undo_records(ctx: ToolContext) -> None:
    agent, _ = make_agent(ctx, [reply(TextDelta("1"))])
    await agent.run_turn("one", Events())

    agent.reset(recorder=MemoryRecorder())

    events = Events()
    assert await agent.undo(events) is None
    assert events.notices() == [("info", NOTHING_TO_UNDO)]


class SlowWrite:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        real = Path.write_bytes

        def write_bytes(path: Path, data: Any) -> int:
            if path.name == name:
                self.started.set()
                assert self.release.wait(5.0), "the test never released the write"
            return real(path, data)

        monkeypatch.setattr(Path, "write_bytes", write_bytes)


async def test_an_interrupted_turn_with_a_landed_write_is_undoable(
    ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slow = SlowWrite(monkeypatch, "a.py")
    edit = call("edit", path="a.py", old_string="hi", new_string="bye")
    agent, _ = make_agent(ctx, [reply(edit)], [ONCE])
    events = Events()
    task = asyncio.create_task(agent.run_turn("edit it", events))
    assert await asyncio.to_thread(slow.started.wait, 5.0)

    task.cancel()
    slow.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events.items[-1] == TurnEnded("interrupted")
    assert (workspace / "a.py").read_text() == "print('bye')\n"

    prompt = await agent.undo(Events())

    assert prompt == "edit it"
    assert (workspace / "a.py").read_text() == "print('hi')\n"
    assert agent.history == []


async def test_a_conflicting_file_is_skipped_and_listed(ctx: ToolContext, workspace: Path) -> None:
    agent, _ = make_agent(ctx, EDIT_AND_CREATE, [ONCE, ONCE])
    await agent.run_turn("change", Events())
    (workspace / "a.py").write_text("by hand\n")

    events = Events()
    await agent.undo(events)

    assert (workspace / "a.py").read_text() == "by hand\n"
    assert events.notices() == [
        ("warn", "Undid the last turn: skipped a.py (changed since the turn) · deleted new/n.txt")
    ]
    assert agent.history == []


async def test_a_big_file_is_still_edited_but_listed_as_too_large(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "big.txt").write_text("x" * 50)
    script = [
        reply(
            call("edit", "c1", path="big.txt", old_string="x" * 50, new_string="y"),
            call("edit", "c2", path="a.py", old_string="hi", new_string="bye"),
        ),
        reply(TextDelta("done")),
    ]
    agent, _ = make_agent(ctx, script, [ONCE, ONCE], undo_file_limit=20)
    await agent.run_turn("edit", Events())
    assert (workspace / "big.txt").read_text() == "y"

    events = Events()
    await agent.undo(events)

    assert (workspace / "a.py").read_text() == "print('hi')\n"
    assert events.notices() == [("warn", "Undid the last turn: skipped big.txt (too large) · restored a.py")]


async def test_a_failing_session_save_still_undoes_and_shows_the_error(ctx: ToolContext) -> None:
    agent, recorder = make_agent(ctx, [reply(TextDelta("done"))])
    await agent.run_turn("hi", Events())
    recorder.fail = 1

    events = Events()
    prompt = await agent.undo(events)

    assert prompt == "hi"
    assert agent.history == []
    assert next(level for level, _ in events.notices()) == "error"


async def test_the_next_request_after_an_undo_carries_only_the_remaining_history(ctx: ToolContext) -> None:
    agent, _ = make_agent(ctx, [reply(TextDelta("1")), reply(TextDelta("2")), reply(TextDelta("3"))])
    await agent.run_turn("one", Events())
    await agent.run_turn("two", Events())
    await agent.undo(Events())

    await agent.run_turn("three", Events())

    provider: Any = agent.provider
    sent = [m.content for m in provider.requests[-1][0][1:]]
    assert sent == ["one", "1", "three"]



async def test_a_write_after_an_automatic_compaction_in_the_same_turn_leaves_nothing_to_undo(
    ctx: ToolContext, workspace: Path
) -> None:
    window = 4000
    first: list[ProviderEvent | Gate] = [call("read", "r9", path="a.py"), Done(3250, 50)]
    second = reply(call("write", "c1", path="n.txt", content="x"))
    agent, _ = make_agent(ctx, [first, second, reply(TextDelta("done"))], [ONCE], window=window)
    for n, size in enumerate((2000, 2000, 40), start=1):
        agent.history += [
            Message("user", f"u{n}"),
            Message("assistant", "", [ToolCall(f"r{n}", "read", {"path": "a.py"})]),
            Message("tool", f"{n}" * size, tool_call_id=f"r{n}", name="read"),
            Message("assistant", f"a{n}"),
        ]

    events = Events()
    await agent.run_turn("u4", events)

    assert (workspace / "n.txt").exists()
    assert not any(level == "error" for level, _ in events.notices())
    undo_events = Events()
    assert await agent.undo(undo_events) is None
    assert (workspace / "n.txt").exists()


async def test_a_cancel_during_undo_still_cuts_the_history_and_records_it(
    ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, recorder = make_agent(ctx, EDIT_AND_CREATE, [ONCE, ONCE])
    await agent.run_turn("change things", Events())
    started, release = threading.Event(), threading.Event()
    real = os.unlink

    def slow_unlink(path: Any) -> None:
        if Path(path).name == "n.txt":
            started.set()
            assert release.wait(5.0)
        real(path)

    monkeypatch.setattr("src.agent.undo.os.unlink", slow_unlink)
    task = asyncio.create_task(agent.undo(Events()))
    assert await asyncio.to_thread(started.wait, 5.0)

    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert agent.history == []
    assert recorder.undos == [RecordedUndo(0)]
    assert not agent.busy
    assert not (workspace / "new" / "n.txt").exists()


async def test_undo_keeps_tool_meta_of_a_reused_call_id_the_remaining_history_still_uses(
    ctx: ToolContext, workspace: Path
) -> None:
    script = [
        reply(call("write", "same", path="x.txt", content="1")),
        reply(TextDelta("a")),
        reply(call("write", "same", path="y.txt", content="2")),
        reply(TextDelta("b")),
    ]
    agent, _ = make_agent(ctx, script, [ONCE, ONCE])
    await agent.run_turn("one", Events())
    await agent.run_turn("two", Events())
    assert "same" in agent.tool_meta

    await agent.undo(Events())

    assert "same" in agent.tool_meta


async def test_undo_drops_tool_meta_no_remaining_message_uses(ctx: ToolContext) -> None:
    script = [reply(call("write", "only", path="x.txt", content="1")), reply(TextDelta("a"))]
    agent, _ = make_agent(ctx, script, [ONCE])
    await agent.run_turn("one", Events())
    await agent.undo(Events())
    assert agent.tool_meta == {}


async def test_an_unexpected_restore_error_still_cuts_the_history_records_it_and_reports(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, recorder = make_agent(ctx, [reply(TextDelta("a"))])
    await agent.run_turn("one", Events())

    def boom(*args: Any) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr("src.agent.loop.restore_turn", boom)
    events = Events()
    prompt = await agent.undo(events)

    assert prompt == "one"
    assert agent.history == []
    assert recorder.undos == [RecordedUndo(0)]
    assert any(level == "error" and "kaboom" in text for level, text in events.notices())
    assert not agent.busy
