"""Agent.undo: files, history, prompt, session record, and when a turn is not undoable."""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from src.engine.approval import Choice, Decision
from src.engine.events import AgentEvent, Notice
from src.engine.loop import Agent, AgentBusy
from src.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.tools import default_tools
from src.engine.tools.base import ToolContext
from src.engine.undo import NOTHING_TO_UNDO

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

    monkeypatch.setattr("src.engine.undo.os.unlink", slow_unlink)
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


async def test_an_unexpected_restore_error_still_cuts_the_history_records_it_and_reports(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, recorder = make_agent(ctx, [reply(TextDelta("a"))])
    await agent.run_turn("one", Events())

    def boom(*args: Any) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr("src.engine.loop.restore_turn", boom)
    events = Events()
    prompt = await agent.undo(events)

    assert prompt == "one"
    assert agent.history == []
    assert recorder.undos == [RecordedUndo(0)]
    assert any(level == "error" and "kaboom" in text for level, text in events.notices())
    assert not agent.busy
