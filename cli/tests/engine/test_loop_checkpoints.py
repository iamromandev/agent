"""Agent turns take a checkpoint before and after, record both on the undo record, and survive a broken store."""

from __future__ import annotations

import asyncio
import os
import shutil
from datetime import date
from pathlib import Path
from typing import cast

import pytest
from src.engine.approval import Choice, Decision
from src.engine.checkpoints.store import CheckpointStore, Snapshot
from src.engine.events import AgentEvent, Notice
from src.engine.guardrails.setup import Guardrails
from src.engine.loop import Agent
from src.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.tools import default_tools
from src.engine.tools.base import ToolContext

from tests.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider


class Events:
    def __init__(self) -> None:
        self.items: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.items.append(event)

    def notices(self) -> list[tuple[str, str]]:
        return [(e.level, e.text) for e in self.items if isinstance(e, Notice)]


class FakeStore:
    """The part of CheckpointStore the agent uses."""

    def __init__(self, commits: list[str | None], warning: str | None = None, gate: Gate | None = None) -> None:
        self.commits = list(commits)
        self.warning = warning
        self.gate = gate
        self.calls = 0
        self.gate_on_call: int | None = None  # make a gate at this call
        self.gate_made = asyncio.Event()

    async def snapshot(self) -> Snapshot | None:
        self.calls += 1
        if self.gate_on_call == self.calls:
            self.gate = Gate()
            self.gate_made.set()
        if self.gate is not None:
            self.gate.reached.set()
            await self.gate.release.wait()
        commit = self.commits.pop(0)
        return Snapshot(commit, ()) if commit else None

    def take_warning(self) -> str | None:
        warning, self.warning = self.warning, None
        return warning


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.py").write_text("print('hi')\n")
    return root.resolve()


def make_agent(workspace: Path, store: object, turns: int = 1) -> Agent:
    script: list[list[ProviderEvent | Gate]] = [[TextDelta("ok"), Done(10, 5)] for _ in range(turns)]
    return Agent(
        ScriptedProvider(script, context_window=1000),
        default_tools(),
        RecordingApprover([]),
        ToolContext(workspace=workspace, bash_timeout=5.0),
        25,
        today=lambda: date(2026, 11, 12),
        recorder=MemoryRecorder(),
        checkpoints=cast(CheckpointStore, store),
    )


async def test_a_turn_records_a_before_and_an_after_commit(workspace: Path) -> None:
    agent = make_agent(workspace, FakeStore(["c-before", "c-after"]))

    await agent.run_turn("go", Events())

    turn = agent._undo.pop()
    assert turn is not None
    assert (turn.before, turn.after) == ("c-before", "c-after")


async def test_without_a_store_a_turn_has_no_commits(workspace: Path) -> None:
    agent = make_agent(workspace, None)

    await agent.run_turn("go", Events())

    turn = agent._undo.pop()
    assert turn is not None
    assert (turn.before, turn.after) == (None, None)


async def test_a_failed_first_snapshot_skips_the_end_snapshot_and_warns_once(workspace: Path) -> None:
    store = FakeStore([None, None], warning="checkpoints off: boom; /undo restores write/edit changes only")
    agent = make_agent(workspace, store, turns=2)
    events = Events()

    await agent.run_turn("one", events)
    await agent.run_turn("two", events)

    assert store.calls == 2  # one start snapshot per turn; no end snapshot when there was no start
    assert events.notices().count(("warn", "checkpoints off: boom; /undo restores write/edit changes only")) == 1
    turn = agent._undo.pop()
    assert turn is not None
    assert turn.before is None and turn.after is None


async def test_a_warning_is_not_lost_when_showing_it_fails(workspace: Path) -> None:
    text = "checkpoints off: boom; /undo restores write/edit changes only"
    store = FakeStore([None, None], warning=text)
    agent = make_agent(workspace, store, turns=2)
    events = Events()
    failures = [True]

    def flaky(event: AgentEvent) -> None:
        if isinstance(event, Notice) and event.text == text and failures:
            failures.pop()
            raise RuntimeError("the screen is gone")
        events(event)

    await agent.run_turn("one", flaky)
    await agent.run_turn("two", flaky)

    assert events.notices().count(("warn", text)) == 1


async def test_a_failed_end_snapshot_leaves_before_only(workspace: Path) -> None:
    agent = make_agent(workspace, FakeStore(["c-before", None]))

    await agent.run_turn("go", Events())

    turn = agent._undo.pop()
    assert turn is not None
    assert (turn.before, turn.after) == ("c-before", None)


async def test_a_cancel_during_the_first_snapshot_is_safe(workspace: Path) -> None:
    gate = Gate()
    agent = make_agent(workspace, FakeStore(["c1"], gate=gate))
    events = Events()

    task = asyncio.ensure_future(agent.run_turn("go", events))
    await gate.reached.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not agent.busy
    turn = agent._undo.pop()
    assert turn is not None
    assert (turn.before, turn.after) == (None, None)


async def test_a_cancel_during_the_end_snapshot_keeps_before_and_does_not_mask_the_turn(workspace: Path) -> None:
    store = FakeStore(["c-before", "c-after"])
    store.gate_on_call = 2
    agent = make_agent(workspace, store)
    task = asyncio.ensure_future(agent.run_turn("go", Events()))
    await store.gate_made.wait()
    assert store.gate is not None
    await store.gate.reached.wait()
    task.cancel()
    await task  # the turn had finished: the cancel in the end snapshot does not turn it into an error

    assert not agent.busy
    turn = agent._undo.pop()
    assert turn is not None
    assert (turn.before, turn.after) == ("c-before", None)


async def test_a_store_that_raises_does_not_break_the_turn(workspace: Path) -> None:
    class Broken(FakeStore):
        async def snapshot(self) -> Snapshot | None:
            raise RuntimeError("bug")

    agent = make_agent(workspace, Broken([]))
    events = Events()

    await agent.run_turn("go", events)

    assert [t for t in events.notices() if t[0] == "error"] == []
    assert not agent.busy


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_a_real_store_gives_two_commits_that_differ_when_the_turn_changed_files(
    workspace: Path, tmp_path: Path
) -> None:
    store = CheckpointStore(workspace, tmp_path / "data" / "checkpoints", Guardrails.off(workspace), dict(os.environ))
    script: list[list[ProviderEvent | Gate]] = [
        [
            ToolCallEvent(ToolCall("c1", "bash", {"command": "echo more >> a.py && touch c.py"})),
            Done(1, 1),
        ],
        [TextDelta("done"), Done(1, 1)],
    ]
    agent = Agent(
        ScriptedProvider(script, context_window=1000),
        default_tools(),
        RecordingApprover([Decision(Choice.ONCE)]),
        ToolContext(workspace=workspace, bash_timeout=5.0),
        25,
        today=lambda: date(2026, 11, 12),
        recorder=MemoryRecorder(),
        checkpoints=store,
    )

    await agent.run_turn("go", Events())

    turn = agent._undo.pop()
    assert turn is not None
    assert turn.before and turn.after and turn.before != turn.after
    store.close()
