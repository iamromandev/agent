"""Agent turns take a checkpoint before and after, record both on the undo record, and survive a broken store."""

from __future__ import annotations

import asyncio
import os
import shutil
from datetime import date
from pathlib import Path
from typing import cast

import pytest
from core.engine.approval import Choice, Decision
from core.engine.checkpoints.git import GitError
from core.engine.checkpoints.store import CheckpointStore, DiffEntry, Snapshot
from core.engine.events import AgentEvent, Notice
from core.engine.guardrails.setup import Guardrails
from core.engine.loop import Agent
from core.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from core.engine.tools import default_tools
from core.engine.tools.base import ToolContext
from core.testing.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider


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


# --- /undo through the checkpoints -------------------------------------------------------------------------

OUTSIDE = "Effects outside the workspace and running processes are not undone."


def bash_script(*commands: str) -> list[list[ProviderEvent | Gate]]:
    script: list[list[ProviderEvent | Gate]] = []
    for i, command in enumerate(commands):
        script.append([ToolCallEvent(ToolCall(f"c{i}", "bash", {"command": command})), Done(1, 1)])
        script.append([TextDelta("done"), Done(1, 1)])
    return script


def real_agent(
    workspace: Path, tmp_path: Path, *commands: str, script: list[list[ProviderEvent | Gate]] | None = None
) -> tuple[Agent, CheckpointStore]:
    store = CheckpointStore(workspace, tmp_path / "data" / "checkpoints", Guardrails.off(workspace), dict(os.environ))
    script = script if script is not None else bash_script(*commands)
    agent = Agent(
        ScriptedProvider(script, context_window=1000),
        default_tools(),
        RecordingApprover([Decision(Choice.ONCE) for step in script if any(isinstance(e, ToolCallEvent) for e in step)]),
        ToolContext(workspace=workspace, bash_timeout=10.0),
        25,
        today=lambda: date(2026, 11, 12),
        recorder=MemoryRecorder(),
        checkpoints=store,
    )
    return agent, store


# --- what undo must not delete or double count -------------------------------------------------------------


def make_large(path: Path, size: int = 20 * 1024 * 1024) -> None:
    with open(path, "wb") as f:
        f.truncate(size)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_a_change_to_the_git_exclude_file_is_recorded_on_the_turn(workspace: Path, tmp_path: Path) -> None:
    (workspace / ".git" / "info").mkdir(parents=True)
    (workspace / ".git" / "info" / "exclude").write_text("hidden.txt\n")
    agent, store = real_agent(workspace, tmp_path, "echo more >> .git/info/exclude")
    await agent.run_turn("go", Events())

    turn = agent._undo.pop()

    assert turn is not None
    assert turn.exclude_changed is True and turn.before_time is not None
    store.close()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_a_large_file_that_existed_and_the_turn_changed_is_named_as_skipped(workspace: Path, tmp_path: Path) -> None:
    make_large(workspace / "big.bin")
    agent, store = real_agent(workspace, tmp_path, "echo x >> big.bin && echo y > small.txt")
    events = Events()
    await agent.run_turn("go", events)

    await agent.undo(events)

    assert not (workspace / "small.txt").exists()
    assert events.notices()[-2] == ("warn", "Undid the last turn: 0 restored, 1 deleted, 1 skipped: big.bin")
    store.close()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_a_large_file_that_existed_and_the_turn_deleted_is_named_as_skipped(workspace: Path, tmp_path: Path) -> None:
    make_large(workspace / "big.bin")
    agent, store = real_agent(workspace, tmp_path, "rm big.bin")
    events = Events()
    await agent.run_turn("go", events)

    await agent.undo(events)

    assert events.notices()[-2] == ("warn", "Undid the last turn: 0 restored, 0 deleted, 1 skipped: big.bin")
    store.close()


# --- files the checkpoints leave out, changed by write/edit ------------------------------------------------


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_undo_restores_an_ignored_file_that_write_changed(workspace: Path, tmp_path: Path) -> None:
    (workspace / ".gitignore").write_text("build/\n")
    (workspace / "build").mkdir()
    (workspace / "build" / "out.txt").write_text("old")
    script: list[list[ProviderEvent | Gate]] = [
        [ToolCallEvent(ToolCall("c0", "write", {"path": "build/out.txt", "content": "new"})), Done(1, 1)],
        [ToolCallEvent(ToolCall("c1", "bash", {"command": "echo y > small.txt"})), Done(1, 1)],
        [TextDelta("done"), Done(1, 1)],
    ]
    agent, store = real_agent(workspace, tmp_path, script=script)
    events = Events()
    await agent.run_turn("go", events)
    assert (workspace / "build" / "out.txt").read_text() == "new"

    await agent.undo(events)

    assert (workspace / "build" / "out.txt").read_text() == "old"
    assert not (workspace / "small.txt").exists()
    assert events.notices()[-2] == ("info", "Undid the last turn: 1 restored, 1 deleted")
    store.close()


# --- when the checkpoint restore cannot start --------------------------------------------------------------


class RestoreStore(FakeStore):
    """A store whose diff and blob reads are scripted."""

    def __init__(self, commits: list[str | None]) -> None:
        super().__init__(commits)
        self.entries: list[DiffEntry] = []
        self.read_error: Exception | None = None
        self.diff_gate: Gate | None = None
        self.diff_cancelled = False

    async def diff(self, before: str, after: str) -> list[DiffEntry]:
        if self.diff_gate is not None:
            self.diff_gate.reached.set()
            try:
                await self.diff_gate.release.wait()
            except asyncio.CancelledError:
                self.diff_cancelled = True
                raise
        return self.entries

    async def read_blobs(self, shas: list[str], **_: object) -> dict[str, bytes]:
        if self.read_error is not None:
            raise self.read_error
        return {}


def write_agent(workspace: Path, store: object) -> Agent:
    script: list[list[ProviderEvent | Gate]] = [
        [ToolCallEvent(ToolCall("c0", "write", {"path": "n.txt", "content": "hi"})), Done(1, 1)],
        [TextDelta("done"), Done(1, 1)],
    ]
    return Agent(
        ScriptedProvider(script, context_window=1000),
        default_tools(),
        RecordingApprover([Decision(Choice.ONCE)]),
        ToolContext(workspace=workspace, bash_timeout=5.0),
        25,
        today=lambda: date(2026, 11, 12),
        recorder=MemoryRecorder(),
        checkpoints=cast(CheckpointStore, store),
    )


async def test_the_fallback_reason_is_escaped_and_cut(workspace: Path) -> None:
    store = RestoreStore(["c-before", "c-after"])
    store.entries = [DiffEntry("a.py", 0o100644, "a" * 40, 0o100644, "b" * 40)]
    store.read_error = GitError("\x1b[31m" + "x" * 500)
    agent = write_agent(workspace, store)
    events = Events()
    await agent.run_turn("go", events)

    await agent.undo(events)

    [text] = [t for level, t in events.notices() if t.startswith("Checkpoint restore failed")]
    assert "\x1b" not in text and "\\x1b[31m" in text
    assert len(text) < 300


async def test_esc_during_the_checkpoint_read_stops_git_and_still_undoes_the_files(workspace: Path) -> None:
    store = RestoreStore(["c-before", "c-after"])
    store.diff_gate = Gate()
    agent = write_agent(workspace, store)
    events = Events()
    await agent.run_turn("go", events)

    task = asyncio.ensure_future(agent.undo(events))
    await store.diff_gate.reached.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)

    assert store.diff_cancelled  # git was told to stop, not waited for
    assert not agent.busy
    assert agent.history == []  # the turn is cut, as for any undo
    assert not (workspace / "n.txt").exists()  # and its files are put back from the file log


# --- a custom subagent's bash is part of the parent turn -----------------------------------------------------


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_ac19_undo_removes_a_file_a_custom_agents_bash_created(workspace: Path, tmp_path: Path) -> None:
    from core.engine.agents.loader import AgentDef, Agents
    from core.engine.tools import read_only_tools

    store = CheckpointStore(workspace, tmp_path / "data" / "checkpoints", Guardrails.off(workspace), dict(os.environ))
    defn = AgentDef("w", "w agent", "You are w.", ("bash",), None, None, "project", Path("/x/w.md"))
    script: list[list[ProviderEvent | Gate]] = [
        [ToolCallEvent(ToolCall("t1", "task", {"description": "d", "prompt": "p", "agent": "w"})), Done(1, 1)],
        [ToolCallEvent(ToolCall("b1", "bash", {"command": "echo hi > out.txt"})), Done(1, 1)],
        [TextDelta("made it"), Done(1, 1)],
        [TextDelta("done"), Done(1, 1)],
    ]
    agent = Agent(
        ScriptedProvider(script, context_window=100_000),
        default_tools(),
        RecordingApprover([Decision(Choice.ONCE)]),
        ToolContext(workspace=workspace, bash_timeout=10.0),
        25,
        today=lambda: date(2026, 11, 12),
        recorder=MemoryRecorder(),
        checkpoints=store,
        task_tools=read_only_tools(None),
        agents=Agents({"w": defn}, ()),
    )
    await agent.run_turn("go", Events())
    assert (workspace / "out.txt").read_text() == "hi\n"

    await agent.undo(Events())

    assert not (workspace / "out.txt").exists()
    store.close()
