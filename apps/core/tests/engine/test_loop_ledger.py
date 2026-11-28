"""`Agent` with a shared `TurnLedger` and checkpoint store (two sessions of one daemon): what `/undo` leaves alone."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from core.engine.approval import Choice, Decision
from core.engine.checkpoints.git import GitError
from core.engine.checkpoints.store import CheckpointStore, DiffEntry
from core.engine.events import AgentEvent, Notice
from core.engine.guardrails.setup import Guardrails
from core.engine.ledger import TurnLedger
from core.engine.loop import Agent
from core.engine.provider.base import Done, Message, ProviderError, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from core.engine.session import ToolMeta
from core.engine.tools import default_tools
from core.engine.tools.base import ToolContext
from core.testing.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider
from core.testing.helpers import wait_until

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

Script = list[list[ProviderEvent | Gate]]
OTHER = "(changed by another session)"


class Events:
    def __init__(self) -> None:
        self.items: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.items.append(event)

    def text(self) -> str:
        return " ".join(e.text for e in self.items if isinstance(e, Notice))


def write(n: str, path: str, body: str) -> ToolCallEvent:
    return ToolCallEvent(ToolCall(f"w{n}", "write", {"path": path, "content": body}))


def bash(n: str, command: str) -> ToolCallEvent:
    return ToolCallEvent(ToolCall(f"b{n}", "bash", {"command": command}))


def done() -> list[ProviderEvent | Gate]:
    return [TextDelta("ok"), Done(10, 5)]


class World:
    def __init__(self, tmp_path: Path) -> None:
        self.workspace = (tmp_path / "ws").resolve()
        self.workspace.mkdir()
        self.store = CheckpointStore(self.workspace, tmp_path / "data", Guardrails.off(self.workspace), dict(os.environ))
        self.ledger = TurnLedger()

    def agent(self, owner: str, script: Script, window: int = 100_000) -> Agent:
        return Agent(
            ScriptedProvider(script, context_window=window),
            default_tools(),
            RecordingApprover([Decision(Choice.ONCE)] * 30),
            ToolContext(workspace=self.workspace, bash_timeout=5.0),
            25,
            today=lambda: date(2026, 11, 12),
            recorder=MemoryRecorder(),
            checkpoints=self.store,
            ledger=self.ledger,
            owner=owner,
        )

    def read(self, name: str) -> str:
        return (self.workspace / name).read_text()


async def overlap(world: World) -> tuple[Agent, Agent]:
    """A writes a.py and s.py and is held open at its second request; meanwhile B writes b.py and s.py and ends. A is
    then let finish."""
    gate = Gate()
    a = world.agent("A", [[write("1", "a.py", "A"), write("2", "s.py", "A"), Done(1, 1)], [gate, *done()]])
    b = world.agent("B", [[write("3", "b.py", "B"), write("4", "s.py", "B"), Done(1, 1)], done()])
    running = asyncio.ensure_future(a.run_turn("a", Events()))
    await asyncio.wait_for(gate.reached.wait(), 5)
    await b.run_turn("b", Events())
    gate.release.set()
    await running
    return a, b


async def reset_for_new(agent: Agent) -> None:
    agent.reset(recorder=MemoryRecorder())


async def reset_for_switch(agent: Agent) -> None:
    agent.reset(recorder=MemoryRecorder(), history=[Message("user", "old"), Message("assistant", "reply")])


@pytest.mark.parametrize("retire", [reset_for_new, reset_for_switch], ids=["new", "switch"])
async def test_undo_still_skips_the_path_after_the_other_session_started_a_new_session_or_switched(
    tmp_path: Path, retire: Callable[[Agent], Awaitable[None]]
) -> None:
    # Bug caught: `/new` or a switch dropping A's ledger entries, so B's undo reverted s.py, which A also wrote.
    world = World(tmp_path)
    a, b = await overlap(world)
    await retire(a)
    events = Events()

    await b.undo(events)

    assert not (world.workspace / "b.py").exists() and world.read("a.py") == "A" and world.read("s.py") == "B"
    assert f"s.py {OTHER}" in events.text() and "1 deleted" in events.text()


async def test_undo_skips_a_path_the_other_session_wrote_after_its_own_undo_log_was_cleared_mid_turn(
    tmp_path: Path,
) -> None:
    # Bug caught: auto-compaction mid-turn clearing A's undo log dropped its ledger entry (so A's later writes were
    # unknown to B's undo) and left it without an `after` snapshot (so its bash/write changes had no source).
    world = World(tmp_path)
    gate = Gate()
    history = [
        m
        for n in range(1, 5)
        for m in (
            Message("user", f"u{n}"),
            Message("assistant", "", [ToolCall(f"c{n}", "read", {"path": f"{n}.py"})]),
            Message("tool", f"{n}" * (2000 if n == 4 else 1200), tool_call_id=f"c{n}", name="read"),
            Message("assistant", f"a{n}" + "." * 300),
        )
    ]
    a = world.agent(
        "A", [[ProviderError("down")], [gate, write("1", "s.py", "A"), Done(1, 1)], done()], window=1000
    )  # the first request is the summary of the compaction, which fails: the pruning is kept
    for m in history:
        a._record(m, ToolMeta(f"read {m.tool_call_id}", "1 line") if m.role == "tool" else None)
    b = world.agent("B", [[write("2", "b.py", "B"), write("3", "s.py", "B"), Done(1, 1)], done()])
    running = asyncio.ensure_future(a.run_turn("a", Events()))
    await asyncio.wait_for(gate.reached.wait(), 5)
    assert a.undo_log.peek() is None and a.undo_log.current is None  # compaction cleared it mid-turn
    await b.run_turn("b", Events())
    gate.release.set()
    await running
    events = Events()

    await b.undo(events)

    assert not (world.workspace / "b.py").exists() and world.read("s.py") == "A"
    assert f"s.py {OTHER}" in events.text()


async def test_undo_skips_a_path_the_other_session_edited_with_bash_on_top_of_the_undone_turns_write(
    tmp_path: Path,
) -> None:
    # Bug caught: every path in the undone turn's own file log dropped from the other turn's diff, so the other
    # session's bash edit of a file this turn wrote was treated as the undone turn's.
    world = World(tmp_path)
    gate = Gate()
    a = world.agent("A", [[gate, bash("1", "echo U > s.py"), Done(1, 1)], done()])
    b = world.agent("B", [[write("2", "s.py", "B"), Done(1, 1)], done()])
    running = asyncio.ensure_future(a.run_turn("a", Events()))
    await asyncio.wait_for(gate.reached.wait(), 5)
    await b.run_turn("b", Events())
    gate.release.set()
    await running
    events = Events()

    await b.undo(events)

    assert world.read("s.py") == "U\n" and f"s.py {OTHER}" in events.text()


async def test_a_file_the_undone_turns_own_bash_made_that_shows_in_the_other_diff_is_skipped_as_maybe_theirs(
    tmp_path: Path,
) -> None:
    # Bug caught: naming another session as the cause of a change that may be the undone turn's own bash.
    world = World(tmp_path)
    gate = Gate()
    a = world.agent("A", [[gate, *done()]])
    b = world.agent("B", [[bash("1", "echo T > t.txt"), Done(1, 1)], done()])
    running = asyncio.ensure_future(a.run_turn("a", Events()))
    await asyncio.wait_for(gate.reached.wait(), 5)
    await b.run_turn("b", Events())
    gate.release.set()
    await running
    events = Events()

    await b.undo(events)

    assert world.read("t.txt") == "T\n"
    assert "t.txt (may have been changed by another session)" in events.text()


async def test_a_diff_that_fails_leaves_every_file_of_the_undone_turn_alone_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug caught: the failure swallowed, the undo going on as if no other session had changed anything.
    world = World(tmp_path)
    gate = Gate()
    a = world.agent("A", [[write("1", "s.py", "A"), Done(1, 1)], [gate, *done()]])
    b = world.agent("B", [[write("2", "b.py", "B"), write("3", "s.py", "B"), Done(1, 1)], done()])
    running = asyncio.ensure_future(a.run_turn("a", Events()))
    await asyncio.wait_for(gate.reached.wait(), 5)
    await b.run_turn("b", Events())
    a_before = next(e.before for e in world.ledger.entries if e.owner == "A")
    real = world.store.diff

    async def diff(before: str, after: str) -> list[DiffEntry]:
        if before == a_before:
            raise GitError("boom")
        return await real(before, after)

    monkeypatch.setattr(world.store, "diff", diff)
    events = Events()

    await b.undo(events)

    assert world.read("b.py") == "B" and world.read("s.py") == "B"  # nothing was touched
    text = events.text()
    assert "Could not tell what other sessions changed" in text and "boom" in text
    assert "b.py (could not check other sessions)" in text and f"s.py {OTHER}" in text  # A's log still names s.py
    gate.release.set()
    await running


async def test_esc_while_the_undo_waits_for_the_snapshot_lock_changes_nothing_and_leaves_the_lock_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug caught: the ledger's fresh snapshot not interruptible: Esc waited out the lock and then a 5 s snapshot.
    world = World(tmp_path)
    gate = Gate()
    a = world.agent("A", [[write("1", "s.py", "A"), Done(1, 1)], [gate, *done()]])
    b = world.agent("B", [[write("2", "b.py", "B"), write("3", "s.py", "B"), Done(1, 1)], done()])
    running = asyncio.ensure_future(a.run_turn("a", Events()))
    await asyncio.wait_for(gate.reached.wait(), 5)
    await b.run_turn("b", Events())
    waiting = asyncio.Event()
    real = world.store.snapshot

    async def spy() -> Any:
        waiting.set()
        return await real()

    monkeypatch.setattr(world.store, "snapshot", spy)
    turn = b.undo_log.peek()

    async with world.store._lock:  # another snapshot is in progress
        undoing = asyncio.ensure_future(b.undo(Events()))
        await asyncio.wait_for(waiting.wait(), 5)
        undoing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(undoing, 5)

    assert world.read("b.py") == "B" and world.read("s.py") == "B"
    assert b.undo_log.peek() is turn and b.history  # still undoable: the cancel changed nothing
    monkeypatch.setattr(world.store, "snapshot", real)
    gate.release.set()
    await running
    events = Events()
    await b.undo(events)
    assert not (world.workspace / "b.py").exists() and f"s.py {OTHER}" in events.text()


async def test_two_turns_that_run_at_once_are_each_undone_without_touching_the_other_sides_files(
    tmp_path: Path,
) -> None:
    # Bug caught: a corrupted shadow repo, or an undo that reverts the other session's files (or the shared one),
    # when both turns' snapshots and writes really overlap, checked on both sides.
    rounds = 6
    world = World(tmp_path)
    gates_a, gates_b = [Gate() for _ in range(rounds)], [Gate() for _ in range(rounds)]
    script_a: Script = [
        [gates_a[k], write(f"a{k}", f"a{k}.py", "A"), write(f"as{k}", "s.py", f"A{k}"), Done(1, 1)] for k in range(rounds)
    ]
    script_b: Script = [
        [gates_b[k], write(f"b{k}", f"b{k}.py", "B"), write(f"bs{k}", "s.py", f"B{k}"), Done(1, 1)] for k in range(rounds)
    ]
    a = world.agent("A", [*script_a, done()])
    b = world.agent("B", [*script_b, done()])
    both = asyncio.gather(a.run_turn("a", Events()), b.run_turn("b", Events()))  # no one waits for the other
    for k in range(rounds - 1):
        await asyncio.wait_for(asyncio.gather(gates_a[k].reached.wait(), gates_b[k].reached.wait()), 20)
        gates_a[k].release.set()
        gates_b[k].release.set()
    final = rounds - 1  # the last write of s.py is B's, whatever the scheduling
    await asyncio.wait_for(asyncio.gather(gates_a[final].reached.wait(), gates_b[final].reached.wait()), 20)
    gates_a[final].release.set()
    await wait_until(lambda: world.read("s.py") == f"A{final}")
    gates_b[final].release.set()
    await asyncio.wait_for(both, 20)
    assert world.read("s.py") == f"B{rounds - 1}"

    undo_a, undo_b = Events(), Events()
    await a.undo(undo_a)
    mine_a = sorted(p.name for p in world.workspace.glob("a*.py"))
    await b.undo(undo_b)

    assert mine_a == []  # A's own files are gone, B's are still there until B's undo
    assert [p.name for p in world.workspace.iterdir()] == ["s.py"]  # B's own files are gone too
    assert world.read("s.py") == f"B{rounds - 1}"  # nobody's undo touched the shared file
    assert f"s.py {OTHER}" in undo_a.text() and f"s.py {OTHER}" in undo_b.text()
    shadow = world.store.folder
    done_git = subprocess.run(
        ["git", "--git-dir", str(shadow), "fsck", "--strict"], capture_output=True, text=True, env={"PATH": os.environ["PATH"]}
    )
    assert done_git.returncode == 0, done_git.stderr
    assert world.store.enabled, world.store.off_reason
