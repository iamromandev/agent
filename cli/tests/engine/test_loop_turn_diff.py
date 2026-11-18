"""`Agent.last_turn_diff`: the turn `/undo` would undo, and nothing after the history is cleared."""

from __future__ import annotations

import os
import shutil
from datetime import date
from pathlib import Path

import pytest
from src.engine.approval import Choice, Decision
from src.engine.checkpoints.store import CheckpointStore
from src.engine.guardrails.setup import Guardrails
from src.engine.loop import Agent, AgentBusy
from src.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.session import NullRecorder
from src.engine.tools import default_tools
from src.engine.tools.base import ToolContext
from src.engine.turn_diff import render_lines

from tests.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root.resolve()


async def paths(agent: Agent) -> list[str]:
    diff = await agent.last_turn_diff()
    assert diff is not None
    return [i.path for i in diff.items]


def bash(command: str, n: int = 0) -> list[ProviderEvent | Gate]:
    return [ToolCallEvent(ToolCall(f"c{n}", "bash", {"command": command})), Done(10, 5)]


def done() -> list[ProviderEvent | Gate]:
    return [TextDelta("ok"), Done(10, 5)]


def make_agent(workspace: Path, tmp_path: Path, script: list[list[ProviderEvent | Gate]], checkpoints: bool = True) -> Agent:
    store = (
        CheckpointStore(workspace, tmp_path / "data", Guardrails.off(workspace), dict(os.environ)) if checkpoints else None
    )
    return Agent(
        ScriptedProvider(script, context_window=1000),
        default_tools(),
        RecordingApprover([Decision(Choice.ONCE)] * 10),
        ToolContext(workspace=workspace, bash_timeout=5.0),
        25,
        today=lambda: date(2026, 11, 12),
        recorder=MemoryRecorder(),
        checkpoints=store,
    )


async def test_there_is_nothing_before_a_turn(workspace: Path, tmp_path: Path) -> None:
    agent = make_agent(workspace, tmp_path, [])

    assert await agent.last_turn_diff() is None


async def test_the_diff_is_the_last_turn_and_reset_clears_it(workspace: Path, tmp_path: Path) -> None:
    agent = make_agent(workspace, tmp_path, [bash("echo hi > a.txt"), done()])
    await agent.run_turn("go", lambda e: None)

    diff = await agent.last_turn_diff()

    assert diff is not None and [(i.status, i.path) for i in diff.items] == [("A", "a.txt")]

    agent.reset(recorder=NullRecorder())
    assert await agent.last_turn_diff() is None


async def test_after_two_turns_and_one_undo_it_is_the_first_turn(workspace: Path, tmp_path: Path) -> None:
    agent = make_agent(
        workspace, tmp_path, [bash("echo 1 > one.txt"), done(), bash("echo 2 > two.txt", 1), done()]
    )
    await agent.run_turn("one", lambda e: None)
    await agent.run_turn("two", lambda e: None)
    assert await paths(agent) == ["two.txt"]

    await agent.undo(lambda e: None)

    assert await paths(agent) == ["one.txt"]


async def test_without_a_store_the_write_log_is_used(workspace: Path, tmp_path: Path) -> None:
    write: list[ProviderEvent | Gate] = [ToolCallEvent(ToolCall("w", "write", {"path": "n.txt", "content": "x\n"})), Done(1, 1)]
    agent = make_agent(workspace, tmp_path, [write, done()], checkpoints=False)
    await agent.run_turn("go", lambda e: None)

    diff = await agent.last_turn_diff()

    assert diff is not None and diff.fallback and [i.path for i in diff.items] == ["n.txt"]
    assert render_lines(await agent.load_diff_file(diff, diff.items[0]))[-1].text == "+x"


async def test_it_is_refused_while_a_turn_runs(workspace: Path, tmp_path: Path) -> None:
    agent = make_agent(workspace, tmp_path, [])
    agent._busy = True

    with pytest.raises(AgentBusy):
        await agent.last_turn_diff()


async def test_a_diff_is_current_only_while_its_turn_is_the_newest(workspace: Path, tmp_path: Path) -> None:
    agent = make_agent(
        workspace, tmp_path, [bash("echo 1 > one.txt"), done(), bash("echo 2 > two.txt", 1), done()]
    )
    await agent.run_turn("one", lambda e: None)
    first = await agent.last_turn_diff()
    assert first is not None and agent.is_current_diff(first)

    await agent.run_turn("two", lambda e: None)
    assert not agent.is_current_diff(first)  # a newer turn exists

    second = await agent.last_turn_diff()
    assert second is not None and agent.is_current_diff(second)
    await agent.undo(lambda e: None)
    assert not agent.is_current_diff(second)  # its turn was undone
    agent.reset(recorder=NullRecorder())
    assert not agent.is_current_diff(first)
