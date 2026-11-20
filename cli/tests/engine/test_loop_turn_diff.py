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
from src.engine.tools import default_tools
from src.engine.tools.base import ToolContext

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


async def test_it_is_refused_while_a_turn_runs(workspace: Path, tmp_path: Path) -> None:
    agent = make_agent(workspace, tmp_path, [])
    agent._busy = True

    with pytest.raises(AgentBusy):
        await agent.last_turn_diff()
