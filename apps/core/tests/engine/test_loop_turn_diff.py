"""`Agent.last_turn_diff`: the turn `/undo` would undo, and nothing after the history is cleared."""

from __future__ import annotations

import os
import shutil
from datetime import date
from pathlib import Path

import pytest
from core.engine.approval import Choice, Decision
from core.engine.checkpoints.store import CheckpointStore
from core.engine.guardrails.setup import Guardrails
from core.engine.loop import Agent, AgentBusy
from core.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from core.engine.tools import default_tools
from core.engine.tools.base import ToolContext
from core.testing.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider

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


async def test_a_custom_agents_write_and_edit_are_listed_with_checkpoints_off(workspace: Path, tmp_path: Path) -> None:
    from pathlib import Path as _P

    from core.engine.agents.loader import AgentDef, Agents
    from core.engine.tools import read_only_tools

    (workspace / "a.py").write_text("print('hi')\n")
    defn = AgentDef("w", "w agent", "You are w.", ("write", "edit"), None, None, "project", _P("/x/w.md"))

    def call(call_id: str, name: str, **args: object) -> list[ProviderEvent | Gate]:
        return [ToolCallEvent(ToolCall(call_id, name, dict(args))), Done(1, 1)]

    script: list[list[ProviderEvent | Gate]] = [
        call("t1", "task", description="d", prompt="p", agent="w"),
        call("w1", "write", path="n.txt", content="made\n"),
        call("e1", "edit", path="a.py", old_string="hi", new_string="bye"),
        [TextDelta("report"), Done(1, 1)],
        done(),
    ]
    agent = Agent(
        ScriptedProvider(script, context_window=100_000),
        default_tools(),
        RecordingApprover([Decision(Choice.ONCE)] * 2),
        ToolContext(workspace=workspace, bash_timeout=5.0),
        25,
        today=lambda: date(2026, 11, 12),
        recorder=MemoryRecorder(),
        task_tools=read_only_tools(None),
        agents=Agents({"w": defn}, ()),
    )
    await agent.run_turn("go", lambda event: None)

    assert sorted(await paths(agent)) == ["a.py", "n.txt"]
