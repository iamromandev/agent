"""generate_image and edit_image refuse while a model warm-up runs (AC7)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from core.engine.approval import Choice, Decision
from core.engine.guardrails.setup import Guardrails
from core.engine.image_pull import REFUSED, ImagePuller
from core.engine.imagegen import EDIT_EXTRA, MfluxGenerator
from core.engine.loop import Agent
from core.engine.mode import Mode
from core.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from core.engine.tools.base import ToolContext
from core.engine.tools.image import EditImageTool, GenerateImageTool
from core.testing.fakes import MemoryRecorder, RecordingApprover, ScriptedProvider
from core.testing.image_fakes import fake_mflux, png

CALLS: dict[str, dict[str, Any]] = {
    "generate_image": {"prompt": "p", "path": "a.png", "width": 256, "height": 256},
    "edit_image": {"prompt": "p", "image": "src.png", "path": "a.png"},
}


def _agent(tmp_path: Path, name: str, busy: bool, mode: Mode = Mode.BUILD) -> tuple[Agent, RecordingApprover]:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "src.png").write_bytes(png(32, 32))
    guard = Guardrails.off(workspace)
    gen = MfluxGenerator(fake_mflux(tmp_path, "g"), "x/y", 5, guard)
    edit = MfluxGenerator(fake_mflux(tmp_path, "e"), "x/y", 5, guard, extra=EDIT_EXTRA)
    puller = ImagePuller({"generate": gen, "edit": edit}, None, guard)
    puller.busy = busy
    tools: Sequence[Any] = [GenerateImageTool(gen, pull=puller), EditImageTool(edit, pull=puller)]
    approver = RecordingApprover([Decision(Choice.ONCE)])
    provider = ScriptedProvider(
        [[ToolCallEvent(ToolCall("c1", name, CALLS[name])), Done(1, 1)], [TextDelta("ok"), Done(1, 1)]], context_window=100_000
    )
    agent = Agent(provider, list(tools), approver, ToolContext(workspace=workspace.resolve(), bash_timeout=5.0), 5, recorder=MemoryRecorder())
    agent.set_mode(mode)
    return agent, approver


@pytest.mark.parametrize("name", ["generate_image", "edit_image"])
async def test_the_tool_is_refused_without_a_dialog_while_a_pull_runs(tmp_path: Path, name: str) -> None:
    agent, approver = _agent(tmp_path, name, busy=True)
    await agent.run_turn("go", lambda event: None)
    assert approver.requests == []
    tool_message = next(m for m in agent.history if m.role == "tool")
    assert tool_message.content == REFUSED


@pytest.mark.parametrize("name", ["generate_image", "edit_image"])
async def test_the_tool_asks_normally_when_no_pull_runs(tmp_path: Path, name: str) -> None:
    agent, approver = _agent(tmp_path, name, busy=False)
    await agent.run_turn("go", lambda event: None)
    assert len(approver.requests) == 1


@pytest.mark.parametrize("name", ["generate_image", "edit_image"])
async def test_plan_mode_refuses_before_the_pull_check(tmp_path: Path, name: str) -> None:
    agent, approver = _agent(tmp_path, name, busy=True, mode=Mode.PLAN)
    await agent.run_turn("go", lambda event: None)
    tool_message = next(m for m in agent.history if m.role == "tool")
    assert tool_message.content != REFUSED
    assert approver.requests == []
