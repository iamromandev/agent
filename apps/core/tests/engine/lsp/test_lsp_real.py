"""NFR4: the real `ty` and `ruff` next to the interpreter, through `edit` in the loop. Skipped where a binary is missing."""

from __future__ import annotations

import os
import re
import sys
from datetime import date
from pathlib import Path

import pytest
from core.engine.approval import Choice, Decision
from core.engine.guardrails.setup import prepare as prepare_guard
from core.engine.loop import Agent
from core.engine.lsp.config import LspSetup, ServerSpec
from core.engine.lsp.hub import LspHub
from core.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from core.engine.tools import default_tools
from core.engine.tools.base import ToolContext
from core.testing.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider


def binary(name: str) -> Path:
    path = Path(sys.executable).parent / name
    if not path.exists():
        pytest.skip(f"{name} is not installed next to the interpreter")
    return path


def edit(call_id: str, old: str, new: str) -> list[ProviderEvent]:
    args = {"path": "a.py", "old_string": old, "new_string": new}
    return [ToolCallEvent(ToolCall(call_id, "edit", args)), Done(1, 1)]


async def run_edits(tmp_path: Path, spec: ServerSpec, source: str, edits: list[tuple[str, str]], *, warnings: bool = False) -> list[str]:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.py").write_text(source)
    hub = LspHub(LspSetup(True, warnings, (spec,), ()), ws.resolve(), dict(os.environ), wait=10, start_wait=15, max_lines=20)
    script: list[list[ProviderEvent | Gate]] = []
    for n, (old, new) in enumerate(edits):
        script.append([*edit(f"c{n}", old, new)])
        script.append([TextDelta("ok"), Done(1, 1)])
    ctx = ToolContext(ws.resolve(), 5.0, guard=prepare_guard(ws.resolve(), None, None, {}, home=tmp_path / "home"))
    agent = Agent(
        ScriptedProvider(script, context_window=100_000),
        default_tools(),
        RecordingApprover([Decision(Choice.ONCE)] * (len(edits) + 2)),
        ctx,
        25,
        today=lambda: date(2026, 11, 14),
        recorder=MemoryRecorder(),
        lsp=hub,
    )
    try:
        for _ in edits:
            await agent.run_turn("go", lambda _event: None)
    finally:
        await hub.close()
    return [m.content for m in agent.history if m.role == "tool"]


async def test_real_ty_reports_a_type_error_then_nothing_once_fixed(tmp_path: Path) -> None:
    spec = ServerSpec("ty", str(binary("ty")), ("server",), frozenset({".py"}), {}, "global")
    broken, fixed = await run_edits(tmp_path, spec, "x = 1\n", [("x = 1", 'x: int = "a"'), ('x: int = "a"', "x: int = 1")])
    errors = [line for line in broken.splitlines() if re.match(r"a\.py:\d+:\d+ error: ", line)]
    assert len(errors) == 1 and errors[0].endswith("(invalid-assignment)")
    assert "[diagnostics" not in fixed


async def test_real_ruff_warnings_appear_only_when_asked_for(tmp_path: Path) -> None:
    spec = ServerSpec("ruff", str(binary("ruff")), ("server",), frozenset({".py"}), {}, "global")
    change = [("x = 1", "import os\nx = 1")]
    (tmp_path / "quiet").mkdir()
    (tmp_path / "loud").mkdir()
    (quiet,) = await run_edits(tmp_path / "quiet", spec, "x = 1\n", change)
    (loud,) = await run_edits(tmp_path / "loud", spec, "x = 1\n", change, warnings=True)
    assert "[diagnostics" not in quiet
    assert re.search(r"a\.py:1:\d+ warning: .*F401", loud)
