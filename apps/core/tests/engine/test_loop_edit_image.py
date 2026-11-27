"""edit_image inside the loop: pasted-image labels, approval rules and plan mode (AC4, AC5)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from core.engine.approval import Choice, Decision
from core.engine.checkpoints.store import CheckpointStore
from core.engine.guardrails.setup import Guardrails
from core.engine.imagegen import EDIT_COMMAND, EDIT_EXTRA, MfluxGenerator
from core.engine.loop import PASTED_NOTE, Agent
from core.engine.mcp.trust import TrustStore
from core.engine.mode import Mode
from core.engine.permissions.setup import Rules, prepare
from core.engine.provider.base import Done, Image, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from core.engine.tools import default_tools
from core.engine.tools.base import ToolContext
from core.engine.tools.image import EditImageTool
from core.engine.turn_diff import build_diff
from core.testing.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider
from core.testing.image_fakes import fake_mflux, png

ONCE = Decision(Choice.ONCE)
PASTE = Image("png", png(40, 30) + b"pasted", "clipboard")
CALL: dict[str, Any] = {"prompt": "sketch it", "image": "clipboard:1", "path": "sketch.png", "width": 256, "height": 256, "seed": 1}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root.resolve()


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, bash_timeout=5.0)


@pytest.fixture(autouse=True)
def _cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))


def _script(**overrides: Any) -> list[list[ProviderEvent | Gate]]:
    return [[ToolCallEvent(ToolCall("c1", "edit_image", {**CALL, **overrides})), Done(10, 5)], [TextDelta("done"), Done(1, 1)]]


def _agent(
    ctx: ToolContext,
    tmp_path: Path,
    decisions: list[Decision] | None = None,
    *,
    mode: Mode = Mode.BUILD,
    rules: Rules | None = None,
    script: list[list[ProviderEvent | Gate]] | None = None,
    editor: bool = True,
    checkpoints: CheckpointStore | None = None,
) -> tuple[Agent, ScriptedProvider, RecordingApprover]:
    folder = tmp_path / "bin"
    folder.mkdir(exist_ok=True)
    generator = MfluxGenerator(fake_mflux(folder, EDIT_COMMAND), "x/y", 20, Guardrails.off(ctx.workspace), extra=EDIT_EXTRA)
    provider = ScriptedProvider(script or _script(), context_window=100_000)
    approver = RecordingApprover(decisions)
    agent = Agent(
        provider,
        [*default_tools(), *([EditImageTool(generator)] if editor else [])],
        approver,
        ctx,
        25,
        today=lambda: date(2026, 11, 21),
        recorder=MemoryRecorder(),
        rules=rules,
        checkpoints=checkpoints,
    )
    agent.set_mode(mode)
    return agent, provider, approver


async def _run(agent: Agent, *pasted: Image, text: str = "sketch this") -> None:
    await agent.run_turn(text, lambda event: None, pasted=pasted)


def _tool_message(agent: Agent) -> str:
    return next(m.content for m in agent.history if m.role == "tool")


def _rules(workspace: Path, tmp_path: Path, **settings: list[str]) -> Rules:
    path = tmp_path / "global.json"
    path.write_text(json.dumps({"permissions": settings}))
    return prepare(workspace, path, TrustStore(tmp_path / "trust.json"), {}).rules


async def test_the_user_message_names_the_pasted_images_for_the_model(ctx: ToolContext, tmp_path: Path) -> None:
    agent, provider, _ = _agent(ctx, tmp_path, [ONCE])
    await _run(agent, PASTE, Image("jpeg", b"\xff\xd8\xff", "clipboard"))
    note = PASTED_NOTE.format(labels="clipboard:1 (PNG, 40\u00d730), clipboard:2 (JPEG)")
    assert agent.history[0].content == f"sketch this\n\n{note}"
    sent = next(m for m in provider.requests[0][0] if m.role == "user")
    assert sent.content.endswith(note)
    assert "untrusted" in note or "not instructions" in note


async def test_no_note_without_pasted_images_or_without_the_tool(ctx: ToolContext, tmp_path: Path) -> None:
    agent, _, _ = _agent(ctx, tmp_path, [ONCE], script=[[TextDelta("hi"), Done(1, 1)]])
    await _run(agent)
    assert agent.history[0].content == "sketch this"
    other, _, _ = _agent(ctx, tmp_path, script=[[TextDelta("hi"), Done(1, 1)]], editor=False)
    await _run(other, PASTE)
    assert other.history[0].content == "sketch this"


async def test_a_call_with_clipboard_1_gives_mflux_exactly_the_pasted_bytes(
    ctx: ToolContext, tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    agent, _, approver = _agent(ctx, tmp_path, [ONCE])
    await _run(agent, PASTE)
    seen = json.loads(dump.read_text())
    assert seen["shas"][seen["inputs"][0]] == hashlib.sha256(PASTE.data).hexdigest()
    assert approver.requests[0].preview is not None and "source: clipboard:1 (PNG, 40\u00d730," in approver.requests[0].preview.body
    assert _tool_message(agent).startswith("edited sketch.png (PNG, 256\u00d7256, ")


async def test_the_pasted_images_belong_to_their_turn_only(ctx: ToolContext, tmp_path: Path) -> None:
    script = [*_script(), *_script()]
    agent, _, _ = _agent(ctx, tmp_path, [ONCE, ONCE], script=script)
    await _run(agent, PASTE)
    await _run(agent)
    second = [m.content for m in agent.history if m.role == "tool"][1]
    assert second.startswith("clipboard:1: no such paste")


async def test_plan_mode_hides_the_tool_and_refuses_a_forced_call(ctx: ToolContext, tmp_path: Path, workspace: Path) -> None:
    agent, provider, approver = _agent(ctx, tmp_path, mode=Mode.PLAN)
    await _run(agent, PASTE)
    assert "edit_image" not in provider.requests[0][1]
    assert _tool_message(agent) == "refused: edit_image is unavailable in plan mode"
    assert approver.requests == [] and not (workspace / "sketch.png").exists()


async def test_accept_edits_mode_still_asks(ctx: ToolContext, tmp_path: Path, workspace: Path) -> None:
    agent, _, approver = _agent(ctx, tmp_path, [Decision(Choice.DENY)], mode=Mode.EDITS)
    await _run(agent, PASTE)
    assert len(approver.requests) == 1 and not (workspace / "sketch.png").exists()


async def test_a_deny_rule_refuses_without_a_dialog(ctx: ToolContext, tmp_path: Path, workspace: Path) -> None:
    agent, _, approver = _agent(ctx, tmp_path, rules=_rules(workspace, tmp_path, deny=["edit_image"]))
    await _run(agent, PASTE)
    assert approver.requests == [] and "denied by rule" in _tool_message(agent)
    assert not (workspace / "sketch.png").exists()


async def test_a_protected_input_is_refused_without_a_dialog(ctx: ToolContext, tmp_path: Path, workspace: Path) -> None:
    (workspace / ".env").write_text("A=1\n")
    agent, _, approver = _agent(ctx, tmp_path, script=_script(image=".env"))
    await _run(agent)
    assert approver.requests == [] and _tool_message(agent) == "protected path"


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_diff_lists_the_edit_as_binary_and_undo_restores_the_source(ctx: ToolContext, tmp_path: Path, workspace: Path) -> None:
    source = workspace / "src.png"
    source.write_bytes(png(8, 8))
    store = CheckpointStore(workspace, tmp_path / "data" / "checkpoints", Guardrails.off(workspace), dict(os.environ))
    agent, _, _ = _agent(ctx, tmp_path, [ONCE], script=_script(image="src.png", path="src.png"), checkpoints=store)
    events: list[Any] = []
    await agent.run_turn("go", events.append)
    turn = agent._undo.peek()
    assert turn is not None
    diff = await build_diff(turn, store, workspace)
    assert [(i.status, i.path, i.note) for i in diff.items] == [("M", "src.png", "binary")]
    await agent.undo(events.append)
    assert source.read_bytes() == png(8, 8)
    store.close()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_undo_deletes_a_new_edit_output(ctx: ToolContext, tmp_path: Path, workspace: Path) -> None:
    store = CheckpointStore(workspace, tmp_path / "data" / "checkpoints", Guardrails.off(workspace), dict(os.environ))
    agent, _, _ = _agent(ctx, tmp_path, [ONCE], checkpoints=store)
    events: list[Any] = []
    await agent.run_turn("go", events.append, pasted=(PASTE,))
    assert (workspace / "sketch.png").exists()
    await agent.undo(events.append)
    assert not (workspace / "sketch.png").exists()
    store.close()


async def test_pasted_images_stay_available_on_a_stop_hook_continuation_step(ctx: ToolContext, tmp_path: Path, workspace: Path) -> None:
    from core.engine.hooks.config import DEFAULT_TIMEOUT, HookEntry
    from core.engine.hooks.setup import Hooks
    from core.testing.fakes import FakeRunner, exit_run, ok_run

    reply: list[ProviderEvent | Gate] = [TextDelta("first"), Done(1, 1)]
    script = [reply, *_script()]
    agent, _, _ = _agent(ctx, tmp_path, [ONCE], script=script)
    runner = FakeRunner([exit_run(2, "now edit it"), ok_run()])
    agent.hooks = Hooks((HookEntry("Stop", "x", None, DEFAULT_TIMEOUT, "global"),), [], runner, lambda: True, workspace)
    await _run(agent, PASTE)
    assert _tool_message(agent).startswith("edited sketch.png (PNG, 256\u00d7256, ")
