"""generate_image inside the loop: plan mode, approval rules, redaction, undo and /diff (AC4, AC7, AC14, AC15)."""

from __future__ import annotations

import os
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from src.engine.approval import Choice, Decision
from src.engine.checkpoints.store import CheckpointStore
from src.engine.events import AgentEvent, TaskProgress, ToolFinished, ToolProgress
from src.engine.guardrails.setup import Guardrails
from src.engine.imagegen import MfluxGenerator, png_size
from src.engine.loop import Agent
from src.engine.mcp.trust import TrustStore
from src.engine.mode import Mode
from src.engine.permissions.setup import Rules, prepare
from src.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.tools import default_tools
from src.engine.tools.base import ToolContext
from src.engine.turn_diff import build_diff

from tests.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider
from tests.image_fakes import fake_mflux

ONCE = Decision(Choice.ONCE)
CALL: dict[str, Any] = {"prompt": "a plane", "path": "icon.png", "width": 256, "height": 256, "seed": 1}


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
    args = {**CALL, **overrides}
    return [[ToolCallEvent(ToolCall("c1", "generate_image", args)), Done(10, 5)], [TextDelta("done"), Done(1, 1)]]


def _agent(
    ctx: ToolContext,
    tmp_path: Path,
    decisions: list[Decision] | None = None,
    *,
    mode: Mode = Mode.BUILD,
    rules: Rules | None = None,
    script: list[list[ProviderEvent | Gate]] | None = None,
    checkpoints: CheckpointStore | None = None,
) -> tuple[Agent, ScriptedProvider, RecordingApprover]:
    from src.engine.tools.image import GenerateImageTool

    folder = tmp_path / "bin"
    folder.mkdir(exist_ok=True)
    generator = MfluxGenerator(fake_mflux(folder), "x/y", 20, Guardrails.off(ctx.workspace))
    provider = ScriptedProvider(script or _script(), context_window=100_000)
    approver = RecordingApprover(decisions)
    agent = Agent(
        provider,
        [*default_tools(), GenerateImageTool(generator)],
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


async def _run(agent: Agent) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    await agent.run_turn("go", events.append)
    return events


def _tool_message(agent: Agent) -> str:
    return next(m.content for m in agent.history if m.role == "tool")


def _rules(workspace: Path, tmp_path: Path, **settings: list[str]) -> Rules:
    (workspace / ".agent").mkdir(exist_ok=True)
    path = tmp_path / "global.json"
    import json

    path.write_text(json.dumps({"permissions": settings}))
    return prepare(workspace, path, TrustStore(tmp_path / "trust.json"), {}).rules


async def test_plan_mode_hides_the_tool_and_refuses_a_forced_call(ctx: ToolContext, tmp_path: Path, workspace: Path) -> None:
    agent, provider, approver = _agent(ctx, tmp_path, mode=Mode.PLAN)
    await _run(agent)
    assert "generate_image" not in provider.requests[0][1]
    assert _tool_message(agent) == "refused: generate_image is unavailable in plan mode"
    assert approver.requests == [] and not (workspace / "icon.png").exists()


async def test_the_tool_is_offered_in_build_mode(ctx: ToolContext, tmp_path: Path) -> None:
    agent, provider, _ = _agent(ctx, tmp_path, [ONCE])
    await _run(agent)
    assert "generate_image" in provider.requests[0][1]


async def test_accept_edits_mode_still_asks(ctx: ToolContext, tmp_path: Path, workspace: Path) -> None:
    agent, _, approver = _agent(ctx, tmp_path, [Decision(Choice.DENY)], mode=Mode.EDITS)
    await _run(agent)
    assert len(approver.requests) == 1
    assert not (workspace / "icon.png").exists()


async def test_the_dialog_shows_the_preview_and_an_allow_decision_generates(ctx: ToolContext, tmp_path: Path, workspace: Path) -> None:
    agent, _, approver = _agent(ctx, tmp_path, [ONCE])
    await _run(agent)
    request = approver.requests[0]
    assert request.preview is not None and "size: 256\u00d7256, 9 steps, seed 1" in request.preview.body
    assert png_size((workspace / "icon.png").read_bytes()) == (256, 256)
    assert _tool_message(agent).startswith("generated icon.png (PNG, 256\u00d7256, ")


async def test_a_deny_rule_refuses_without_a_dialog(ctx: ToolContext, tmp_path: Path, workspace: Path) -> None:
    rules = _rules(workspace, tmp_path, deny=["generate_image"])
    agent, _, approver = _agent(ctx, tmp_path, rules=rules)
    await _run(agent)
    assert approver.requests == []
    assert "denied by rule" in _tool_message(agent)
    assert not (workspace / "icon.png").exists()


async def test_an_allow_rule_skips_the_dialog(ctx: ToolContext, tmp_path: Path, workspace: Path) -> None:
    rules = _rules(workspace, tmp_path, allow=["generate_image"])
    agent, _, approver = _agent(ctx, tmp_path, rules=rules)
    await _run(agent)
    assert approver.requests == []
    assert (workspace / "icon.png").exists()


async def test_an_allow_rule_for_one_prompt_skips_only_that_prompt(ctx: ToolContext, tmp_path: Path, workspace: Path) -> None:
    rules = _rules(workspace, tmp_path, allow=["generate_image(a plane)"])
    agent, _, approver = _agent(ctx, tmp_path, [Decision(Choice.DENY)], rules=rules, script=_script(prompt="another"))
    await _run(agent)
    assert len(approver.requests) == 1


async def test_a_secret_in_stderr_is_redacted_in_the_result(
    ctx: ToolContext, tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "token")
    agent, _, _ = _agent(ctx, tmp_path, [ONCE])
    events = await _run(agent)
    message = _tool_message(agent)
    assert message.startswith("mflux failed (exit 1)") and "[redacted:github-token]" in message
    assert "ghp_" + "a" * 40 not in message
    assert next(e for e in events if isinstance(e, ToolFinished)).result.is_error


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_undo_removes_a_new_image_and_diff_lists_it_as_binary(ctx: ToolContext, tmp_path: Path, workspace: Path) -> None:
    store = CheckpointStore(workspace, tmp_path / "data" / "checkpoints", Guardrails.off(workspace), dict(os.environ))
    agent, _, _ = _agent(ctx, tmp_path, [ONCE], checkpoints=store)
    events = await _run(agent)
    turn = agent._undo.peek()
    assert turn is not None
    diff = await build_diff(turn, store, workspace)
    assert [(i.status, i.path, i.note) for i in diff.items] == [("A", "icon.png", "binary")]
    await agent.undo(events.append)
    assert not (workspace / "icon.png").exists()
    store.close()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_undo_restores_an_overwritten_image(ctx: ToolContext, tmp_path: Path, workspace: Path) -> None:
    old = workspace / "icon.png"
    old.write_bytes(b"\x89PNG\r\n\x1a\nold image")
    store = CheckpointStore(workspace, tmp_path / "data" / "checkpoints", Guardrails.off(workspace), dict(os.environ))
    agent, _, _ = _agent(ctx, tmp_path, [ONCE], checkpoints=store)
    events = await _run(agent)
    assert png_size(old.read_bytes()) == (256, 256)
    await agent.undo(events.append)
    assert old.read_bytes() == b"\x89PNG\r\n\x1a\nold image"
    store.close()


async def test_undo_works_from_the_file_log_without_checkpoints(ctx: ToolContext, tmp_path: Path, workspace: Path) -> None:
    agent, _, _ = _agent(ctx, tmp_path, [ONCE])
    events = await _run(agent)
    assert (workspace / "icon.png").exists()
    await agent.undo(events.append)
    assert not (workspace / "icon.png").exists()


async def test_progress_reaches_the_events_but_not_the_history_or_the_session(
    ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "progress")
    agent, _, _ = _agent(ctx, tmp_path, [ONCE])
    events = await _run(agent)
    texts = [e.event.text for e in events if isinstance(e, TaskProgress) and isinstance(e.event, ToolProgress)]
    assert texts and all(e.call_id == "c1" for e in events if isinstance(e, TaskProgress))
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert not finished.result.is_error
    for text in (_tool_message(agent), finished.result.output):
        assert "loading model" not in text and "step " not in text and "downloading" not in text
    recorder = agent.recorder
    assert not any(t in repr(vars(recorder)) for t in ("loading model", "downloading model", "starting\u2026"))
