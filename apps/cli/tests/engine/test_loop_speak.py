"""speak inside the loop: approval rules, plan mode, Esc and /undo through the journal."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from src.engine.approval import Choice, Decision
from src.engine.guardrails.setup import Guardrails
from src.engine.loop import Agent
from src.engine.mcp.trust import TrustStore
from src.engine.mode import Mode
from src.engine.permissions.setup import Rules, prepare
from src.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.engine.tools import default_tools
from src.engine.tools import media as tools_media
from src.engine.tools.base import ToolContext
from src.engine.tools.media import SpeakTool

from tests.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider
from tests.helpers import assert_all_gone, wait_until
from tests.media_fakes import fake_speech, wav_bytes

ONCE = Decision(Choice.ONCE)
CALL: dict[str, Any] = {"text": "Hello there.", "path": "a.wav"}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root.resolve()


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, bash_timeout=5.0)


@pytest.fixture(autouse=True)
def _helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "bin"
    folder.mkdir()
    monkeypatch.setattr(tools_media, "SPEECH_HELPER", fake_speech(folder))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))


def _script() -> list[list[ProviderEvent | Gate]]:
    return [[ToolCallEvent(ToolCall("c1", "speak", CALL)), Done(10, 5)], [TextDelta("done"), Done(1, 1)]]


def _agent(
    ctx: ToolContext, decisions: list[Decision] | None = None, *, mode: Mode = Mode.BUILD, rules: Rules | None = None
) -> tuple[Agent, ScriptedProvider, RecordingApprover]:
    provider = ScriptedProvider(_script(), context_window=100_000)
    approver = RecordingApprover(decisions)
    speaker = SpeakTool("x/y", 20, Guardrails.off(ctx.workspace))
    agent = Agent(
        provider,
        default_tools(speaker=speaker),
        approver,
        ctx,
        25,
        today=lambda: date(2026, 11, 21),
        recorder=MemoryRecorder(),
        rules=rules,
    )
    agent.set_mode(mode)
    return agent, provider, approver


def _tool_message(agent: Agent) -> str:
    return next(m.content for m in agent.history if m.role == "tool")


def _rules(workspace: Path, tmp_path: Path, **settings: list[str]) -> Rules:
    path = tmp_path / "global.json"
    path.write_text(json.dumps({"permissions": settings}))
    return prepare(workspace, path, TrustStore(tmp_path / "trust.json"), {}).rules


async def test_a_call_asks_and_writes_the_wav(ctx: ToolContext, workspace: Path) -> None:
    agent, _, approver = _agent(ctx, [ONCE])
    await agent.run_turn("narrate", lambda event: None)
    assert len(approver.requests) == 1
    preview = approver.requests[0].preview
    assert preview is not None and preview.body.startswith("text: Hello there.\nvoice: af_heart, speed 1")
    assert (workspace / "a.wav").read_bytes() == wav_bytes(1.0)
    assert _tool_message(agent).startswith("spoke a.wav (WAV, 1.0 s, 24 kHz, ")


async def test_plan_mode_hides_the_tool_and_refuses_a_forced_call(ctx: ToolContext, workspace: Path) -> None:
    agent, provider, approver = _agent(ctx, mode=Mode.PLAN)
    await agent.run_turn("narrate", lambda event: None)
    assert "speak" not in provider.requests[0][1]
    assert _tool_message(agent) == "refused: speak is unavailable in plan mode"
    assert approver.requests == [] and not (workspace / "a.wav").exists()


async def test_accept_edits_mode_still_asks(ctx: ToolContext, workspace: Path) -> None:
    agent, _, approver = _agent(ctx, [Decision(Choice.DENY)], mode=Mode.EDITS)
    await agent.run_turn("narrate", lambda event: None)
    assert len(approver.requests) == 1 and not (workspace / "a.wav").exists()


async def test_a_deny_rule_on_the_path_refuses_without_a_dialog(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    agent, _, approver = _agent(ctx, rules=_rules(workspace, tmp_path, deny=["speak(a.wav)"]))
    await agent.run_turn("narrate", lambda event: None)
    assert approver.requests == [] and "denied by rule" in _tool_message(agent)
    assert not (workspace / "a.wav").exists()


async def test_esc_during_the_call_is_the_loops_interrupted_result_and_the_group_dies(
    ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pids = tmp_path / "pids"
    monkeypatch.setenv("FAKE_SPEECH_MODE", "sleep")
    monkeypatch.setenv("FAKE_SPEECH_PIDS", str(pids))
    agent, _, _ = _agent(ctx, [ONCE])
    task = asyncio.create_task(agent.run_turn("narrate", lambda event: None))
    await wait_until(lambda: pids.exists() and len(pids.read_text().split()) == 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _tool_message(agent) == "Interrupted by user"
    await assert_all_gone(*(int(v) for v in pids.read_text().split()))
    assert list(workspace.iterdir()) == []


async def test_undo_removes_a_new_wav_and_restores_an_overwritten_one(ctx: ToolContext, workspace: Path) -> None:
    events: list[Any] = []
    agent, _, _ = _agent(ctx, [ONCE])
    await agent.run_turn("narrate", events.append)
    assert (workspace / "a.wav").exists()
    await agent.undo(events.append)
    assert not (workspace / "a.wav").exists()

    (workspace / "a.wav").write_bytes(b"old")
    again, _, _ = _agent(ctx, [ONCE])
    await again.run_turn("narrate", events.append)
    assert (workspace / "a.wav").read_bytes() == wav_bytes(1.0)
    await again.undo(events.append)
    assert (workspace / "a.wav").read_bytes() == b"old"
