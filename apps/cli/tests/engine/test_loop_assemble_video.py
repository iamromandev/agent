"""assemble_video inside the loop: approval rules, plan mode, Esc, redaction and /undo through the journal."""

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
from src.engine.tools.media import AssembleVideoTool

from tests.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider
from tests.helpers import assert_all_gone, wait_until
from tests.image_fakes import png
from tests.media_fakes import fake_ffmpeg, fake_media, wav_bytes

ONCE = Decision(Choice.ONCE)
CALL: dict[str, Any] = {
    "scenes": [{"image": "a.png", "audio": "n.wav", "text": "Hello there"}, {"image": "a.png", "duration": 2}],
    "path": "out.mp4",
}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.png").write_bytes(png(40, 30))
    (root / "n.wav").write_bytes(wav_bytes(2.0))
    return root.resolve()


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, bash_timeout=5.0)


@pytest.fixture(autouse=True)
def _fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "bin"
    folder.mkdir()
    monkeypatch.setattr(tools_media, "MEDIA_HELPER", fake_media(folder))
    fake_ffmpeg(folder)
    (folder / "temp").mkdir()
    monkeypatch.setenv("TMPDIR", str(folder / "temp"))
    monkeypatch.setattr("tempfile.tempdir", None)


def _script() -> list[list[ProviderEvent | Gate]]:
    return [[ToolCallEvent(ToolCall("c1", "assemble_video", CALL)), Done(10, 5)], [TextDelta("done"), Done(1, 1)]]


def _agent(
    ctx: ToolContext,
    tmp_path: Path,
    decisions: list[Decision] | None = None,
    *,
    mode: Mode = Mode.BUILD,
    rules: Rules | None = None,
) -> tuple[Agent, ScriptedProvider, RecordingApprover]:
    provider = ScriptedProvider(_script(), context_window=100_000)
    approver = RecordingApprover(decisions)
    video = AssembleVideoTool(tmp_path / "bin" / "ffmpeg-fake", 20, Guardrails.off(ctx.workspace))
    agent = Agent(
        provider,
        default_tools(video=video),
        approver,
        ctx,
        25,
        today=lambda: date(2026, 11, 22),
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


async def test_a_call_asks_and_writes_the_mp4(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    agent, _, approver = _agent(ctx, tmp_path, [ONCE])
    await agent.run_turn("make a reel", lambda event: None)
    assert len(approver.requests) == 1
    preview = approver.requests[0].preview
    assert preview is not None
    lines = preview.body.split("\n")
    assert lines[0] == "scene 1: a.png, narration n.wav (2.0 s), zoom-in, text: Hello there"
    assert lines[1] == "scene 2: a.png, 2.0 s, zoom-in"
    assert lines[2:] == ["music: none", "crossfade: 0.5 s", "total: 4.5 s"]
    assert (workspace / "out.mp4").read_bytes()[4:8] == b"ftyp"
    assert _tool_message(agent).startswith("assembled out.mp4 (MP4, 1080\N{MULTIPLICATION SIGN}1920, 4.5 s, 2 scenes, ")


async def test_plan_mode_hides_the_tool_and_refuses_a_forced_call(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    agent, provider, approver = _agent(ctx, tmp_path, mode=Mode.PLAN)
    await agent.run_turn("make a reel", lambda event: None)
    assert "assemble_video" not in provider.requests[0][1]
    assert _tool_message(agent) == "refused: assemble_video is unavailable in plan mode"
    assert approver.requests == [] and not (workspace / "out.mp4").exists()


async def test_accept_edits_mode_still_asks(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    agent, _, approver = _agent(ctx, tmp_path, [Decision(Choice.DENY)], mode=Mode.EDITS)
    await agent.run_turn("make a reel", lambda event: None)
    assert len(approver.requests) == 1 and not (workspace / "out.mp4").exists()


async def test_a_deny_rule_on_the_path_refuses_without_a_dialog(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    agent, _, approver = _agent(ctx, tmp_path, rules=_rules(workspace, tmp_path, deny=["assemble_video(out.mp4)"]))
    await agent.run_turn("make a reel", lambda event: None)
    assert approver.requests == [] and "denied by rule" in _tool_message(agent)
    assert not (workspace / "out.mp4").exists()


async def test_esc_during_the_call_is_the_loops_interrupted_result_and_the_group_dies(
    ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pids = tmp_path / "pids"
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "sleep")
    monkeypatch.setenv("FAKE_FFMPEG_PIDS", str(pids))
    agent, _, _ = _agent(ctx, tmp_path, [ONCE])
    task = asyncio.create_task(agent.run_turn("make a reel", lambda event: None))
    await wait_until(lambda: pids.exists() and len(pids.read_text().split()) == 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _tool_message(agent) == "Interrupted by user"
    await assert_all_gone(*(int(v) for v in pids.read_text().split()))
    assert not (workspace / "out.mp4").exists()
    assert list((tmp_path / "bin" / "temp").iterdir()) == []


async def test_a_secret_in_ffmpegs_stderr_is_redacted_before_the_model_sees_it(
    ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "token")
    agent, _, _ = _agent(ctx, tmp_path, [ONCE])
    await agent.run_turn("make a reel", lambda event: None)
    message = _tool_message(agent)
    assert message.startswith("ffmpeg failed (exit 1)") and "ghp_aaaa" not in message


async def test_undo_removes_a_new_mp4_and_restores_an_overwritten_one(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    events: list[Any] = []
    agent, _, _ = _agent(ctx, tmp_path, [ONCE])
    await agent.run_turn("make a reel", events.append)
    assert (workspace / "out.mp4").exists()
    await agent.undo(events.append)
    assert not (workspace / "out.mp4").exists()

    (workspace / "out.mp4").write_bytes(b"old")
    again, _, _ = _agent(ctx, tmp_path, [ONCE])
    await again.run_turn("make a reel", events.append)
    assert (workspace / "out.mp4").read_bytes()[4:8] == b"ftyp"
    await again.undo(events.append)
    assert (workspace / "out.mp4").read_bytes() == b"old"
