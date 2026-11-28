"""The runtime builds the LSP hub from the settings and gives it to the main agent; close and kill_all stop servers."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from core.engine.approval import Choice, Decision
from core.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from core.runtime import Runtime, build_runtime
from core.testing.fakes import FakeCatalog, Gate, RecordingApprover, ScriptedProvider
from core.testing.helpers import assert_all_gone
from core.testing.lsp_fakes import fake_lsp


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_lsp(bin_dir, "ty")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_LSP_DUMP", str(tmp_path / "dump.jsonl"))
    return root.resolve()


def settings(value: object) -> None:
    folder = Path(os.environ["XDG_CONFIG_HOME"]) / "agent"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "settings.json").write_text(json.dumps({"lsp": value}))


def build(ws: Path, provider: ScriptedProvider | None = None) -> Runtime:
    return build_runtime(ws, set(), terminals=False, provider=provider or ScriptedProvider([]), catalog=FakeCatalog())


def pids(tmp_path: Path) -> set[int]:
    dump = tmp_path / "dump.jsonl"
    return {json.loads(x)["pid"] for x in dump.read_text().splitlines()} if dump.exists() else set()


def test_ac1_without_the_key_there_is_no_hub(ws: Path) -> None:
    rt = build(ws)
    asyncio.run(rt.close())
    assert rt.lsp is None


def test_with_the_key_there_is_a_hub_and_its_warnings_join_the_launch_warnings(ws: Path) -> None:
    settings({"servers": {"mydsl": {"args": []}}})
    rt = build(ws)
    asyncio.run(rt.close())
    assert rt.lsp is not None and "ty" in [row[0] for row in rt.lsp.listing()]
    assert any("mydsl" in w for w in rt.warnings)


def write_script(content: str) -> list[list[ProviderEvent | Gate]]:
    call: list[ProviderEvent | Gate] = [ToolCallEvent(ToolCall("c1", "write", {"path": "a.py", "content": content})), Done(1, 1)]
    call2: list[ProviderEvent | Gate] = [ToolCallEvent(ToolCall("c2", "write", {"path": "a.py", "content": content + "y"})), Done(1, 1)]
    return [call, call2, [TextDelta("done"), Done(1, 1)]]


async def test_ac2_two_writes_run_one_server_and_the_result_has_diagnostics(ws: Path, tmp_path: Path) -> None:
    settings(True)
    rt = build(ws, ScriptedProvider(write_script("x = 2\n")))
    agent = rt.agent_factory(RecordingApprover([Decision(Choice.ONCE)] * 2))
    await agent.run_turn("go", lambda e: None)
    result = next(m.content for m in agent.history if m.tool_call_id == "c2")
    assert "[diagnostics from ty:" in result and len(pids(tmp_path)) == 1
    found = pids(tmp_path)
    await rt.close()
    await assert_all_gone(*found)


async def test_ac13_kill_all_leaves_no_server_alive(ws: Path, tmp_path: Path) -> None:
    settings(True)
    rt = build(ws, ScriptedProvider(write_script("x = 2\n")))
    agent = rt.agent_factory(RecordingApprover([Decision(Choice.ONCE)] * 2))
    await agent.run_turn("go", lambda e: None)
    found = pids(tmp_path)
    assert found
    rt.kill_all()
    await assert_all_gone(*found)


async def test_ac13_close_leaves_no_server_alive(ws: Path, tmp_path: Path) -> None:
    settings(True)
    rt = build(ws, ScriptedProvider(write_script("x = 2\n")))
    agent = rt.agent_factory(RecordingApprover([Decision(Choice.ONCE)] * 2))
    await agent.run_turn("go", lambda e: None)
    found = pids(tmp_path)
    assert found
    await rt.close()
    await assert_all_gone(*found)


def test_every_agent_the_factory_makes_has_the_runtimes_hub(ws: Path) -> None:
    """The daemon builds each session's agent with `rt.agent_factory`, so this is the daemon path too."""
    settings(True)
    rt = build(ws)
    agent = rt.agent_factory(RecordingApprover([]))
    asyncio.run(rt.close())
    assert rt.lsp is not None and agent.lsp is rt.lsp


async def test_a_held_global_marker_keeps_global_servers_off_until_trust(ws: Path, tmp_path: Path) -> None:
    from core.engine.configguard import GlobalMarker, marker_path

    GlobalMarker(marker_path()).write(["~/.config/agent/settings.json"])
    settings(True)
    rt = build(ws, ScriptedProvider(write_script("x = 2\n") + write_script("x = 3\n")))
    assert rt.lsp is not None and rt.lsp.listing()[0][2:] == ("untrusted", "untrusted")
    agent = rt.agent_factory(RecordingApprover([Decision(Choice.ONCE)] * 2))
    await agent.run_turn("go", lambda e: None)
    assert pids(tmp_path) == set()  # held: nothing started, no block

    await asyncio.to_thread(rt.config_guard.trust_global)  # the launch dialog's Trust
    agent = rt.agent_factory(RecordingApprover([Decision(Choice.ONCE)] * 2))
    await agent.run_turn("go", lambda e: None)
    assert len(pids(tmp_path)) == 1
    await rt.close()

