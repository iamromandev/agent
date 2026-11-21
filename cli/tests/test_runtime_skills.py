"""The runtime ships the built-in skills to the app and to headless mode."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from src.engine.mode import Mode
from src.runtime import build_runtime

from tests.fakes import FakeCatalog, RecordingApprover, ScriptedProvider


def test_ac9_the_runtime_has_the_builtin_skills_and_the_prompt_lists_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    ws = tmp_path / "ws"
    ws.mkdir()
    rt = build_runtime(ws, set(), terminals=False, provider=ScriptedProvider([]), catalog=FakeCatalog())
    agent = rt.agent_factory(RecordingApprover())
    prompt = agent._system(Mode.BUILD, agent.instructions()).content
    asyncio.run(rt.close())
    assert {s.name for s in rt.skills.sorted() if s.source == "builtin"} >= {"commit", "debug", "tdd"}
    section = rt.skills.section()
    assert section is not None and "- tdd:" in section
    assert "- tdd:" in prompt and "- commit:" in prompt and "- debug:" in prompt
