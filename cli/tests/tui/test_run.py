from __future__ import annotations

from pathlib import Path

import pytest
import src.tui
from src.agent import Agent
from src.agent.provider.ollama import OllamaProvider
from src.config.settings import get_settings
from src.tui.app import AgentApp


def test_run_wires_settings_into_the_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENT_MODEL", "foo")
    monkeypatch.setenv("AGENT_NUM_CTX", "4096")
    monkeypatch.setenv("AGENT_MAX_STEPS", "7")
    get_settings.cache_clear()
    ran: list[AgentApp] = []
    monkeypatch.setattr(AgentApp, "run", lambda self, **_: ran.append(self))

    try:
        src.tui.run(tmp_path)
    finally:
        get_settings.cache_clear()

    (app,) = ran
    assert isinstance(app.provider, OllamaProvider)
    assert (app.provider.model, app.provider.context_window) == ("foo", 4096)
    assert app.workspace == tmp_path.resolve()
    agent = app.agent
    assert isinstance(agent, Agent)
    assert agent.max_steps == 7
    assert agent.ctx.workspace == tmp_path.resolve()
    assert {t.name for t in agent.tools} == {"read", "glob", "grep", "write", "edit", "bash"}
