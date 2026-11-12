from __future__ import annotations

from pathlib import Path

import pytest
import src.tui
from src.agent import Agent
from src.agent.provider.base import Message
from src.agent.provider.ollama import OllamaProvider
from src.agent.session import FileRecorder, SessionStore, workspace_dir
from src.config.settings import get_settings
from src.tui import StartChoice
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


def _run_app(monkeypatch: pytest.MonkeyPatch, workspace: Path, **kwargs: object) -> AgentApp:
    ran: list[AgentApp] = []
    monkeypatch.setattr(AgentApp, "run", lambda self, **_: ran.append(self))
    src.tui.run(workspace, **kwargs)  # ty: ignore[invalid-argument-type]
    (app,) = ran
    return app


def test_run_passes_the_start_choice_to_the_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = _run_app(monkeypatch, tmp_path, start=StartChoice("resume", "01JF3K"))

    assert app.start == StartChoice("resume", "01JF3K")


def test_run_starts_a_new_session_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert _run_app(monkeypatch, tmp_path).start == StartChoice("new")


def test_run_builds_the_store_under_the_sessions_dir_from_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sessions = tmp_path / "saved"
    monkeypatch.setenv("AGENT_SESSIONS_DIR", str(sessions))
    get_settings.cache_clear()
    workspace = tmp_path / "w"
    workspace.mkdir()

    app = _run_app(monkeypatch, workspace)

    assert isinstance(app.store, SessionStore)
    assert app.store.list() == []
    recorder = app.agent._recorder
    assert isinstance(recorder, FileRecorder)
    recorder.message(Message("user", "hi"), None)
    recorder.close()
    assert recorder.path is not None
    assert recorder.path.parent == workspace_dir(sessions, workspace)
    assert [i.id for i in app.store.list()] == [recorder.path.stem]


def test_run_passes_the_compaction_settings_to_the_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENT_COMPACT_AT", "0.5")
    monkeypatch.setenv("AGENT_COMPACT_KEEP_TURNS", "3")
    get_settings.cache_clear()

    agent = _run_app(monkeypatch, tmp_path).agent

    assert (agent.compact_at, agent.compact_keep_turns) == (0.5, 3)


def test_run_gives_the_app_a_recorder_factory_for_new_sessions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = _run_app(monkeypatch, tmp_path)

    first, second = app.recorder_factory(), app.recorder_factory()

    assert isinstance(first, FileRecorder) and isinstance(second, FileRecorder)
    assert first is not second
    assert first.path is None  # lazy: nothing on disk until the first message
