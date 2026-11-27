from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
import src.tui
import src.tui.app as tui_app
from core.client import OllamaClient, SearxClient, WebClient
from core.config.settings import get_settings
from core.engine.mcp.hub import McpHub
from core.engine.terminal import manager as terminal_manager
from core.engine.terminal.manager import TerminalManager
from core.testing.helpers import assert_all_gone
from src.tui.app import AgentApp


@pytest.fixture(autouse=True)
def ollama(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Ollama as `run` sees it at launch: `tags` entries and `show` bodies by name (the network is blocked)."""
    entries: list[dict] = []
    answers: dict[str, Any] = {"show": {}, "tags": entries}

    async def tags(self: OllamaClient) -> list[dict]:
        return entries

    async def show(self: OllamaClient, name: str, timeout: float) -> dict:
        return answers["show"].get(name, {})

    monkeypatch.setattr(OllamaClient, "tags", tags)
    monkeypatch.setattr(OllamaClient, "show", show)
    return answers


def test_the_search_client_is_closed_on_quit_and_if_setup_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    closed: list[str] = []

    async def search_aclose(self: SearxClient) -> None:
        closed.append("search")

    monkeypatch.setenv("AGENT_SEARCH_URL", "http://localhost:8888")
    get_settings.cache_clear()
    monkeypatch.setattr(SearxClient, "aclose", search_aclose)

    def failing_app(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("no app")

    monkeypatch.setattr(tui_app, "AgentApp", failing_app)
    try:
        with pytest.raises(RuntimeError, match="no app"):
            src.tui.run(tmp_path)
    finally:
        get_settings.cache_clear()

    assert closed == ["search"]


def _run_app(monkeypatch: pytest.MonkeyPatch, workspace: Path, **kwargs: object) -> AgentApp:
    ran: list[AgentApp] = []
    monkeypatch.setattr(AgentApp, "run", lambda self, **_: ran.append(self))
    src.tui.run(workspace, **kwargs)  # ty: ignore[invalid-argument-type]
    (app,) = ran
    return app


class _Wired:
    """What `run` hands to AgentApp, and the order in which quitting closes things."""

    def __init__(self) -> None:
        self.on_close: Callable[[], Awaitable[None]] | None = None
        self.manager: TerminalManager | None = None
        self.order: list[str] = []


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Wired:
    seen = _Wired()

    class SpyManager(TerminalManager):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, grace=0.3, **kwargs)
            seen.manager = self

        async def close_all(self) -> None:
            seen.order.append("terminals")
            await super().close_all()

    class FakeApp:
        return_code = None

        def __init__(self, *args: Any, on_close: Callable[[], Awaitable[None]] | None = None, **kwargs: Any) -> None:
            seen.on_close = on_close

        def run(self, **_: object) -> None:
            pass

    async def aclose(self: OllamaClient) -> None:
        seen.order.append("client")

    async def web_aclose(self: WebClient) -> None:
        seen.order.append("web")

    monkeypatch.setattr(terminal_manager, "TerminalManager", SpyManager)
    monkeypatch.setattr(WebClient, "aclose", web_aclose)
    monkeypatch.setattr(tui_app, "AgentApp", FakeApp)
    monkeypatch.setattr(OllamaClient, "aclose", aclose)
    src.tui.run(tmp_path)
    seen.order.clear()  # the launch's own probe client was closed during `run`; the tests look at the quit order
    return seen


def test_a_failing_app_still_kills_the_terminals_it_leaves(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The app can fail before it unmounts (and closes the terminals); `run` must not leave them behind."""
    monkeypatch.setenv("SHELL", "/bin/sh")
    started: list[int] = []

    def failing_run(self: AgentApp, **_: object) -> None:
        async def start() -> None:
            assert self.terminals is not None
            started.append(self.terminals.start("api", "trap '' TERM HUP; sleep 1000").process.pid)

        asyncio.run(start())  # the loop is gone afterwards, as it is when the app has crashed
        raise RuntimeError("boom")

    monkeypatch.setattr(AgentApp, "run", failing_run)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            src.tui.run(tmp_path)
        asyncio.run(assert_all_gone(*started))
    finally:
        for pid in started:  # a failing assertion must not leave the process running
            with contextlib.suppress(OSError):
                os.killpg(pid, signal.SIGKILL)


async def test_quitting_closes_everything_else_even_if_stopping_the_mcp_servers_fails(
    wired: _Wired, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert wired.on_close is not None

    async def broken(self: McpHub) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(McpHub, "close_all", broken)

    with pytest.raises(RuntimeError, match="boom"):
        await wired.on_close()

    assert wired.order == ["terminals", "web", "client"]


def test_run_builds_one_guard_from_both_settings_files_and_shares_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "config"
    (config / "agent").mkdir(parents=True)
    (config / "agent" / "settings.json").write_text('{"guardrails": {"keepEnv": ["KEPT_TOKEN"], "protected": ["*.vault"]}}')
    workspace = tmp_path / "ws"
    (workspace / ".agent").mkdir(parents=True)
    (workspace / ".agent" / "settings.json").write_text('{"guardrails": {"sandbox": {"enabled": false}, "protected": ["p/**"]}}')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_agentsecret1234")
    monkeypatch.setenv("KEPT_TOKEN", "kept-value-1")
    get_settings.cache_clear()

    try:
        app = _run_app(monkeypatch, workspace)
    finally:
        get_settings.cache_clear()

    guard = app.guard
    assert app.agent.ctx.guard is guard
    assert app.terminals is not None
    assert app.terminals._guard is guard
    assert "GITHUB_TOKEN" in guard.removed_env
    assert "KEPT_TOKEN" not in guard.removed_env
    assert guard.paths.extra_count == 2
    (warning,) = [w for w in app._permission_warnings if "guardrails" in w]
    assert "guardrails.sandbox is no longer supported and is ignored" in warning
    assert "ghp_agentsecret1234" not in " ".join(app._permission_warnings)


def _project_model(workspace: Path, model: object) -> None:
    path = workspace / ".agent" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"model": model}))


def test_run_says_on_stderr_when_asking_ollama_takes_long(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    async def slow_show(self: OllamaClient, name: str, timeout: float) -> dict:
        await asyncio.sleep(0.2)
        return {}

    monkeypatch.setattr(OllamaClient, "show", slow_show)
    monkeypatch.setattr("core.engine.provider.models.SLOW_AFTER", 0.02)
    monkeypatch.setenv("AGENT_MODEL", "foo\x1b[2J")
    get_settings.cache_clear()
    try:
        _run_app(monkeypatch, tmp_path)
    finally:
        get_settings.cache_clear()

    err = capsys.readouterr().err
    assert err == "Asking Ollama about foo\\x1b[2J…\n"
