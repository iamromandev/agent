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
from src.agent import Agent
from src.agent.mcp.hub import McpHub
from src.agent.provider.base import Message
from src.agent.provider.ollama import OllamaProvider
from src.agent.session import FileRecorder, SessionStore, workspace_dir
from src.agent.terminal import manager as terminal_manager
from src.agent.terminal.manager import TerminalManager
from src.client import OllamaClient, WebClient
from src.config.settings import get_settings
from src.tui import StartChoice
from src.tui.app import AgentApp

from tests.helpers import assert_all_gone


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
    assert {t.name for t in agent.tools} == {
        *("read", "glob", "grep", "write", "edit", "bash"),
        *("term_start", "term_read", "term_stop"),
        "web_fetch",
    }


def test_run_gives_the_app_the_terminal_manager_the_tools_use(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ran: list[AgentApp] = []
    monkeypatch.setattr(AgentApp, "run", lambda self, **_: ran.append(self))

    src.tui.run(tmp_path)

    (app,) = ran
    assert isinstance(app.terminals, TerminalManager)
    start = next(tool for tool in app.agent.tools if tool.name == "term_start")
    assert start._terminals is app.terminals  # ty: ignore[unresolved-attribute]


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
    return seen


async def test_quitting_stops_the_terminals_before_the_client_is_closed(wired: _Wired, tmp_path: Path) -> None:
    assert wired.on_close is not None and wired.manager is not None
    terminal = wired.manager.start("api", "sleep 30")
    pid = terminal.process.pid
    try:
        await wired.on_close()

        assert wired.order == ["terminals", "web", "client"]
        assert terminal.status[0] == "exited"
        await assert_all_gone(pid)
    finally:
        await wired.manager.close_all()  # a failing assertion must not leave the process running


async def test_quitting_closes_the_client_even_if_stopping_the_terminals_fails(
    wired: _Wired, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert wired.on_close is not None and wired.manager is not None

    async def broken() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(wired.manager, "close_all", broken)

    with pytest.raises(RuntimeError, match="boom"):
        await wired.on_close()

    assert wired.order == ["web", "client"]


async def test_quitting_closes_the_clients_even_if_closing_the_web_client_fails(
    wired: _Wired, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert wired.on_close is not None

    async def broken(self: WebClient) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(WebClient, "aclose", broken)

    with pytest.raises(RuntimeError, match="boom"):
        await wired.on_close()

    assert wired.order == ["terminals", "client"]


def test_the_web_client_is_closed_if_setting_up_the_app_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    closed: list[str] = []

    async def web_aclose(self: WebClient) -> None:
        closed.append("web")

    def failing_app(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("no app")

    monkeypatch.setattr(WebClient, "aclose", web_aclose)
    monkeypatch.setattr(tui_app, "AgentApp", failing_app)

    with pytest.raises(RuntimeError, match="no app"):
        src.tui.run(tmp_path)

    assert closed == ["web"]


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


def test_run_returns_the_exit_code_of_the_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(AgentApp, "run", lambda self, **_: None)
    monkeypatch.setattr(AgentApp, "return_code", property(lambda self: 143))

    assert src.tui.run(tmp_path) == 143


def test_run_builds_the_mcp_hub_from_the_global_config_without_starting_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "config"
    (config / "agent").mkdir(parents=True)
    (config / "agent" / "mcp.json").write_text('{"servers": {"notes": {"command": "uvx", "args": ["n"]}}}')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("AGENT_MCP_TIMEOUT", "3")
    monkeypatch.setenv("AGENT_MCP_START_TIMEOUT", "90")
    get_settings.cache_clear()

    app = _run_app(monkeypatch, tmp_path)

    assert app.mcp is not None
    assert [(s.config.name, s.config.command, s.config.source, s.status) for s in app.mcp.states()] == [
        ("notes", "uvx", "global", "starting")
    ]
    assert app.mcp._timeout == 3.0
    assert app.mcp._start_timeout == 90.0
    assert app.agent.all_tools() == app.agent.tools  # nothing is ready, so nothing is added


def test_run_reports_an_invalid_mcp_config_as_a_warning_and_starts_without_servers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "config"
    (config / "agent").mkdir(parents=True)
    (config / "agent" / "mcp.json").write_text("{not json")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))

    app = _run_app(monkeypatch, tmp_path)

    assert app.mcp is not None and app.mcp.states() == []
    events: list[tuple[str, str, str]] = []
    app.mcp.listen(events.append)
    assert [(kind, server) for kind, server, _ in events] == [("warn", "")]
    assert str(config / "agent" / "mcp.json") in events[0][2]


def test_run_kills_what_is_left_of_the_mcp_servers_after_the_app_ends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    killed: list[str] = []
    monkeypatch.setattr(McpHub, "kill_all", lambda self: killed.append("mcp"))

    _run_app(monkeypatch, tmp_path)
    assert killed == ["mcp"]

    def failing_run(self: AgentApp, **_: object) -> None:
        raise RuntimeError("app crashed")

    monkeypatch.setattr(AgentApp, "run", failing_run)
    with pytest.raises(RuntimeError, match="app crashed"):
        src.tui.run(tmp_path)
    assert killed == ["mcp", "mcp"]


async def test_quitting_closes_the_mcp_servers_first(wired: _Wired, monkeypatch: pytest.MonkeyPatch) -> None:
    assert wired.on_close is not None

    async def close_all(self: McpHub) -> None:
        wired.order.append("mcp")

    monkeypatch.setattr(McpHub, "close_all", close_all)

    await wired.on_close()

    assert wired.order == ["mcp", "terminals", "web", "client"]


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


def test_run_really_kills_a_server_the_app_left_running_when_its_loop_is_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.client.test_mcp import fake_config
    from tests.helpers import gone, wait_until

    config = tmp_path / "config"
    (config / "agent").mkdir(parents=True)
    server = fake_config()
    (config / "agent" / "mcp.json").write_text(
        json.dumps({"servers": {"fake": {"command": server.command, "args": list(server.args), "env": dict(server.env)}}})
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    loop = asyncio.new_event_loop()
    pids: list[int] = []

    def abandoned_run(self: AgentApp, **_: object) -> None:
        """The app's loop dies with the server still running: nothing closed the hub."""
        hub = self.mcp
        assert hub is not None

        async def start() -> None:
            hub.start()
            await wait_until(lambda: hub.states()[0].status == "ready", 15)
            pids.append(int((await hub.call("fake", "pid", {})).parts[0].split()[0]))

        loop.run_until_complete(start())

    monkeypatch.setattr(AgentApp, "run", abandoned_run)
    workspace = tmp_path / "w"
    workspace.mkdir()
    try:
        src.tui.run(workspace)

        (pid,) = pids
        loop.run_until_complete(wait_until(lambda: gone(pid), 5))
    finally:
        loop.run_until_complete(asyncio.sleep(0.1))
        loop.close()


def test_run_really_kills_a_server_that_was_still_starting_when_the_app_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.client.test_mcp import fake_config
    from tests.helpers import gone, wait_until

    config = tmp_path / "config"
    (config / "agent").mkdir(parents=True)
    server = fake_config("slow_init")
    (config / "agent" / "mcp.json").write_text(
        json.dumps({"servers": {"fake": {"command": server.command, "args": list(server.args), "env": dict(server.env)}}})
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    loop = asyncio.new_event_loop()
    pids: list[int] = []

    def abandoned_run(self: AgentApp, **_: object) -> None:
        hub = self.mcp
        assert hub is not None

        async def start() -> None:
            hub.start()
            connector = hub._connector
            await wait_until(lambda: bool(connector._pids), 10)  # ty: ignore[unresolved-attribute]
            pids.extend(connector._pids)  # ty: ignore[unresolved-attribute]

        loop.run_until_complete(start())

    monkeypatch.setattr(AgentApp, "run", abandoned_run)
    workspace = tmp_path / "w"
    workspace.mkdir()
    try:
        src.tui.run(workspace)

        (pid,) = pids
        loop.run_until_complete(wait_until(lambda: gone(pid), 5))
    finally:
        loop.run_until_complete(asyncio.sleep(0.1))
        loop.close()
