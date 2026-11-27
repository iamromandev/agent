"""MCP servers inside the app: they start in the background, say what happened, and stop with the app."""

from __future__ import annotations

from pathlib import Path

from core.engine.mcp.config import ServerConfig
from core.engine.mcp.contract import ToolInfo
from core.engine.mcp.hub import McpHub
from core.engine.provider.base import Done, TextDelta
from core.testing.fakes import FakeConnection, FakeConnector
from src.tui.app import AgentApp
from src.tui.widgets.transcript import NoticeEntry

from tests.tui.conftest import AppFactory, until

ECHO = ToolInfo("echo", "Echo it", {"type": "object", "properties": {"text": {"type": "string"}}})


def _notices(app: AgentApp) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _hub(
    tmp_path: Path, connector: FakeConnector, *names: str, warnings: list[str] | None = None
) -> McpHub:
    return McpHub(connector, tmp_path.resolve(), [ServerConfig(n, "cmd") for n in names], 5.0, warnings or [])


def _reply(text: str = "ok") -> list:
    return [TextDelta(text), Done(1, 1)]


async def test_a_server_that_exits_is_a_warning_that_names_it(make_app: AppFactory, tmp_path: Path) -> None:
    connection = FakeConnection([ECHO])
    hub = _hub(tmp_path, FakeConnector({"a": connection}), "a")
    app = make_app(mcp=hub)
    async with app.run_test() as pilot:
        await until(pilot, lambda: any("mcp a ready" in n for n in _notices(app)))
        connection.trigger_exit("the server exited")
        await until(pilot, lambda: any("mcp a exited: the server exited" in n for n in _notices(app)))


async def test_config_warnings_are_shown_at_startup(make_app: AppFactory, tmp_path: Path) -> None:
    hub = _hub(tmp_path, FakeConnector(), warnings=["~/.config/agent/mcp.json: invalid JSON: boom"])
    app = make_app(mcp=hub)
    async with app.run_test() as pilot:
        await until(pilot, lambda: any("mcp.json: invalid JSON: boom" in n for n in _notices(app)))
