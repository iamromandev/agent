"""Project MCP servers: nothing starts until the user trusts the project file; `/mcp` lists every server."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from src.engine.mcp.config import ServerConfig
from src.engine.mcp.contract import McpError, ToolInfo
from src.engine.mcp.hub import McpHub
from src.engine.mcp.setup import McpSetup, prepare
from src.engine.mcp.trust import TrustStore, trust_store_path
from src.tui.app import AgentApp
from src.tui.screens.mcp_trust import McpTrustScreen, describe
from src.tui.widgets.transcript import NoticeEntry
from textual.widgets import Label, OptionList

from tests.fakes import FakeConnection, FakeConnector
from tests.tui.conftest import AppFactory, type_and_send, until

ECHO = ToolInfo("echo", "Echo it", {"type": "object"})
QUERY = ToolInfo("query", "Run it", {"type": "object"})


class Launch:
    """What `src.tui.run` wires for a folder: both config files, the trust store, the hub and the app."""

    def __init__(self, make_app: AppFactory, tmp_path: Path, connector: FakeConnector) -> None:
        self.make_app = make_app
        self.workspace = tmp_path.resolve()
        self.store = TrustStore(trust_store_path())  # under the test's XDG_DATA_HOME, outside the workspace
        self.connector = connector
        self.global_path = tmp_path / "config" / "mcp.json"
        self.hub: McpHub

    def write_project(self, servers: dict[str, object]) -> Path:
        path = self.workspace / ".agent" / "mcp.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"servers": servers}))
        return path

    def write_global(self, servers: dict[str, object]) -> None:
        self.global_path.parent.mkdir(parents=True, exist_ok=True)
        self.global_path.write_text(json.dumps({"servers": servers}))

    def app(self, store: TrustStore | None = None, remember: bool = True) -> AgentApp:
        setup: McpSetup = prepare(self.workspace, self.global_path, store or self.store)
        self.hub = McpHub(self.connector, self.workspace, setup.servers, 5.0, setup.warnings, untrusted=setup.untrusted)
        return self.make_app(mcp=self.hub, mcp_trust=setup.remember if remember else None)


@pytest.fixture
def launch(make_app: AppFactory, tmp_path: Path) -> Launch:
    connector = FakeConnector(
        {"notes": FakeConnection([ECHO]), "db": FakeConnection([QUERY]), "web": FakeConnection()}
    )
    return Launch(make_app, tmp_path, connector)


def _notices(app: AgentApp) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _is_prompt(app: AgentApp) -> bool:
    """The trust prompt is fully mounted: its options have focus and "Don't trust" is highlighted."""
    screen = app.screen
    if not isinstance(screen, McpTrustScreen):
        return False
    options = screen.query(OptionList)
    return bool(options) and options.first().highlighted == 0 and screen.focused is options.first()


async def _wait_for_guard(pilot) -> None:
    """The prompt ignores keys for a moment after it opens; wait that out before answering like a user."""
    await until(pilot, lambda: time.monotonic() >= pilot.app._keys_ignored_until)


async def _answer(pilot, option_id: str) -> None:
    await _wait_for_guard(pilot)
    options = pilot.app.screen.query_one(OptionList)
    options.highlighted = options.get_option_index(option_id)
    await pilot.press("enter")


async def test_trusting_starts_the_servers_and_remembers_the_choice(launch: Launch) -> None:
    launch.write_project({"db": {"command": "psql"}})
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await _answer(pilot, "trust")
        await until(pilot, lambda: any("mcp db ready · 1 tool" in n for n in _notices(app)))
        assert not _is_prompt(app)
        assert [name for name, _ in launch.connector.connects] == ["db"]
    assert prepare(launch.workspace, launch.global_path, launch.store).untrusted == frozenset()


async def test_after_trusting_a_relaunch_does_not_ask_and_starts_the_servers(launch: Launch) -> None:
    launch.write_project({"db": {"command": "psql"}})
    first = launch.app()
    async with first.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(first))
        await _answer(pilot, "trust")
        await until(pilot, lambda: launch.hub.states()[0].status == "ready")

    second = launch.app()
    async with second.run_test() as pilot:
        await until(pilot, lambda: launch.hub.states()[0].status == "ready")
        assert not _is_prompt(second)


async def test_a_trust_that_cannot_be_saved_still_starts_the_servers_and_warns(launch: Launch) -> None:
    launch.write_project({"db": {"command": "psql"}})

    def broken() -> None:
        raise OSError("disk full")

    setup = prepare(launch.workspace, launch.global_path, launch.store)
    launch.hub = McpHub(launch.connector, launch.workspace, setup.servers, 5.0, untrusted=setup.untrusted)
    app = launch.make_app(mcp=launch.hub, mcp_trust=broken)
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await _answer(pilot, "trust")
        await until(pilot, lambda: launch.hub.states()[0].status == "ready")
        assert any("disk full" in n for n in _notices(app))


async def test_markup_in_a_command_is_shown_as_written(launch: Launch) -> None:
    launch.write_project({"db": {"command": "[bold red]x[/", "args": ["[link=http://e]"]}})
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        text = " ".join(str(label.render()) for label in app.screen.query(Label))
        assert "[bold red]x[/" in text


async def test_mcp_lists_every_server_with_source_status_reason_and_tools(make_app: AppFactory, tmp_path: Path) -> None:
    connector = FakeConnector(
        {"notes": FakeConnection([ECHO, QUERY]), "empty": FakeConnection()},
        failures={"bad": McpError("command not found: nope")},
    )
    connector.gates["slow"] = asyncio.Event()
    configs = [
        ServerConfig("notes", "n"),
        ServerConfig("bad", "nope"),
        ServerConfig("empty", "e"),
        ServerConfig("slow", "s"),
        ServerConfig("db", "psql", source="project"),
    ]
    hub = McpHub(connector, tmp_path.resolve(), configs, 5.0, untrusted={"db"})
    app = make_app(mcp=hub)
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await _wait_for_guard(pilot)
        await pilot.press("escape")  # decline: db stays untrusted
        await until(pilot, lambda: not _is_prompt(app))
        await until(pilot, lambda: [s.status for s in hub.states()] == ["ready", "failed", "ready", "starting", "untrusted"])
        await type_and_send(pilot, "/mcp")
        await until(pilot, lambda: any("notes (global) ready" in n for n in _notices(app)))
        (notice,) = [n for n in _notices(app) if "notes (global) ready" in n]
        assert notice.split(" ", 1)[1].splitlines() == [  # after the level icon
            "notes (global) ready",
            "  mcp__notes__echo",
            "  mcp__notes__query",
            "bad (global) failed: command not found: nope",
            "empty (global) ready",
            "slow (global) starting",
            "db (project) untrusted",
        ]


async def test_mcp_without_servers_says_so(make_app: AppFactory, tmp_path: Path) -> None:
    app = make_app(mcp=McpHub(FakeConnector(), tmp_path.resolve(), [], 5.0))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/mcp")
        await until(pilot, lambda: any("No MCP servers" in n for n in _notices(app)))


ESCAPED = [
    ("\x1b", "\\x1b"),
    ("\r", "\\x0d"),
    ("\x07", "\\x07"),
    ("\n", "\\x0a"),
    ("\x00", "\\x00"),
    ("\x7f", "\\x7f"),
    ("\u202e", "\\u202e"),
    ("\u202a", "\\u202a"),
    ("\u202d", "\\u202d"),
    ("\u2066", "\\u2066"),
    ("\u2069", "\\u2069"),
    ("\u200b", "\\u200b"),
    ("é", "\\xe9"),
    ("я", "\\u044f"),
    ("\U0001f600", "\\U0001f600"),
]


@pytest.mark.parametrize("where", ["name", "command", "arg", "env"])
def test_every_non_printable_or_non_ascii_character_is_escaped_in_the_prompt(where: str) -> None:
    for char, shown in ESCAPED:
        bad = f"a{char}b"
        server = ServerConfig(
            bad if where == "name" else "n",
            bad if where == "command" else "cmd",
            (bad if where == "arg" else "arg",),
            {bad: "v"} if where == "env" else {},
            "project",
        )

        plain = describe(server).plain

        assert char not in plain, repr(char)
        assert f"a{shown}b" in plain, repr(char)
        assert all(c.isascii() and (c.isprintable() or c == "\n") for c in plain), repr(char)  # only our own line break


async def test_declining_over_a_shadowed_global_server_starts_neither_and_mcp_says_why(launch: Launch) -> None:
    launch.write_global({"db": {"command": "global-db"}})
    launch.write_project({"db": {"command": "project-db"}})
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await _answer(pilot, "decline")
        await until(pilot, lambda: not _is_prompt(app))
        await type_and_send(pilot, "/mcp")
        await until(pilot, lambda: any("db (project)" in n for n in _notices(app)))
        (notice,) = [n for n in _notices(app) if "db (project)" in n]
        assert notice.split(" ", 1)[1] == "db (project) untrusted · overrides the global 'db'"
        assert launch.connector.connects == []  # the global one is replaced, so it does not run either
