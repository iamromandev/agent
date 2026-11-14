"""Project MCP servers: nothing starts until the user trusts the project file; `/mcp` lists every server."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from src.agent.mcp.config import ServerConfig
from src.agent.mcp.contract import McpError, ToolInfo
from src.agent.mcp.hub import McpHub
from src.agent.mcp.setup import McpSetup, prepare
from src.agent.mcp.trust import TrustStore, trust_store_path
from src.tui.app import HELP, AgentApp
from src.tui.screens.mcp_trust import McpTrustScreen, describe
from src.tui.widgets.transcript import NoticeEntry
from textual.containers import VerticalScroll
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


async def test_an_untrusted_project_file_asks_listing_each_server_name_and_full_command_line(launch: Launch) -> None:
    launch.write_project(
        {
            "db": {"command": "psql", "args": ["-h", "my host", "--flag"], "env": {"PGPASSWORD": "x"}},
            "web": {"command": "npx"},
        }
    )
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        text = " ".join(str(label.render()) for label in app.screen.query(Label))
        assert "db" in text
        assert "psql -h 'my host' --flag" in text
        assert "PGPASSWORD" in text
        assert "web" in text
        assert "npx" in text


async def test_nothing_is_spawned_while_the_project_servers_are_untrusted(launch: Launch) -> None:
    launch.write_global({"notes": {"command": "n"}})
    launch.write_project({"db": {"command": "psql"}})
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await until(pilot, lambda: any("mcp notes ready" in n for n in _notices(app)))
        await pilot.pause(0.1)
        assert [name for name, _ in launch.connector.connects] == ["notes"]  # NFR3
        assert [s.status for s in launch.hub.states()] == ["ready", "untrusted"]


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


async def test_declining_keeps_the_servers_stopped_and_the_agent_usable(launch: Launch) -> None:
    launch.write_project({"db": {"command": "psql"}})
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await _answer(pilot, "decline")
        await until(pilot, lambda: any("not trusted" in n and "db" in n for n in _notices(app)))
        assert not _is_prompt(app)
        await type_and_send(pilot, "/help")
        await until(pilot, lambda: any("/quit" in n for n in _notices(app)))
        assert launch.connector.connects == []
        assert launch.hub.states()[0].status == "untrusted"
        assert launch.hub.tools() == []
    assert launch.store.is_trusted(launch.workspace, (launch.workspace / ".agent" / "mcp.json").read_bytes()) is False


async def test_escape_declines(launch: Launch) -> None:
    launch.write_project({"db": {"command": "psql"}})
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await _wait_for_guard(pilot)
        await pilot.press("escape")
        await until(pilot, lambda: not _is_prompt(app))
        await pilot.pause(0.1)
        assert launch.connector.connects == []


async def test_a_stray_enter_declines(launch: Launch) -> None:
    launch.write_project({"db": {"command": "psql"}})
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await _wait_for_guard(pilot)
        await pilot.press("enter")
        await until(pilot, lambda: not _is_prompt(app))
        await pilot.pause(0.1)
        assert launch.connector.connects == []


async def test_declining_asks_again_on_the_next_launch(launch: Launch) -> None:
    launch.write_project({"db": {"command": "psql"}})
    first = launch.app()
    async with first.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(first))
        await _answer(pilot, "decline")
        await until(pilot, lambda: not _is_prompt(first))

    second = launch.app()
    async with second.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(second))


async def test_editing_the_project_file_asks_again(launch: Launch) -> None:
    project = launch.write_project({"db": {"command": "psql"}})
    first = launch.app()
    async with first.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(first))
        await _answer(pilot, "trust")
        await until(pilot, lambda: launch.hub.states()[0].status == "ready")
    launch.connector.connects.clear()

    launch.write_project({"db": {"command": "psql", "args": ["--changed"]}})
    assert project.exists()
    second = launch.app()
    async with second.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(second))
        await pilot.pause(0.1)
        assert launch.connector.connects == []


async def test_only_global_servers_do_not_ask(launch: Launch) -> None:
    launch.write_global({"notes": {"command": "n"}})
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: launch.hub.states()[0].status == "ready")
        assert not _is_prompt(app)


async def test_a_project_server_replaces_the_global_one_of_the_same_name(launch: Launch) -> None:
    launch.write_global({"db": {"command": "global-db"}})
    launch.write_project({"db": {"command": "project-db"}})
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await _answer(pilot, "trust")
        await until(pilot, lambda: launch.hub.states()[0].status == "ready")
        assert [(c.command, c.source) for c in launch.connector.configs] == [("project-db", "project")]


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


async def test_quitting_with_the_prompt_open_spawns_nothing(launch: Launch) -> None:
    launch.write_project({"db": {"command": "psql"}})
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
    assert launch.connector.connects == []


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


async def test_mcp_without_a_hub_says_so(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/mcp")
        await until(pilot, lambda: any("No MCP servers" in n for n in _notices(app)))


async def test_the_options_start_on_dont_trust_and_a_few_stray_keys_right_after_opening_do_not_trust(
    launch: Launch,
) -> None:
    launch.write_project({"db": {"command": "psql"}})
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        options = app.screen.query_one(OptionList)
        assert [options.get_option_at_index(i).id for i in range(options.option_count)] == ["decline", "trust"]

        await pilot.press("up", "enter")  # typed for something else, just as the prompt appeared
        await pilot.pause(0.1)

        assert _is_prompt(app)  # ignored: still open
        assert launch.connector.connects == []
        assert launch.store.is_trusted(launch.workspace, (launch.workspace / ".agent" / "mcp.json").read_bytes()) is False


async def test_keys_work_again_once_the_guard_has_passed(launch: Launch) -> None:
    launch.write_project({"db": {"command": "psql"}})
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await _wait_for_guard(pilot)
        await pilot.press("up", "enter")  # Up wraps to "Trust and start them": a deliberate choice now
        await until(pilot, lambda: launch.hub.states()[0].status == "ready")


async def test_the_prompt_warns_that_files_the_commands_run_are_not_checked(launch: Launch) -> None:
    launch.write_project({"db": {"command": "psql"}})
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        text = " ".join(str(label.render()) for label in app.screen.query(Label))
        assert "scripts, packages" in text
        assert "not checked" in text


@pytest.mark.parametrize(
    ("char", "shown"),
    [
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
    ],
)
@pytest.mark.parametrize("where", ["name", "command", "arg", "env"])
def test_every_non_printable_or_non_ascii_character_is_escaped_in_the_prompt(char: str, shown: str, where: str) -> None:
    bad = f"a{char}b"
    server = ServerConfig(
        bad if where == "name" else "n",
        bad if where == "command" else "cmd",
        (bad if where == "arg" else "arg",),
        {bad: "v"} if where == "env" else {},
        "project",
    )

    plain = describe(server).plain

    assert char not in plain
    assert f"a{shown}b" in plain
    assert all(c.isascii() and (c.isprintable() or c == "\n") for c in plain)  # only our own line break remains


def test_ordinary_text_is_shown_unchanged() -> None:
    plain = describe(ServerConfig("db", "psql", ("-h", "my host"), {"PGUSER": "u"}, "project")).plain
    assert "db" in plain
    assert "psql -h 'my host'" in plain
    assert "PGUSER" in plain


async def test_a_long_list_of_servers_can_be_scrolled_with_the_keyboard_to_the_last_one(launch: Launch) -> None:
    launch.write_project({f"srv{i:02d}": {"command": f"cmd{i:02d}", "args": ["--flag", "x" * 20]} for i in range(30)})
    app = launch.app()
    async with app.run_test(size=(100, 40)) as pilot:
        await until(pilot, lambda: _is_prompt(app))
        scroller = app.screen.query_one("#servers", VerticalScroll)
        await until(pilot, lambda: scroller.max_scroll_y > 0)  # the list does not fit: it is capped (layout is async)
        assert scroller.scroll_y == 0

        await _wait_for_guard(pilot)
        await pilot.press("tab")  # to the list
        assert app.screen.focused is scroller
        await pilot.press("end")

        assert scroller.scroll_y == scroller.max_scroll_y
        last = [label for label in scroller.query(Label) if "srv29" in str(label.render())]
        assert last, "the last server is in the list"
        visible = scroller.region
        assert last[0].region.y + last[0].region.height <= visible.y + visible.height + 1  # in view at the end


async def test_shift_tab_goes_back_to_the_options_after_scrolling(launch: Launch) -> None:
    launch.write_project({f"srv{i:02d}": {"command": "c"} for i in range(30)})
    app = launch.app()
    async with app.run_test(size=(100, 40)) as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await _wait_for_guard(pilot)
        await pilot.press("tab")
        assert isinstance(app.screen.focused, VerticalScroll)
        await pilot.press("shift+tab")
        assert isinstance(app.screen.focused, OptionList)


async def test_a_project_server_that_replaces_a_global_one_says_so_in_the_prompt(launch: Launch) -> None:
    launch.write_global({"db": {"command": "global-db"}})
    launch.write_project({"db": {"command": "project-db"}, "web": {"command": "w"}})
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        labels = [str(label.render()) for label in app.screen.query("#servers Label")]
        (db,) = [t for t in labels if "project-db" in t]
        (web,) = [t for t in labels if "w" in t.split()]
        assert "overrides the global 'db'" in db
        assert "overrides" not in web


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


def test_help_lists_the_mcp_command() -> None:
    assert "/mcp" in HELP


@pytest.mark.parametrize("doc", ["README.md", "AGENTS.md"])
def test_the_docs_describe_project_config_trust_and_the_mcp_command(doc: str) -> None:
    text = (Path(__file__).resolve().parents[3] / doc).read_text()

    missing = [n for n in (".agent/mcp.json", "mcp-trust.json", "/mcp", "trust") if n not in text]

    assert missing == []
