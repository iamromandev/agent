"""MCP servers inside the app: they start in the background, say what happened, and stop with the app."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from src.agent.mcp.config import ServerConfig
from src.agent.mcp.contract import McpError, ToolInfo
from src.agent.mcp.hub import McpHub
from src.agent.provider.base import Done, TextDelta
from src.client.mcp import SdkConnector
from src.tui.app import AgentApp
from src.tui.widgets.transcript import AssistantEntry, NoticeEntry

from tests.client.test_mcp import child_pid, fake_config, with_child
from tests.fakes import FakeConnection, FakeConnector
from tests.helpers import assert_all_gone, wait_until
from tests.tui.conftest import AppFactory, scripted, settled, type_and_send, until

ECHO = ToolInfo("echo", "Echo it", {"type": "object", "properties": {"text": {"type": "string"}}})


def _notices(app: AgentApp) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _hub(
    tmp_path: Path, connector: FakeConnector, *names: str, warnings: list[str] | None = None
) -> McpHub:
    return McpHub(connector, tmp_path.resolve(), [ServerConfig(n, "cmd") for n in names], 5.0, warnings or [])


def _reply(text: str = "ok") -> list:
    return [TextDelta(text), Done(1, 1)]


async def test_the_input_is_usable_while_a_server_is_still_starting_and_its_tools_arrive_for_a_later_turn(
    make_app: AppFactory, tmp_path: Path
) -> None:
    connector = FakeConnector({"fake": FakeConnection([ECHO])})
    connector.gates["fake"] = asyncio.Event()  # the server is "slow to initialize" until the test lets it
    hub = _hub(tmp_path, connector, "fake")
    app = make_app([_reply("one"), _reply("two")], mcp=hub)
    async with app.run_test() as pilot:
        await wait_until(lambda: connector.connects != [])
        assert hub.states()[0].status == "starting"

        await type_and_send(pilot, "first")
        await settled(app)
        assert any("one" in e.text for e in app.query(AssistantEntry))
        assert "mcp__fake__echo" not in scripted(app).requests[0][1]

        connector.gates["fake"].set()
        await until(pilot, lambda: any("mcp fake ready · 1 tool" in n for n in _notices(app)))
        await type_and_send(pilot, "second")
        await settled(app)
        assert "mcp__fake__echo" in scripted(app).requests[1][1]


async def test_a_ready_server_is_announced_with_its_tool_count(make_app: AppFactory, tmp_path: Path) -> None:
    hub = _hub(tmp_path, FakeConnector({"a": FakeConnection([ECHO, ECHO])}), "a")
    app = make_app(mcp=hub)
    async with app.run_test() as pilot:
        await until(pilot, lambda: any("mcp a ready" in n for n in _notices(app)))
        assert any("mcp a ready · 2 tools" in n for n in _notices(app))


async def test_a_failed_server_is_a_warning_that_names_it_and_the_reason(make_app: AppFactory, tmp_path: Path) -> None:
    connector = FakeConnector({"good": FakeConnection([ECHO])}, failures={"bad": McpError("command not found: nope")})
    hub = _hub(tmp_path, connector, "bad", "good")
    app = make_app(mcp=hub)
    async with app.run_test() as pilot:
        await until(pilot, lambda: any("mcp bad" in n for n in _notices(app)) and any("mcp good" in n for n in _notices(app)))
        assert any("mcp bad failed: command not found: nope" in n for n in _notices(app))
        assert any("mcp good ready" in n for n in _notices(app))


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


async def test_a_skipped_duplicate_tool_is_a_warning_naming_the_server(make_app: AppFactory, tmp_path: Path) -> None:
    connection = FakeConnection([ToolInfo("do/thing", "", {}), ToolInfo("do_thing", "", {})])
    hub = _hub(tmp_path, FakeConnector({"my.srv": connection}), "my.srv")
    app = make_app([_reply()], mcp=hub)
    async with app.run_test() as pilot:
        await until(pilot, lambda: any("mcp my.srv ready" in n for n in _notices(app)))
        await type_and_send(pilot, "go")  # the tools are listed when a turn starts
        await settled(app)
        assert any("mcp my.srv: tool \"do_thing\" skipped" in n for n in _notices(app))


async def test_quitting_stops_every_server(make_app: AppFactory, tmp_path: Path) -> None:
    connection = FakeConnection([ECHO])
    hub = _hub(tmp_path, FakeConnector({"a": connection}), "a")
    app = make_app(mcp=hub)
    async with app.run_test() as pilot:
        await until(pilot, lambda: hub.states()[0].status == "ready")
        assert connection.closed == 0
    assert connection.closed == 1


async def test_ctrl_c_when_idle_stops_every_server(make_app: AppFactory, tmp_path: Path) -> None:
    connection = FakeConnection([ECHO])
    hub = _hub(tmp_path, FakeConnector({"a": connection}), "a")
    app = make_app(mcp=hub)
    async with app.run_test() as pilot:
        await until(pilot, lambda: hub.states()[0].status == "ready")
        await pilot.press("ctrl+c")
        await until(pilot, lambda: not app.is_running or app._shutting_down)
    assert connection.closed == 1


async def test_a_server_still_starting_at_quit_is_cancelled(make_app: AppFactory, tmp_path: Path) -> None:
    connector = FakeConnector({"a": FakeConnection()})
    connector.gates["a"] = asyncio.Event()
    hub = _hub(tmp_path, connector, "a")
    app = make_app(mcp=hub)
    async with app.run_test():
        await wait_until(lambda: connector.connects != [])
    assert hub._tasks["a"].done()


async def test_the_servers_are_stopped_even_if_the_app_fails(
    make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = FakeConnection([ECHO])
    hub = _hub(tmp_path, FakeConnector({"a": connection}), "a")
    app = make_app(mcp=hub)

    def broken(self: AgentApp) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(AgentApp, "action_toggle_mode", broken)
    with pytest.raises(RuntimeError, match="boom"):
        async with app.run_test() as pilot:
            await until(pilot, lambda: hub.states()[0].status == "ready")
            await pilot.press("tab")
            await pilot.pause(0.2)
    assert connection.closed == 1


async def test_new_keeps_the_same_server_connection(make_app: AppFactory, tmp_path: Path) -> None:
    connection = FakeConnection([ECHO])
    connector = FakeConnector({"a": connection})
    hub = _hub(tmp_path, connector, "a")
    app = make_app(mcp=hub)
    async with app.run_test() as pilot:
        await until(pilot, lambda: hub.states()[0].status == "ready")
        await type_and_send(pilot, "/new")
        await pilot.pause(0.2)
        await type_and_send(pilot, "/clear")
        await pilot.pause(0.2)
        assert hub.states()[0].status == "ready"
        assert connection.closed == 0
        assert [name for name, _ in connector.connects] == ["a"]
        assert [t.name for t in hub.tools()] == ["mcp__a__echo"]


async def test_a_real_server_keeps_its_pid_across_new_and_is_gone_after_quit(
    make_app: AppFactory, tmp_path: Path
) -> None:
    hub = McpHub(SdkConnector(), tmp_path.resolve(), [fake_config()], 20.0)
    app = make_app(mcp=hub)
    async with app.run_test() as pilot:
        await until(pilot, lambda: hub.states()[0].status == "ready", timeout=15)
        before = (await hub.call("fake", "pid", {})).parts[0].split()[0]
        await type_and_send(pilot, "/new")
        await pilot.pause(0.2)
        after = (await hub.call("fake", "pid", {})).parts[0].split()[0]
        assert before == after
    await assert_all_gone(int(before))


@pytest.mark.parametrize("doc", ["README.md", "AGENTS.md"])
def test_the_docs_describe_the_mcp_config_tool_names_and_timeout(doc: str) -> None:
    text = (Path(__file__).resolve().parents[3] / doc).read_text()

    missing = [n for n in ("mcp.json", "mcp__<server>__<tool>", "AGENT_MCP_TIMEOUT", "AGENT_MCP_START_TIMEOUT", "plan mode") if n not in text]

    assert missing == []


async def test_the_input_works_while_a_real_slow_server_starts_and_its_tools_come_after(
    make_app: AppFactory, tmp_path: Path
) -> None:
    """NFR1 and AC3 with a real process: the server sleeps 5 s before it answers `initialize`."""
    hub = McpHub(SdkConnector(), tmp_path.resolve(), [fake_config("slow_init")], 30.0)
    app = make_app([_reply("one"), _reply("two")], mcp=hub)
    started = time.monotonic()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "first")
        await settled(app)
        assert time.monotonic() - started < 4.5  # the answer came before the server could be ready
        assert hub.states()[0].status == "starting"
        assert any("one" in e.text for e in app.query(AssistantEntry))
        assert "mcp__fake__echo" not in scripted(app).requests[0][1]

        await until(pilot, lambda: hub.states()[0].status == "ready", timeout=20)
        await type_and_send(pilot, "second")
        await settled(app)
        assert "mcp__fake__echo" in scripted(app).requests[1][1]


@pytest.mark.parametrize("how", ["quit", "ctrl+c", "error"])
async def test_every_way_out_stops_the_server_and_what_it_started(
    how: str, make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pidfile = tmp_path / "child.pid"
    hub = McpHub(SdkConnector(), tmp_path.resolve(), [with_child(pidfile)], 20.0)
    app = make_app(mcp=hub)

    def broken(self: AgentApp) -> None:
        raise RuntimeError("boom")

    async def run() -> None:
        async with app.run_test() as pilot:
            await until(pilot, lambda: hub.states()[0].status == "ready", timeout=15)
            server = int((await hub.call("fake", "pid", {})).parts[0].split()[0])
            pids.extend([server, await child_pid(pidfile)])
            if how == "ctrl+c":
                await pilot.press("ctrl+c")
            elif how == "error":
                monkeypatch.setattr(AgentApp, "action_toggle_mode", broken)
                await pilot.press("tab")
                await pilot.pause(0.2)

    pids: list[int] = []
    if how == "error":
        with pytest.raises(RuntimeError, match="boom"):
            await run()
    else:
        await run()

    assert len(pids) == 2
    started = time.monotonic()
    await assert_all_gone(*pids)
    assert time.monotonic() - started < 5
