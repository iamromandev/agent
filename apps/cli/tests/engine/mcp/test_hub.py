from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from pathlib import Path

import pytest
from src.engine.mcp.config import ServerConfig
from src.engine.mcp.contract import McpError, McpExited, McpTimeout, ToolInfo
from src.engine.mcp.hub import HubEvent, McpHub

from tests.fakes import FakeConnection, FakeConnector
from tests.helpers import wait_until

SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


def tool(name: str, description: str = "") -> ToolInfo:
    return ToolInfo(name, description or f"the {name} tool", SCHEMA)


def servers(*names: str) -> list[ServerConfig]:
    return [ServerConfig(name, "cmd") for name in names]


def make_hub(
    connector: FakeConnector,
    *names: str,
    workspace: Path = Path("/ws"),
    timeout: float = 7.0,
    start_timeout: float = 120.0,
    warnings: Sequence[str] = (),
) -> tuple[McpHub, list[HubEvent]]:
    hub = McpHub(connector, workspace, servers(*names), timeout, warnings, start_timeout=start_timeout)
    events: list[HubEvent] = []
    hub.listen(events.append)
    return hub, events


async def ready(hub: McpHub, events: list[HubEvent], count: int = 1) -> None:
    await wait_until(lambda: sum(e[0] in ("ready", "failed") for e in events) >= count)


async def test_a_call_in_flight_when_the_server_exits_fails_with_mcp_exited() -> None:
    connection = FakeConnection([tool("t")])
    connection.call_gate = asyncio.Event()  # the call never answers
    hub, events = make_hub(FakeConnector({"a": connection}), "a")
    hub.start()
    await ready(hub, events)
    call = asyncio.ensure_future(hub.call("a", "t", {"text": "x"}))
    await wait_until(lambda: connection.calls != [])
    connection.trigger_exit("the server exited")
    with pytest.raises(McpExited, match="the server exited"):
        await asyncio.wait_for(call, 2)
    await hub.close_all()


async def test_a_timeout_is_passed_on_and_the_server_stays_ready() -> None:
    connection = FakeConnection([tool("t")], {"t": McpTimeout("timeout after 7s")})
    hub, events = make_hub(FakeConnector({"a": connection}), "a")
    hub.start()
    await ready(hub, events)
    with pytest.raises(McpTimeout):
        await hub.call("a", "t", {})
    assert hub.states()[0].status == "ready"
    await hub.close_all()


async def test_a_call_to_a_server_that_is_not_ready_fails_without_connecting() -> None:
    connector = FakeConnector(failures={"a": McpError("nope")})
    hub, _ = make_hub(connector, "a")
    with pytest.raises(McpError, match='"a" is not running'):
        await hub.call("a", "t", {})
    with pytest.raises(McpError, match='"zzz" is not running'):
        await hub.call("zzz", "t", {})
    await hub.close_all()


async def test_a_connection_that_arrives_after_close_all_is_closed() -> None:
    late = FakeConnection()

    class Late(FakeConnector):
        async def connect(self, config: ServerConfig, cwd: Path) -> FakeConnection:
            with contextlib.suppress(asyncio.CancelledError):  # a connector that swallows the cancel...
                await asyncio.sleep(3600)
            return late  # ...and still hands a process over

    hub, _ = make_hub(Late(), "a")
    hub.start()
    await asyncio.sleep(0.02)
    await hub.close_all()
    assert late.closed == 1


async def test_start_after_close_all_does_nothing() -> None:
    connector = FakeConnector()
    hub, _ = make_hub(connector, "a")
    await hub.close_all()
    hub.start()
    await asyncio.sleep(0.02)
    assert connector.connects == []


async def test_start_with_names_starts_only_those_and_never_twice() -> None:
    connector = FakeConnector({"a": FakeConnection(), "b": FakeConnection()})
    hub, events = make_hub(connector, "a", "b")
    hub.start(["b"])
    await ready(hub, events)
    hub.start(["b"])
    hub.start()
    await ready(hub, events, 2)
    assert sorted(name for name, _ in connector.connects) == ["a", "b"]
    await hub.close_all()


async def test_kill_all_kills_every_connection_synchronously() -> None:
    a, b = FakeConnection(), FakeConnection()
    hub, events = make_hub(FakeConnector({"a": a, "b": b}), "a", "b")
    hub.start()
    await ready(hub, events, 2)
    hub.kill_all()
    assert (a.killed, b.killed) == (1, 1)
    await hub.close_all()


async def test_a_server_that_exits_while_its_tools_are_being_listed_is_exited_and_never_ready() -> None:
    gate = asyncio.Event()

    class SlowList(FakeConnection):
        async def list_tools(self) -> list[ToolInfo]:
            await gate.wait()
            return [tool("t")]

    connection = SlowList()
    connector = FakeConnector({"a": connection})
    hub, events = make_hub(connector, "a")
    hub.start()
    await wait_until(lambda: connector.connects != [])
    await wait_until(lambda: hub.states()[0].status == "starting" and connection._exit_listeners != [])
    connection.trigger_exit("the server exited")
    assert events == [("exited", "a", "the server exited")]
    gate.set()
    await asyncio.sleep(0.05)
    assert hub.states()[0].status == "exited"
    assert hub.tools() == []
    assert [e[0] for e in events] == ["exited"]  # no "ready", and no second event
    await hub.close_all()


async def test_an_exit_followed_by_a_failing_tool_list_is_reported_once() -> None:
    gate = asyncio.Event()

    class FailingList(FakeConnection):
        async def list_tools(self) -> list[ToolInfo]:
            await gate.wait()
            raise McpExited("the server exited")

    connection = FailingList()
    hub, events = make_hub(FakeConnector({"a": connection}), "a")
    hub.start()
    await wait_until(lambda: connection._exit_listeners != [])
    connection.trigger_exit("the server exited")
    gate.set()
    await asyncio.sleep(0.05)
    assert events == [("exited", "a", "the server exited")]
    assert connection.closed == 1
    await hub.close_all()


async def test_a_server_whose_tool_list_never_arrives_fails_after_the_timeout_and_is_closed() -> None:
    class HangingList(FakeConnection):
        async def list_tools(self) -> list[ToolInfo]:
            await asyncio.sleep(3600)
            return []

    connection = HangingList()
    hub, events = make_hub(FakeConnector({"a": connection}), "a", start_timeout=0.1)
    hub.start()
    await ready(hub, events)
    assert events == [("failed", "a", "timed out starting after 0.1s")]
    assert connection.closed == 1
    await hub.close_all()
    assert connection.closed == 1  # not closed a second time: it left the hub's list


async def test_a_timeout_error_that_is_not_the_start_deadline_fails_the_server_and_closes_it() -> None:
    class TimesOutItself(FakeConnection):
        async def list_tools(self) -> list[ToolInfo]:
            raise TimeoutError("socket timed out")

    connection = TimesOutItself()
    hub, events = make_hub(FakeConnector({"a": connection}), "a", start_timeout=60.0)
    hub.start()
    await ready(hub, events)
    assert events == [("failed", "a", "socket timed out")]
    assert hub.states()[0].status == "failed"
    assert connection.closed == 1
    await hub.close_all()


def make_untrusted_hub(connector: FakeConnector, untrusted: frozenset[str] = frozenset({"db"})) -> tuple[McpHub, list[HubEvent]]:
    configs = [ServerConfig("notes", "cmd"), ServerConfig("db", "psql", ("-h", "x"), source="project")]
    hub = McpHub(connector, Path("/ws"), configs, 7.0, untrusted=untrusted)
    events: list[HubEvent] = []
    hub.listen(events.append)
    return hub, events
