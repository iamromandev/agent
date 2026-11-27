"""`McpHub.stop`: individual servers stopped on purpose (the config watch), their tools withdrawn."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from core.engine.mcp.config import ServerConfig
from core.engine.mcp.contract import McpExited, ToolInfo
from core.engine.mcp.hub import HubEvent, McpHub
from core.testing.fakes import FakeConnection, FakeConnector
from core.testing.helpers import wait_until

SCHEMA = {"type": "object", "properties": {}}


def _hub(connector: FakeConnector) -> tuple[McpHub, list[HubEvent]]:
    servers = [ServerConfig("a", "cmd", source="project"), ServerConfig("b", "cmd", source="global")]
    hub = McpHub(connector, Path("/ws"), servers, 7.0)
    events: list[HubEvent] = []
    hub.listen(events.append)
    return hub, events


async def _ready(hub: McpHub, events: list[HubEvent]) -> None:
    hub.start()
    await wait_until(lambda: sum(e[0] == "ready" for e in events) == 2)


async def test_stop_closes_one_server_withdraws_its_tools_and_keeps_the_other() -> None:
    a, b = FakeConnection([ToolInfo("t", "t", SCHEMA)]), FakeConnection([ToolInfo("u", "u", SCHEMA)])
    hub, events = _hub(FakeConnector({"a": a, "b": b}))
    await _ready(hub, events)

    await hub.stop(["a"], "config changed")

    assert a.closed == 1 and b.closed == 0
    assert [t.name for t in hub.tools()] == ["mcp__b__u"]
    state = {s.config.name: s for s in hub.states()}
    assert state["a"].status == "stopped" and "config changed" in state["a"].reason
    assert any(e[0] == "exited" and e[1] == "a" for e in events)
    await hub.close_all()


async def test_stop_fails_a_call_in_flight_and_refuses_new_ones() -> None:
    a = FakeConnection([ToolInfo("t", "t", SCHEMA)])
    a.call_gate = asyncio.Event()
    hub, events = _hub(FakeConnector({"a": a, "b": FakeConnection()}))
    await _ready(hub, events)
    call = asyncio.ensure_future(hub.call("a", "t", {}))
    await wait_until(lambda: a.calls != [])

    await hub.stop(["a"], "config changed")

    with pytest.raises(McpExited, match="config changed"):
        await asyncio.wait_for(call, 2)
    with pytest.raises(Exception, match="not running"):
        await hub.call("a", "t", {})
    await hub.close_all()


async def test_a_stopped_server_is_not_restarted_by_start() -> None:
    connector = FakeConnector({"a": FakeConnection(), "b": FakeConnection()})
    hub, events = _hub(connector)
    await _ready(hub, events)
    await hub.stop(["a"], "config changed")

    hub.start()
    await asyncio.sleep(0)

    assert [c for c, _ in connector.connects].count("a") == 1
    await hub.close_all()


async def test_stop_of_an_unknown_server_is_a_no_op() -> None:
    hub = McpHub(FakeConnector(), Path("/ws"), [ServerConfig("a", "cmd")], 7.0)

    await hub.stop(["zzz"], "config changed")

    assert hub.states()[0].status == "starting"


async def test_a_stopped_untrusted_server_cannot_be_trusted_and_started_later() -> None:
    connector = FakeConnector({"a": FakeConnection()})
    hub = McpHub(connector, Path("/ws"), [ServerConfig("a", "cmd", source="project")], 7.0, untrusted=["a"])

    await hub.stop(["a"], "config changed")
    hub.start(["a"])  # the trust dialog's yes, after the revoke
    await asyncio.sleep(0.05)

    assert hub.states()[0].status == "stopped" and hub.untrusted() == []
    assert connector.connects == []


async def test_a_server_stopped_before_the_hub_started_is_never_connected() -> None:
    connector = FakeConnector({"a": FakeConnection(), "b": FakeConnection()})
    hub, _ = _hub(connector)

    await hub.stop(["a"], "config changed")  # a revoke before the app starts the hub
    hub.start()
    await asyncio.sleep(0.05)

    assert [c for c, _ in connector.connects] == ["b"]
    await hub.close_all()


async def test_withdraw_takes_the_tools_away_at_once_and_stop_closes_later() -> None:
    a = FakeConnection([ToolInfo("t", "t", SCHEMA)])
    hub, events = _hub(FakeConnector({"a": a, "b": FakeConnection()}))
    await _ready(hub, events)

    assert hub.withdraw(["a"], "config changed") == ["a"]

    assert hub.tools() == [] and hub.states()[0].status == "stopped"
    assert a.closed == 0
    await hub.stop(["a"], "config changed")
    assert a.closed == 1
    assert sum(e[0] == "exited" and e[1] == "a" for e in events) == 1
    await hub.close_all()


async def test_stop_while_a_server_is_still_starting_cancels_the_start() -> None:
    connector = FakeConnector({"a": FakeConnection(), "b": FakeConnection()})
    connector.gates["a"] = asyncio.Event()
    hub, _ = _hub(connector)
    hub.start()
    await wait_until(lambda: ("a", Path("/ws")) in connector.connects)

    await hub.stop(["a"], "config changed")

    assert {s.config.name: s.status for s in hub.states()}["a"] == "stopped"
    await hub.close_all()


def test_sources_maps_each_server_to_where_it_came_from() -> None:
    hub, _ = _hub(FakeConnector())

    assert hub.sources() == {"a": "project", "b": "global"}


def test_untrusted_can_be_asked_for_one_source_only() -> None:
    servers = [ServerConfig("a", "cmd", source="project"), ServerConfig("b", "cmd", source="global")]
    hub = McpHub(FakeConnector(), Path("/ws"), servers, 7.0, untrusted={"a", "b"})

    assert [s.name for s in hub.untrusted()] == ["a", "b"]
    assert [s.name for s in hub.untrusted("project")] == ["a"]
    assert [s.name for s in hub.untrusted("global")] == ["b"]
