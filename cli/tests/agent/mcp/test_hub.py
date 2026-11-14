from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from pathlib import Path

import pytest
from src.agent.mcp.config import ServerConfig
from src.agent.mcp.contract import CallResult, McpError, McpExited, McpTimeout, ToolInfo
from src.agent.mcp.hub import HubEvent, McpHub

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


async def test_nothing_starts_until_start_is_called() -> None:
    connector = FakeConnector()
    hub, _ = make_hub(connector, "a")
    await asyncio.sleep(0.05)
    assert connector.connects == []
    assert [(s.config.name, s.status) for s in hub.states()] == [("a", "starting")]
    await hub.close_all()


async def test_start_returns_at_once_while_a_server_is_still_starting() -> None:
    connector = FakeConnector({"a": FakeConnection([tool("echo")])})
    connector.gates["a"] = asyncio.Event()
    hub, events = make_hub(connector, "a")
    hub.start()
    await wait_until(lambda: connector.connects != [])
    assert hub.tools() == []
    assert hub.states()[0].status == "starting"
    connector.gates["a"].set()
    await ready(hub, events)
    assert [t.name for t in hub.tools()] == ["mcp__a__echo"]
    await hub.close_all()


async def test_a_server_is_started_in_the_workspace_with_its_config() -> None:
    connector = FakeConnector({"a": FakeConnection()})
    hub, events = make_hub(connector, "a", workspace=Path("/work"))
    hub.start()
    await ready(hub, events)
    assert connector.connects == [("a", Path("/work"))]
    assert connector.configs == servers("a")
    await hub.close_all()


async def test_a_ready_server_emits_an_event_with_its_tool_count() -> None:
    connector = FakeConnector({"a": FakeConnection([tool("one"), tool("two")]), "b": FakeConnection([tool("x")])})
    hub, events = make_hub(connector, "a", "b")
    hub.start()
    await ready(hub, events, 2)
    assert sorted(events) == [("ready", "a", "2 tools"), ("ready", "b", "1 tool")]
    await hub.close_all()


async def test_tools_carry_the_servers_description_and_schema() -> None:
    connector = FakeConnector({"fake": FakeConnection([tool("echo", "Echo it back")])})
    hub, events = make_hub(connector, "fake")
    hub.start()
    await ready(hub, events)
    (echo,) = hub.tools()
    spec = echo.spec()
    assert (spec.name, spec.description, spec.parameters) == ("mcp__fake__echo", "Echo it back", SCHEMA)
    await hub.close_all()


async def test_names_are_cleaned_and_a_second_tool_with_the_same_name_is_skipped_with_one_warning() -> None:
    connector = FakeConnector({"my.srv": FakeConnection([tool("do/thing"), tool("do_thing"), tool("other")])})
    hub, events = make_hub(connector, "my.srv")
    hub.start()
    await ready(hub, events)
    assert [t.name for t in hub.tools()] == ["mcp__my_srv__do_thing", "mcp__my_srv__other"]
    assert [t.name for t in hub.tools()] == ["mcp__my_srv__do_thing", "mcp__my_srv__other"]
    warns = [e for e in events if e[0] == "warn"]
    assert len(warns) == 1
    assert warns[0][1] == "my.srv"
    assert "do_thing" in warns[0][2]
    assert "mcp__my_srv__do_thing" in warns[0][2]
    await hub.close_all()


async def test_a_collision_between_two_servers_keeps_the_first_in_config_order() -> None:
    connector = FakeConnector({"a-b": FakeConnection([tool("t")]), "a_b": FakeConnection([tool("t")])})
    hub, events = make_hub(connector, "a-b", "a_b")
    hub.start()
    await ready(hub, events, 2)
    assert [t.name for t in hub.tools()] == ["mcp__a-b__t", "mcp__a_b__t"]  # distinct names: "-" is allowed
    await hub.close_all()


async def test_a_server_with_no_tools_is_ready_with_none() -> None:
    connector = FakeConnector({"a": FakeConnection()})
    hub, events = make_hub(connector, "a")
    hub.start()
    await ready(hub, events)
    assert hub.states()[0].status == "ready"
    assert hub.tools() == []
    await hub.close_all()


async def test_a_server_that_fails_to_start_is_failed_with_the_reason_and_has_no_tools() -> None:
    connector = FakeConnector({"good": FakeConnection([tool("t")])}, failures={"bad": McpError("command not found: x")})
    hub, events = make_hub(connector, "bad", "good")
    hub.start()
    await ready(hub, events, 2)
    assert ("failed", "bad", "command not found: x") in events
    states = {s.config.name: s for s in hub.states()}
    assert (states["bad"].status, states["bad"].reason) == ("failed", "command not found: x")
    assert [t.name for t in hub.tools()] == ["mcp__good__t"]
    await hub.close_all()


async def test_an_unexpected_error_from_the_connector_is_a_failure_not_a_crash() -> None:
    connector = FakeConnector(failures={"a": RuntimeError("bug")})
    hub, events = make_hub(connector, "a")
    hub.start()
    await ready(hub, events)
    assert events == [("failed", "a", "bug")]
    await hub.close_all()


async def test_a_server_whose_tool_list_fails_is_failed_and_closed() -> None:
    class NoList(FakeConnection):
        async def list_tools(self) -> list[ToolInfo]:
            raise McpError("list failed")

    connection = NoList()
    hub, events = make_hub(FakeConnector({"a": connection}), "a")
    hub.start()
    await ready(hub, events)
    assert events == [("failed", "a", "list failed")]
    assert connection.closed == 1
    await hub.close_all()


async def test_a_server_that_exits_is_marked_exited_warned_and_its_tools_are_removed() -> None:
    connection = FakeConnection([tool("t")])
    hub, events = make_hub(FakeConnector({"a": connection}), "a")
    hub.start()
    await ready(hub, events)
    assert len(hub.tools()) == 1
    connection.trigger_exit("the server exited")
    assert ("exited", "a", "the server exited") in events
    state = hub.states()[0]
    assert (state.status, state.reason) == ("exited", "the server exited")
    assert hub.tools() == []
    await hub.close_all()


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


async def test_a_call_goes_to_the_connection_with_the_timeout() -> None:
    connection = FakeConnection([tool("t")], {"t": CallResult(["done"], False)})
    hub, events = make_hub(FakeConnector({"a": connection}), "a", timeout=7.0)
    hub.start()
    await ready(hub, events)
    result = await hub.call("a", "t", {"text": "x"})
    assert result == CallResult(["done"], False)
    assert connection.calls == [("t", {"text": "x"}, 7.0)]
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


async def test_a_cancelled_call_cancels_the_work_on_the_connection() -> None:
    connection = FakeConnection([tool("t")])
    connection.call_gate = asyncio.Event()
    hub, events = make_hub(FakeConnector({"a": connection}), "a")
    hub.start()
    await ready(hub, events)
    call = asyncio.ensure_future(hub.call("a", "t", {}))
    await wait_until(lambda: connection.calls != [])
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    await hub.close_all()


async def test_close_all_closes_every_connection_and_is_idempotent() -> None:
    a, b = FakeConnection([tool("t")]), FakeConnection()
    hub, events = make_hub(FakeConnector({"a": a, "b": b}), "a", "b")
    hub.start()
    await ready(hub, events, 2)
    await hub.close_all()
    await hub.close_all()
    assert (a.closed, b.closed) == (1, 1)
    assert hub.tools() == []


async def test_close_all_cancels_a_server_that_is_still_starting() -> None:
    connector = FakeConnector({"a": FakeConnection()})
    connector.gates["a"] = asyncio.Event()
    hub, events = make_hub(connector, "a")
    hub.start()
    await wait_until(lambda: connector.connects != [])
    await asyncio.wait_for(hub.close_all(), 2)
    assert events == []


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


async def test_a_close_that_raises_does_not_stop_the_others() -> None:
    class Stuck(FakeConnection):
        async def aclose(self) -> None:
            raise OSError("stuck")

    a, b = Stuck(), FakeConnection()
    hub, events = make_hub(FakeConnector({"a": a, "b": b}), "a", "b")
    hub.start()
    await ready(hub, events, 2)
    await hub.close_all()
    assert b.closed == 1


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


async def test_warnings_from_the_config_reach_the_first_listener_only() -> None:
    hub = McpHub(FakeConnector(), Path("/ws"), [], 1.0, ["~/x/mcp.json: invalid JSON: boom"])
    first: list[HubEvent] = []
    second: list[HubEvent] = []
    hub.listen(first.append)
    hub.listen(second.append)
    assert first == [("warn", "", "~/x/mcp.json: invalid JSON: boom")]
    assert second == []
    await hub.close_all()


async def test_events_before_a_listener_exists_are_replayed_to_it() -> None:
    connector = FakeConnector({"a": FakeConnection([tool("t")])})
    hub = McpHub(connector, Path("/ws"), servers("a"), 1.0)
    hub.start()
    await wait_until(lambda: hub.states()[0].status == "ready")
    events: list[HubEvent] = []
    hub.listen(events.append)
    assert events == [("ready", "a", "1 tool")]
    await hub.close_all()


async def test_a_listener_that_raises_does_not_break_the_hub() -> None:
    connector = FakeConnector({"a": FakeConnection([tool("t")])})
    hub = McpHub(connector, Path("/ws"), servers("a"), 1.0)

    def boom(event: HubEvent) -> None:
        raise RuntimeError("ui gone")

    hub.listen(boom)
    hub.start()
    await wait_until(lambda: hub.states()[0].status == "ready")
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


async def test_a_server_that_never_finishes_connecting_fails_after_the_timeout() -> None:
    connector = FakeConnector({"a": FakeConnection()})
    connector.gates["a"] = asyncio.Event()
    hub, events = make_hub(connector, "a", start_timeout=0.1)
    hub.start()
    await ready(hub, events)
    assert events == [("failed", "a", "timed out starting after 0.1s")]
    assert hub.states()[0].status == "failed"
    assert hub.tools() == []
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


async def test_a_slow_but_in_time_start_is_not_a_failure() -> None:
    connector = FakeConnector({"a": FakeConnection([tool("t")])}, delays={"a": 0.05})
    hub, events = make_hub(connector, "a", start_timeout=2.0)
    hub.start()
    await ready(hub, events)
    assert events == [("ready", "a", "1 tool")]
    await hub.close_all()


async def test_kill_all_asks_the_connector_too_so_servers_still_starting_are_killed() -> None:
    connector = FakeConnector()
    connector.gates["a"] = asyncio.Event()
    hub, _ = make_hub(connector, "a")
    hub.start()
    await wait_until(lambda: connector.connects != [])
    hub.kill_all()
    assert connector.kill_alls == 1
    await hub.close_all()


async def test_starting_uses_the_start_timeout_and_calls_use_the_call_timeout() -> None:
    connection = FakeConnection([tool("t")])
    connector = FakeConnector({"a": connection}, delays={"a": 0.3})  # longer than the call timeout
    hub, events = make_hub(connector, "a", timeout=0.05, start_timeout=5.0)
    hub.start()
    await ready(hub, events)
    assert events == [("ready", "a", "1 tool")]  # the call timeout did not cut the start short
    await hub.call("a", "t", {})
    assert connection.calls[0][2] == 0.05  # and the call got the call timeout, not the start one
    await hub.close_all()


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


async def test_a_connector_that_raises_a_bare_timeout_error_fails_the_server() -> None:
    connector = FakeConnector(failures={"a": TimeoutError()})
    hub, events = make_hub(connector, "a", start_timeout=60.0)
    hub.start()
    await ready(hub, events)
    assert events == [("failed", "a", "TimeoutError")]
    await hub.close_all()
