"""The SDK client against a real stdio server (tests/fixtures/fake_mcp_server.py)."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest
from src.client.mcp import SdkConnection, SdkConnector
from src.engine.mcp.config import ServerConfig
from src.engine.mcp.contract import McpError, McpExited, McpTimeout

from tests.conftest import FIXTURES
from tests.helpers import gone, wait_until

SERVER = FIXTURES / "fake_mcp_server.py"

Open = Callable[..., Awaitable[SdkConnection]]


def with_child(pidfile: Path, mode: str = "normal", quiet: bool = False) -> ServerConfig:
    """The fake server started by a shell that first leaves `sleep 300` running next to it (a grandchild).

    The child keeps the server's stdout open, as a real helper process would, unless `quiet`: then the server's
    output ends when the server does, so its exit is seen."""
    redirect = " >/dev/null 2>&1" if quiet else ""
    script = f'sleep 300{redirect} & echo $! > "$0"; exec "$1" "$2"'
    return ServerConfig(
        "fake", "sh", ("-c", script, str(pidfile), sys.executable, str(SERVER)), {"FAKE_MCP_MODE": mode}
    )


async def child_pid(pidfile: Path) -> int:
    await wait_until(lambda: pidfile.exists() and pidfile.read_text().strip() != "")
    return int(pidfile.read_text().strip())


def fake_config(mode: str = "normal", name: str = "fake", **env: str) -> ServerConfig:
    return ServerConfig(name, sys.executable, (str(SERVER),), {"FAKE_MCP_MODE": mode, **env})


@pytest.fixture
async def open_server(tmp_path: Path) -> AsyncIterator[Open]:
    """Open fake servers in tmp_path; whatever is still open at the end is closed."""
    opened: list[SdkConnection] = []

    async def open_(mode: str = "normal", **env: str) -> SdkConnection:
        connection = await asyncio.wait_for(SdkConnector().connect(fake_config(mode, **env), tmp_path), 15)
        assert isinstance(connection, SdkConnection)
        opened.append(connection)
        return connection

    yield open_
    await asyncio.gather(*(c.aclose() for c in opened))


async def server_pid(connection: SdkConnection) -> int:
    result = await connection.call_tool("pid", {}, 10)
    return int(result.parts[0].split()[0])


async def test_texts_are_kept_and_an_image_becomes_a_placeholder(open_server: Open) -> None:
    connection = await open_server()
    result = await connection.call_tool("image", {}, 10)
    assert (result.parts, result.is_error) == (["first", "[image omitted]", "second"], False)


async def test_audio_and_resources_become_placeholders(open_server: Open) -> None:
    connection = await open_server()
    result = await connection.call_tool("mixed", {}, 10)
    assert result.parts == [
        "[audio omitted]",
        "[resource omitted: file:///doc.txt]",
        "[resource omitted: file:///emb.txt]",
    ]


async def test_a_command_that_does_not_exist_fails_with_a_reason(tmp_path: Path) -> None:
    config = ServerConfig("ghost", "definitely-not-a-command-xyz")
    with pytest.raises(McpError, match="definitely-not-a-command-xyz"):
        await asyncio.wait_for(SdkConnector().connect(config, tmp_path), 15)


async def test_a_server_that_writes_garbage_and_exits_fails_with_a_reason(tmp_path: Path) -> None:
    with pytest.raises(McpError, match="not valid JSON-RPC"):
        await asyncio.wait_for(SdkConnector().connect(fake_config("garbage"), tmp_path), 15)


async def test_a_server_that_exits_mid_call_fails_the_call_and_tells_the_listener(open_server: Open) -> None:
    connection = await open_server()
    reasons: list[str] = []
    connection.on_exit(reasons.append)
    with pytest.raises(McpExited):
        await connection.call_tool("exit", {}, 10)
    await wait_until(lambda: len(reasons) == 1)
    assert reasons[0]
    with pytest.raises(McpExited):
        await connection.call_tool("echo", {"text": "x"}, 10)
    assert len(reasons) == 1


async def test_a_server_that_exits_while_idle_tells_the_listener(open_server: Open) -> None:
    connection = await open_server()
    pid = await server_pid(connection)
    reasons: list[str] = []
    connection.on_exit(reasons.append)
    os.kill(pid, 9)
    await wait_until(lambda: len(reasons) == 1)
    with pytest.raises(McpExited):
        await connection.list_tools()


async def test_a_hanging_call_times_out_and_the_server_keeps_working(open_server: Open) -> None:
    connection = await open_server()
    started = time.monotonic()
    with pytest.raises(McpTimeout, match="timeout after 1s"):
        await connection.call_tool("hang", {}, 1)
    assert 0.9 <= time.monotonic() - started < 5
    assert (await connection.call_tool("echo", {"text": "after"}, 10)).parts == ["after"]


async def test_kill_is_a_synchronous_last_resort(open_server: Open) -> None:
    connection = await open_server()
    pid = await server_pid(connection)
    connection.kill()
    await wait_until(lambda: gone(pid), 5)
    connection.kill()  # a second kill, with the process gone, does nothing


async def test_kill_all_kills_a_server_that_is_still_starting(tmp_path: Path) -> None:
    connector = SdkConnector()
    connecting = asyncio.ensure_future(connector.connect(fake_config("slow_init"), tmp_path))
    try:
        await wait_until(lambda: bool(connector._pids))
        (pid,) = connector._pids
        assert not gone(pid)

        connector.kill_all()

        await wait_until(lambda: gone(pid), 5)
        with pytest.raises(McpError):
            await asyncio.wait_for(connecting, 10)
        assert connector._pids == set()
    finally:
        connecting.cancel()
        await asyncio.gather(connecting, return_exceptions=True)


async def test_a_server_that_never_answers_initialize_fails_in_the_hub_and_its_process_is_gone(
    tmp_path: Path,
) -> None:
    from src.engine.mcp.hub import McpHub

    connector = SdkConnector()
    hub = McpHub(connector, tmp_path, [fake_config("hang_init")], 20.0, start_timeout=1.0)
    events: list[tuple[str, str, str]] = []
    hub.listen(events.append)
    started = time.monotonic()
    hub.start()
    await wait_until(lambda: bool(events), 10)
    assert events == [("failed", "fake", "timed out starting after 1s")]
    assert time.monotonic() - started < 8
    await wait_until(lambda: connector._pids == set(), 8)
    await hub.close_all()


def test_the_sdk_function_that_reports_the_server_pid_is_still_there_and_hooked() -> None:
    """kill() and kill_all() depend on a private SDK function. If an SDK update renames it, this fails
    instead of the last-resort kill silently doing nothing."""
    import mcp.client.stdio as sdk_stdio

    spawn = getattr(sdk_stdio, "_create_platform_compatible_process", None)
    assert spawn is not None, "mcp.client.stdio._create_platform_compatible_process is gone: update src/client/mcp.py"
    assert spawn.__name__ == "spawn_and_report", "src/client/mcp.py no longer hooks the SDK's spawn function"


def test_the_non_json_hint_only_applies_shortly_after_the_server_wrote_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from mcp import StdioServerParameters

    now = [100.0]
    monkeypatch.setattr("src.client.mcp.time.monotonic", lambda: now[0])
    connection = SdkConnection(StdioServerParameters(command="x"), set())
    assert connection._hint() == ""
    connection._mark_garbage()
    assert "not valid JSON-RPC" in connection._hint()
    now[0] += 3600
    assert connection._hint() == ""


async def test_cancelling_a_hanging_call_leaves_the_server_ready_and_working(tmp_path: Path) -> None:
    from src.engine.mcp.hub import McpHub

    hub = McpHub(SdkConnector(), tmp_path, [fake_config()], 30.0)
    events: list[tuple[str, str, str]] = []
    hub.listen(events.append)
    hub.start()
    try:
        await wait_until(lambda: any(e[0] == "ready" for e in events), 15)
        call = asyncio.ensure_future(hub.call("fake", "hang", {}))
        await asyncio.sleep(0.5)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(call, 5)

        assert hub.states()[0].status == "ready"
        result = await asyncio.wait_for(hub.call("fake", "echo", {"text": "still here"}), 10)
        assert result.parts == ["still here"]
        assert [e[0] for e in events] == ["ready"]  # no exit or warning came of the cancel
    finally:
        await hub.close_all()
