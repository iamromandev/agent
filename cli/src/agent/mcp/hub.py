"""The MCP hub: one place that owns every configured server's state, starts them in the background and
offers their tools to the agent."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.agent.mcp.config import ServerConfig
from src.agent.mcp.contract import CallResult, McpConnection, McpConnector, McpError, McpExited, ToolInfo
from src.agent.mcp.names import tool_name
from src.agent.mcp.tool import McpTool
from src.agent.tools.base import Tool

Status = Literal["starting", "ready", "failed", "exited", "untrusted"]
HubEvent = tuple[Literal["ready", "failed", "exited", "warn"], str, str]  # (kind, server, text)


@dataclass
class ServerState:
    config: ServerConfig
    status: Status
    reason: str = ""
    tools: list[ToolInfo] = field(default_factory=list)


def _count(tools: int) -> str:
    return f"{tools} tool" if tools == 1 else f"{tools} tools"


class McpHub:
    """Server states, background start, the tools of the ready servers, and calls to them.

    Everything runs on one event loop. `start` needs that loop to be running; a server that is slow or
    broken only changes its own state, and tells the listeners.
    """

    def __init__(
        self,
        connector: McpConnector,
        workspace: Path,
        servers: list[ServerConfig],
        timeout: float,
        warnings: Sequence[str] = (),
        *,
        start_timeout: float = 120.0,
    ) -> None:
        self._connector = connector
        self._workspace = workspace
        self._timeout = timeout  # seconds a tool call may take
        self._start_timeout = start_timeout  # seconds a server may take to connect and list its tools
        self._states = {s.name: ServerState(s, "starting") for s in servers}
        self._connections: dict[str, McpConnection] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._inflight: dict[str, set[tuple[asyncio.Future[CallResult], asyncio.Future[CallResult]]]] = {}
        self._tools: dict[tuple[str, str], McpTool] = {}  # by (server, tool): the same object every time
        self._warned: set[tuple[str, str]] = set()
        self._listeners: list[Callable[[HubEvent], None]] = []
        self._backlog: list[HubEvent] = [("warn", "", w) for w in warnings]  # replayed to the first listener
        self._closed = False

    def listen(self, fn: Callable[[HubEvent], None]) -> None:
        """Hear every event from now on. The first listener also gets what happened before it came."""
        self._listeners.append(fn)
        backlog, self._backlog = self._backlog, []
        for event in backlog:
            self._deliver(fn, event)

    def _emit(self, kind: Literal["ready", "failed", "exited", "warn"], server: str, text: str) -> None:
        event: HubEvent = (kind, server, text)
        if not self._listeners:
            self._backlog.append(event)
            return
        for fn in list(self._listeners):
            self._deliver(fn, event)

    @staticmethod
    def _deliver(fn: Callable[[HubEvent], None], event: HubEvent) -> None:
        with contextlib.suppress(Exception):  # a broken UI must not break the hub
            fn(event)

    def start(self, names: Iterable[str] | None = None) -> None:
        """Start servers in the background (every one that is not untrusted when `names` is None)."""
        if self._closed:
            return
        wanted = [n for n, s in self._states.items() if s.status != "untrusted"] if names is None else list(names)
        for name in wanted:
            if name in self._states and name not in self._tasks:
                self._tasks[name] = asyncio.create_task(self._run(name))

    async def _run(self, name: str) -> None:
        state = self._states[name]
        connection: McpConnection | None = None
        try:
            # One deadline for the whole start: a server that never answers `initialize` or the tool list
            # must not stay "starting" forever. The connector stops a connect that is cancelled by it.
            async with asyncio.timeout(self._start_timeout) as deadline:
                connection = await self._connector.connect(state.config, self._workspace)
                if self._closed:  # closed while connecting, by a connector that did not stop in time
                    return await connection.aclose()
                self._connections[name] = connection
                connection.on_exit(lambda reason: self._exited(name, reason))
                tools = await connection.list_tools()
        except TimeoutError as exc:
            if deadline.expired():
                reason = f"timed out starting after {self._start_timeout:g}s"
            else:  # a timeout of the connection's own, not our deadline: still only this server that failed
                reason = str(exc) or type(exc).__name__
            await self._abort_start(state, connection, reason)
            return
        except Exception as exc:
            await self._abort_start(state, connection, str(exc) or type(exc).__name__)
            return
        if state.status != "starting":  # it exited while the tools were being listed: that was already reported
            return
        state.tools = tools
        state.status = "ready"
        self._emit("ready", name, _count(len(tools)))

    async def _abort_start(self, state: ServerState, connection: McpConnection | None, reason: str) -> None:
        """The start failed: say so (unless an exit already did), and stop whatever was started."""
        name = state.config.name
        if state.status == "starting":
            state.status = "failed"
            state.reason = reason
            self._emit("failed", name, reason)
        if connection is not None:
            self._connections.pop(name, None)
            await connection.aclose()

    def _exited(self, name: str, reason: str) -> None:
        state = self._states[name]
        if self._closed or state.status not in ("starting", "ready"):
            return
        state.status = "exited"
        state.reason = reason
        self._fail_inflight(name, McpExited(reason))
        self._emit("exited", name, reason)

    def _fail_inflight(self, name: str, error: McpError) -> None:
        for outcome, runner in list(self._inflight.pop(name, ())):
            if not outcome.done():
                outcome.set_exception(error)
            runner.cancel()

    def tools(self) -> list[Tool]:
        """A tool for each tool of each ready server, in config order. A name that is taken is skipped (warned once)."""
        offered: list[Tool] = []
        taken: set[str] = set()
        for name, state in self._states.items():
            if state.status != "ready":
                continue
            for info in state.tools:
                full = tool_name(name, info.name)
                if full in taken:
                    if (name, info.name) not in self._warned:
                        self._warned.add((name, info.name))
                        self._emit("warn", name, f'tool "{info.name}" skipped: its name {full} is already taken')
                    continue
                taken.add(full)
                tool = self._tools.get((name, info.name))
                if tool is None or tool.name != full:
                    tool = self._tools[(name, info.name)] = McpTool(self, name, info, full)
                offered.append(tool)
        return offered

    def states(self) -> list[ServerState]:
        return list(self._states.values())

    async def call(self, server: str, tool: str, args: dict[str, Any]) -> CallResult:
        """Call a tool of a ready server. Raises McpError (McpTimeout, McpExited) when it cannot be done."""
        state = self._states.get(server)
        connection = self._connections.get(server)
        if state is None or connection is None or state.status != "ready":
            raise McpError(f'mcp server "{server}" is not running')
        loop = asyncio.get_running_loop()
        outcome: asyncio.Future[CallResult] = loop.create_future()
        runner = asyncio.ensure_future(connection.call_tool(tool, args, self._timeout))

        def relay(done: asyncio.Future[CallResult]) -> None:
            if done.cancelled():
                outcome.cancel()
            elif (error := done.exception()) is not None:
                if not outcome.done():
                    outcome.set_exception(error)
            elif not outcome.done():
                outcome.set_result(done.result())

        runner.add_done_callback(relay)
        entry = (outcome, runner)
        self._inflight.setdefault(server, set()).add(entry)
        try:
            return await outcome
        finally:
            self._inflight.get(server, set()).discard(entry)
            if not runner.done():
                runner.cancel()

    async def close_all(self) -> None:
        """Stop every server and wait until they are gone. Safe to call twice."""
        self._closed = True
        for name in list(self._inflight):
            self._fail_inflight(name, McpExited("the agent is closing"))
        starting = [task for task in self._tasks.values() if not task.done()]
        for task in starting:
            task.cancel()
        await asyncio.gather(*starting, return_exceptions=True)
        connections, self._connections = list(self._connections.values()), {}
        await asyncio.gather(*(c.aclose() for c in connections), return_exceptions=True)
        for state in self._states.values():
            if state.status == "ready":
                state.status = "exited"
                state.reason = "closed"

    def kill_all(self) -> None:
        """Synchronous last resort for when the loop that owns the connections is gone."""
        with contextlib.suppress(Exception):
            self._connector.kill_all()  # also the servers that are still starting: no connection yet
        for connection in list(self._connections.values()):
            with contextlib.suppress(Exception):
                connection.kill()
