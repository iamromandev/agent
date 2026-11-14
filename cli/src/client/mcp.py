"""The MCP client: the official SDK over stdio, behind the McpConnector contract.

This is the only module that imports `mcp`. The SDK runs its transport and session in anyio task groups,
which must be entered and left by the same task, so each connection has one owner task that opens both and
stays inside them until the connection is closed or the server's output ends. Calls are made from the
callers' tasks: the session is safe to use from any task of the loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import os
import signal
import time
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import anyio
import mcp_types as types
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.shared.exceptions import MCPError

from src.agent.mcp.config import ServerConfig
from src.agent.mcp.contract import CallResult, McpConnection, McpError, McpExited, McpTimeout, ToolInfo

# The SDK logs a line it cannot parse with logger.exception; with no handler, logging would print that
# to stderr, over the full-screen UI. A parse failure is reported through the exit reason instead.
logging.getLogger("mcp").addHandler(logging.NullHandler())

CONNECTION_CLOSED = -32000  # the SDK's code for a request that was waiting when the connection ended
_NOT_JSON_RPC = "it wrote output that is not valid JSON-RPC"
_GARBAGE_WINDOW = 5.0  # seconds after such output in which an exit is blamed on it

# stdio_client does not expose the child it starts, and the kills need its pid. The module's spawn function
# is wrapped once to report the pid to the connection being opened (found through a context variable, so each
# owner task reports to its own connection). If a future SDK renames it, there is no pid and the kills do
# nothing; closing still works. A test fails if the function disappears.
_PID_SINK: ContextVar[Callable[[int], None] | None] = ContextVar("mcp_pid_sink", default=None)


def _report_spawned_pids() -> None:
    stdio = importlib.import_module("mcp.client.stdio")
    spawn = getattr(stdio, "_create_platform_compatible_process", None)
    if spawn is None:
        return

    async def spawn_and_report(*args: Any, **kwargs: Any) -> Any:
        process = await spawn(*args, **kwargs)
        if (sink := _PID_SINK.get()) is not None:
            sink(process.pid)
        return process

    vars(stdio)["_create_platform_compatible_process"] = spawn_and_report


_report_spawned_pids()


class _Watched:
    """The server's read stream, noticing when it ends and when the server wrote something that is not JSON-RPC."""

    def __init__(self, inner: Any, on_end: Callable[[], None], on_garbage: Callable[[], None]) -> None:
        self._inner = inner
        self._on_end = on_end
        self._on_garbage = on_garbage

    async def receive(self) -> Any:
        try:
            item = await self._inner.receive()
        except (anyio.EndOfStream, anyio.ClosedResourceError):
            self._on_end()
            raise
        if isinstance(item, Exception):  # the transport hands parse failures over as values
            self._on_garbage()
        return item

    async def aclose(self) -> None:
        await self._inner.aclose()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Any:
        try:
            return await self.receive()
        except anyio.EndOfStream:
            raise StopAsyncIteration from None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        await self.aclose()


def _leaf(exc: BaseException) -> BaseException:
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def _text(content: Any) -> str:
    match content:
        case types.TextContent():
            return content.text
        case types.ImageContent():
            return "[image omitted]"
        case types.AudioContent():
            return "[audio omitted]"
        case types.ResourceLink():
            return f"[resource omitted: {content.uri}]"
        case types.EmbeddedResource():
            return f"[resource omitted: {content.resource.uri}]"
        case _:
            return "[content omitted]"


class SdkConnection:
    """One running server. Built by SdkConnector.connect."""

    def __init__(self, params: StdioServerParameters, pids: set[int]) -> None:
        self._params = params
        self._pids = pids  # the connector's set of live server pids: this one is in it from spawn to the end
        self._session: ClientSession | None = None
        self._scope: anyio.CancelScope | None = None
        self._task: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[None] | None = None
        self._ended = asyncio.Event()  # the server's output ended: it exited, or its pipes broke
        self._closing = False  # aclose() was called: the end is ours, not an exit
        self._garbage_at: float | None = None  # when the server last wrote something that is not JSON-RPC
        self._dead = False
        self._pid: int | None = None
        self._exit_listeners: list[Callable[[str], None]] = []
        self._exit_reason: str | None = None  # set once the server was seen to exit on its own

    async def start(self) -> None:
        """Spawn and initialize; raises McpError with a one-line reason."""
        ready = self._ready = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(self._own(ready))
        try:
            await ready
        except BaseException:
            await self.aclose()  # also when this task is cancelled: the process must not outlive it
            raise

    def _set_pid(self, pid: int) -> None:
        self._pid = pid
        self._pids.add(pid)

    def _recent_garbage(self) -> bool:
        return self._garbage_at is not None and time.monotonic() - self._garbage_at < _GARBAGE_WINDOW

    def _hint(self) -> str:
        return f" ({_NOT_JSON_RPC})" if self._recent_garbage() else ""

    def _kill_group(self) -> None:
        """SIGKILL the server's process group: it leads its own (the SDK starts it in a new session), so this
        reaches its children too, also once the server itself has gone."""
        if self._pid is not None:
            with contextlib.suppress(OSError):
                os.killpg(self._pid, signal.SIGKILL)

    def _startup_failure(self, exc: BaseException) -> str:
        command = self._params.command
        if isinstance(exc, FileNotFoundError):
            return f"command not found: {command}"
        if isinstance(exc, OSError):
            return f"cannot start {command}: {exc}"
        if self._ended.is_set() or (isinstance(exc, MCPError) and exc.code == CONNECTION_CLOSED):
            if self._recent_garbage():
                return f"the server {_NOT_JSON_RPC} and exited"
            return "the server exited during startup"
        return str(exc) or type(exc).__name__

    async def _own(self, ready: asyncio.Future[None]) -> None:
        _PID_SINK.set(self._set_pid)
        failure: str | None = None
        try:
            with anyio.CancelScope() as scope, open(os.devnull, "w") as errlog:
                self._scope = scope
                if self._closing:
                    scope.cancel()
                async with stdio_client(self._params, errlog) as (read, write):
                    watched = _Watched(read, self._ended.set, self._mark_garbage)
                    async with ClientSession(watched, write) as session:
                        await session.initialize()
                        self._session = session
                        if not ready.done():
                            ready.set_result(None)
                        await self._ended.wait()
        except Exception as exc:
            failure = self._startup_failure(_leaf(exc))
        finally:
            self._session = None
            self._dead = True
            self._kill_group()  # the SDK waits only for the server itself: what it started must not outlive it
            if self._pid is not None:
                self._pids.discard(self._pid)
            self._pid = None
            if not ready.done():
                ready.set_exception(McpError(failure or "closed during startup"))
            elif not self._closing:
                self._exit_reason = failure or f"the server exited{self._hint()}"
                for fn in self._exit_listeners:
                    with contextlib.suppress(Exception):
                        fn(self._exit_reason)

    def _mark_garbage(self) -> None:
        self._garbage_at = time.monotonic()

    def _live(self) -> ClientSession:
        if self._session is None or self._dead or self._closing:
            raise McpExited(self._exit_reason or "the server is not running")
        return self._session

    @contextlib.contextmanager
    def _translated(self) -> Any:
        try:
            yield
        except MCPError as exc:
            if exc.code == CONNECTION_CLOSED:
                raise McpExited(self._exit_reason or f"the server exited{self._hint()}") from exc
            raise McpError(exc.message) from exc
        except (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream) as exc:
            raise McpExited(self._exit_reason or f"the server exited{self._hint()}") from exc

    async def list_tools(self) -> list[ToolInfo]:
        session = self._live()
        tools: list[ToolInfo] = []
        cursor: str | None = None
        with self._translated():
            while True:
                params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
                page = await session.list_tools(params=params)
                tools.extend(ToolInfo(t.name, t.description or "", dict(t.input_schema)) for t in page.tools)
                cursor = page.next_cursor
                if not cursor:
                    return tools

    async def call_tool(self, name: str, args: dict[str, Any], timeout: float) -> CallResult:
        session = self._live()
        with self._translated():
            try:
                result = await asyncio.wait_for(session.call_tool(name, args), timeout)
            except TimeoutError:
                raise McpTimeout(f"timeout after {timeout:g}s") from None
        if not isinstance(result, types.CallToolResult):
            raise McpError("the server asked for more input, which is not supported")
        return CallResult([_text(part) for part in result.content], result.is_error)

    def on_exit(self, fn: Callable[[str], None]) -> None:
        if self._exit_reason is not None:  # it already happened
            with contextlib.suppress(Exception):
                fn(self._exit_reason)
            return
        self._exit_listeners.append(fn)

    def kill(self) -> None:
        if not self._dead:
            self._kill_group()

    async def aclose(self) -> None:
        self._closing = True
        if self._scope is not None:
            self._scope.cancel()
        if self._task is not None:
            await asyncio.wait({self._task})  # the SDK stops the process (stdin closed, then SIGTERM, then SIGKILL)


class SdkConnector:
    def __init__(self) -> None:
        self._pids: set[int] = set()  # servers spawned and not yet ended, also those still connecting

    def kill_all(self) -> None:
        for pid in list(self._pids):
            with contextlib.suppress(OSError):
                os.killpg(pid, signal.SIGKILL)

    async def connect(self, config: ServerConfig, cwd: Path) -> McpConnection:
        params = StdioServerParameters(
            command=config.command,
            args=list(config.args),
            env=dict(os.environ) | dict(config.env),
            cwd=cwd,
        )
        connection = SdkConnection(params, self._pids)
        await connection.start()
        return connection
