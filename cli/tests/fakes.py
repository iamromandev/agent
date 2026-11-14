from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agent.approval import ApprovalRequest, Choice, Decision
from src.agent.mcp.config import ServerConfig
from src.agent.mcp.contract import CallResult, ToolInfo
from src.agent.provider.base import Message, ProviderError, ProviderEvent, ToolSpec
from src.agent.session import SessionWriteError, ToolMeta
from src.agent.web import FetchResponse


@dataclass
class Gate:
    """A point in a scripted reply where the stream stops until the test lets it go.

    The provider sets `reached` on arrival and waits for `release`, which nothing sets unless the
    test does. A test waits for `reached`, then cancels, with no sleeps involved.
    """

    reached: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


class ScriptedProvider:
    """A Provider that replays canned replies (events, or Gates to stop at) and records what it was asked."""

    def __init__(
        self,
        script: list[list[ProviderEvent | Gate]],
        delay: float = 0.0,
        model: str = "fake",
        context_window: int = 1000,
        check_result: str | None = None,
    ) -> None:
        self.script = list(script)
        self.delay = delay
        self.model = model
        self.context_window = context_window
        self.check_result = check_result
        self.check_gate: Gate | None = None  # while set, check() stops there until the test releases it
        self.requests: list[tuple[list[Message], list[str]]] = []

    async def stream(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> AsyncGenerator[ProviderEvent]:
        self.requests.append(([copy(m) for m in messages], [t.name for t in tools]))
        events: list[ProviderEvent | Gate] = (
            self.script.pop(0) if self.script else [ProviderError("script exhausted")]
        )
        for event in events:
            if isinstance(event, Gate):
                event.reached.set()
                await event.release.wait()
                continue
            if self.delay:
                await asyncio.sleep(self.delay)
            yield event

    async def check(self) -> str | None:
        if self.check_gate is not None:
            self.check_gate.reached.set()
            await self.check_gate.release.wait()
        return self.check_result


class RecordingApprover:
    """An Approver that records each request and answers from a script."""

    def __init__(
        self,
        decisions: list[Decision] | None = None,
        on_ask: Callable[[ApprovalRequest], None] | None = None,
    ) -> None:
        self.decisions = list(decisions or [])
        self.on_ask = on_ask  # runs while the dialog is "open", e.g. to press Tab
        self.requests: list[ApprovalRequest] = []

    async def ask(self, request: ApprovalRequest) -> Decision:
        self.requests.append(request)
        if self.on_ask is not None:
            self.on_ask(request)
        if not self.decisions:
            raise AssertionError(f"unexpected approval request: {request.label}")
        return self.decisions.pop(0)


class BlockingApprover:
    """An Approver that records the request, then waits until released (default: forever)."""

    def __init__(self, answer: Decision | None = None) -> None:
        self.answer = answer or Decision(Choice.ONCE)
        self.requests: list[ApprovalRequest] = []
        self.asked = asyncio.Event()  # set once a request is waiting
        self.release = asyncio.Event()

    async def ask(self, request: ApprovalRequest) -> Decision:
        self.requests.append(request)
        self.asked.set()
        await self.release.wait()
        return self.answer


@dataclass(frozen=True)
class RecordedCompaction:
    history: list[Message]
    tool_meta: dict[str, ToolMeta]
    pruned: int
    summarized: int
    before: int
    after: int


@dataclass(frozen=True)
class RecordedUndo:
    keep: int


class MemoryRecorder:
    """A Recorder that keeps what it was given, and can be told to fail.

    Like FileRecorder, a failed call keeps its record queued: the next successful call writes the
    queued records first, in order, so `messages` never has a gap. `close()` makes one last attempt
    to write what is queued and never raises.
    """

    def __init__(self, fail: int = 0, error: Exception | None = None) -> None:
        self.messages: list[tuple[Message, ToolMeta | None]] = []
        self.compactions: list[RecordedCompaction] = []
        self.undos: list[RecordedUndo] = []
        self.records: list[tuple[Message, ToolMeta | None] | RecordedCompaction | RecordedUndo] = []  # all kinds, in write order
        self.fail = fail  # the next `fail` calls raise
        self.error = error  # what they raise (default: SessionWriteError)
        self.calls = 0
        self.closed = 0
        self._queued: list[tuple[Message, ToolMeta | None] | RecordedCompaction | RecordedUndo] = []

    def _write(self) -> None:
        """One write attempt for everything queued: all of it, or (when told to fail) none."""
        self.calls += 1
        if self.fail > 0:
            self.fail -= 1
            raise self.error or SessionWriteError("disk full")
        for record in self._queued:
            if isinstance(record, RecordedCompaction):
                self.compactions.append(record)
            elif isinstance(record, RecordedUndo):
                self.undos.append(record)
            else:
                self.messages.append(record)
        self.records.extend(self._queued)
        self._queued.clear()

    def message(self, message: Message, meta: ToolMeta | None) -> None:
        self._queued.append((message, meta))
        self._write()

    def compaction(
        self,
        history: Sequence[Message],
        tool_meta: Mapping[str, ToolMeta],
        pruned: int,
        summarized: int,
        before: int,
        after: int,
    ) -> None:
        self._queued.append(
            RecordedCompaction([copy(m) for m in history], dict(tool_meta), pruned, summarized, before, after)
        )
        self._write()

    def undo(self, keep: int) -> None:
        self._queued.append(RecordedUndo(keep))
        self._write()

    def close(self) -> None:
        self.closed += 1
        if self._queued:
            with contextlib.suppress(Exception):
                self._write()


class FakeFetcher:
    """A Fetcher that records the URLs it is asked for and returns a canned response or raises."""

    def __init__(self, response: object = None) -> None:
        self.response = response  # a FetchResponse, or an Exception to raise
        self.calls: list[str] = []

    async def fetch(self, url: str) -> FetchResponse:
        self.calls.append(url)
        if isinstance(self.response, Exception):
            raise self.response
        assert isinstance(self.response, FetchResponse), "FakeFetcher needs a canned response"
        return self.response


class FakeConnection:
    """A McpConnection with scripted tools and results; it records what it is asked."""

    def __init__(self, tools: Sequence[ToolInfo] = (), results: Mapping[str, CallResult | Exception] | None = None) -> None:
        self.tools = list(tools)
        self.results = dict(results or {})  # by tool name: a CallResult, or an Exception to raise
        self.calls: list[tuple[str, dict[str, Any], float]] = []
        self.closed = 0  # aclose() count
        self.killed = 0
        self.cancelled = 0  # call_tool calls that were cancelled while waiting
        self.call_gate: asyncio.Event | None = None  # while set and not set(), call_tool waits
        self._exit_listeners: list[Callable[[str], None]] = []
        self._exited = False

    def on_exit(self, fn: Callable[[str], None]) -> None:
        self._exit_listeners.append(fn)

    def trigger_exit(self, reason: str) -> None:
        """Pretend the server died: the listeners hear it once, and calls waiting on the gate fail."""
        if self._exited:
            return
        self._exited = True
        for fn in self._exit_listeners:
            fn(reason)

    async def list_tools(self) -> list[ToolInfo]:
        return list(self.tools)

    async def call_tool(self, name: str, args: dict[str, Any], timeout: float) -> CallResult:
        self.calls.append((name, args, timeout))
        if self.call_gate is not None:
            try:
                await self.call_gate.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        result = self.results.get(name, CallResult([f"ok {name}"], False))
        if isinstance(result, Exception):
            raise result
        return result

    def kill(self) -> None:
        self.killed += 1

    async def aclose(self) -> None:
        self.closed += 1


class FakeConnector:
    """A McpConnector that hands out FakeConnections by server name, optionally after a delay or with a failure."""

    def __init__(
        self,
        connections: Mapping[str, FakeConnection] | None = None,
        failures: Mapping[str, Exception] | None = None,
        delays: Mapping[str, float] | None = None,
    ) -> None:
        self.connections = dict(connections or {})
        self.failures = dict(failures or {})
        self.delays = dict(delays or {})
        self.gates: dict[str, asyncio.Event] = {}  # while a server's gate exists and is unset, connect waits
        self.connects: list[tuple[str, Path]] = []  # (server, cwd) of every connect, even a failing one
        self.configs: list[ServerConfig] = []
        self.kill_alls = 0  # kill_all() calls

    def kill_all(self) -> None:
        self.kill_alls += 1

    async def connect(self, config: ServerConfig, cwd: Path) -> FakeConnection:
        self.connects.append((config.name, cwd))
        self.configs.append(config)
        if (gate := self.gates.get(config.name)) is not None:
            await gate.wait()
        if (delay := self.delays.get(config.name)):
            await asyncio.sleep(delay)
        if (failure := self.failures.get(config.name)) is not None:
            raise failure
        return self.connections.setdefault(config.name, FakeConnection())
