from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field

from src.agent.approval import ApprovalRequest, Choice, Decision
from src.agent.provider.base import Message, ProviderError, ProviderEvent, ToolSpec
from src.agent.session import SessionWriteError, ToolMeta


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
