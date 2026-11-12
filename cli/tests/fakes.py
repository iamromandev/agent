from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Sequence
from copy import copy
from dataclasses import dataclass, field

from src.agent.approval import ApprovalRequest, Choice, Decision
from src.agent.provider.base import Message, ProviderError, ProviderEvent, ToolSpec


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
