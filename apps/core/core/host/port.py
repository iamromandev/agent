"""How a client drives a live session: `SessionPort`, and `LocalPort` for a client in the same process (the TUI)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol

from core.engine.approval import ApprovalRequest, Choice, Decision
from core.engine.events import Notice
from core.engine.mentions_index import FileIndex
from core.engine.mode import Mode
from core.engine.provider.base import Image
from core.engine.provider.models import ModelInfo
from core.engine.session import SessionInfo
from core.engine.text import one_line
from core.engine.turn_diff import DiffItem, FileDiff, TurnDiff
from core.host.asks import Answer, AnswerResult, AskEvent
from core.host.bus import Subscriber
from core.host.session import ModelSwitch, OpenedSession, SendResult, SessionHost, SessionSource, UndoResult

LOCAL_CLIENT = "local"  # the client id of the in-process TUI
SHOWN = 200  # characters of an error's text in its notice

OnEvent = Callable[[object], Awaitable[None]]
OnAsk = Callable[[AskEvent], None]


class SessionPort(Protocol):
    @property
    def session_id(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def mode(self) -> Mode: ...
    @property
    def local(self) -> bool: ...

    def subscribe(self, on_event: OnEvent, on_ask: OnAsk) -> None: ...
    def send(
        self,
        text: str,
        *,
        label: str | None = None,
        init: bool = False,
        draft: str | None = None,
        pasted: Sequence[Image] | None = None,
    ) -> SendResult: ...
    def interrupt(self) -> None: ...
    def set_mode(self, mode: Mode) -> None: ...
    async def undo(self) -> UndoResult: ...
    def compact(self) -> SendResult: ...
    async def new_session(self, command: str = "/new") -> str | None: ...
    async def open_session(self, info: SessionInfo) -> OpenedSession: ...
    async def switch_model(self, name: str, looked: tuple[int | None, str | None] | None = None) -> ModelSwitch: ...
    async def list_models(self) -> list[ModelInfo]: ...
    async def list_sessions(self) -> list[SessionInfo]: ...
    async def diff_list(self) -> TurnDiff | None: ...
    async def diff_file(self, diff: TurnDiff, item: DiffItem) -> FileDiff: ...
    def suggest_paths(self, token: str) -> list[Any]: ...
    def answer(self, ask_id: str, answer: Answer) -> AnswerResult: ...
    def outcome_applied(self, ack: int) -> bool: ...


class LocalPort:
    """The port of a client in the host's own process: calls go straight to the host, and the bus is pumped into the
    client's handlers one event at a time (each one is drawn before the next, and before the host goes on where it
    waits for its subscribers)."""

    local = True

    def __init__(self, host: SessionHost, file_index: FileIndex) -> None:
        self.host = host
        self.file_index = file_index
        self._subscriber: Subscriber | None = None
        self._pump: asyncio.Task[None] | None = None

    @property
    def session_id(self) -> str:
        return self.host.session_id

    @property
    def model(self) -> str:
        return self.host.model

    @property
    def mode(self) -> Mode:
        return self.host.agent.mode

    def subscribe(self, on_event: OnEvent, on_ask: OnAsk) -> None:
        """Start handing the bus's events to `on_event` and the asks to `on_ask` (needs a running loop)."""
        sub = self.host.bus.subscribe(queue_size=0)  # never dropped: the TUI draws everything, as it always did
        self._subscriber = sub
        self._pump = asyncio.get_running_loop().create_task(self._run(sub, on_event, on_ask))

    async def _run(self, sub: Subscriber, on_event: OnEvent, on_ask: OnAsk) -> None:
        while True:
            _, event = await sub.queue.get()
            try:
                if isinstance(event, AskEvent):
                    try:
                        on_ask(event)
                    except Exception as exc:  # no dialog: the ask would wait forever, so it is denied
                        approval = isinstance(event.request, ApprovalRequest)
                        no = Decision(Choice.DENY, "could not show the dialog") if approval else False  # trust: no
                        self.answer(event.ask_id, no)
                        self.host.bus.publish(
                            Notice("error", f"Could not show the approval dialog: {one_line(str(exc), SHOWN)}")
                        )
                else:
                    await on_event(event)
            except asyncio.CancelledError:
                if (task := asyncio.current_task()) is not None and task.cancelling():
                    raise  # the pump itself is being stopped
            except Exception as exc:  # the pump must go on; said once, unless showing a notice is what failed
                if not isinstance(event, Notice):
                    self.host.bus.publish(Notice("error", f"Display error: {one_line(str(exc), SHOWN)}"))
            finally:
                sub.queue.task_done()

    async def close(self) -> None:
        """Stop the pump; what is still queued is dropped (the client is going away)."""
        if self._subscriber is not None:
            self.host.bus.unsubscribe(self._subscriber)
        if self._pump is not None:
            self._pump.cancel()
            with contextlib.suppress(BaseException):
                await self._pump

    async def idle(self) -> None:
        await self.host.idle()

    def send(
        self,
        text: str,
        *,
        label: str | None = None,
        init: bool = False,
        draft: str | None = None,
        pasted: Sequence[Image] | None = None,
    ) -> SendResult:
        return self.host.send(text, label=label, init=init, draft=draft, pasted=pasted)

    def interrupt(self) -> None:
        self.host.interrupt()

    def set_mode(self, mode: Mode) -> None:
        self.host.set_mode(mode)

    async def undo(self) -> UndoResult:
        return await self.host.undo()

    def compact(self) -> SendResult:
        return self.host.compact()

    async def new_session(self, command: str = "/new") -> str | None:
        return await self.host.new_session(command)

    async def open_session(self, info: SessionInfo) -> OpenedSession:
        return await self.host.open_session(info)

    async def session_started(self, source: SessionSource) -> list[Any]:
        return await self.host.session_started(source)

    async def switch_model(self, name: str, looked: tuple[int | None, str | None] | None = None) -> ModelSwitch:
        return await self.host.switch_model(name, looked)

    async def list_models(self) -> list[ModelInfo]:
        return await self.host.list_models()

    async def model_length(self, name: str) -> int | None:
        return await self.host.model_length(name)

    async def list_sessions(self) -> list[SessionInfo]:
        return await self.host.list_sessions()

    async def diff_list(self) -> TurnDiff | None:
        return await self.host.diff_list()

    async def diff_file(self, diff: TurnDiff, item: DiffItem) -> FileDiff:
        return await self.host.diff_file(diff, item)

    def suggest_paths(self, token: str) -> list[Any]:
        return self.file_index.suggest(token)

    def answer(self, ask_id: str, answer: Answer) -> AnswerResult:
        return self.host.asks.answer(ask_id, answer, LOCAL_CLIENT)

    def outcome_applied(self, ack: int) -> bool:
        """The client applied a `TurnOutcome` with this ack; True when a message queued meanwhile started a turn."""
        return self.host.outcome_applied(ack)
