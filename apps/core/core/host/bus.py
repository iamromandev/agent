"""The host's event bus: every event of a live session, numbered, buffered for replay and fanned out to subscribers."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

BUFFER = 2000  # events kept for `since` (a reconnecting client catches up from here)
QUEUE_SIZE = 1000  # events a subscriber may fall behind before it is dropped
REMOTE_DRAIN = 1.0  # seconds `drained` waits for the remote subscribers (a stuck socket must not stall a turn)


@dataclass(frozen=True)
class Overflow:
    """Why a subscriber was dropped: its queue was full when the event with `seq` was published."""

    seq: int


@dataclass(eq=False)
class Subscriber:
    """One client's view of the bus: `(seq, event)` pairs in order. Mark each handled item with `queue.task_done()`
    so `EventBus.drained` knows it was applied."""

    queue: asyncio.Queue[tuple[int, object]]
    overflow: Overflow | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)  # set when it is dropped or unsubscribed
    remote: bool = False  # a client in another process: `drained` waits for it only up to a deadline

    @property
    def dropped(self) -> bool:
        return self.overflow is not None


class EventBus:
    def __init__(self, *, buffer: int = BUFFER, queue_size: int = QUEUE_SIZE) -> None:
        self.seq = 0
        self._buffer: deque[tuple[int, object]] = deque(maxlen=buffer)
        self._queue_size = queue_size
        self.subscribers: list[Subscriber] = []

    def subscribe(self, *, queue_size: int | None = None, remote: bool = False) -> Subscriber:
        """A new subscriber; `queue_size` 0 is unbounded (an in-process client that must see every event). A `remote`
        one (a socket client) is waited for by `drained` only up to its deadline."""
        sub = Subscriber(asyncio.Queue(maxsize=self._queue_size if queue_size is None else queue_size), remote=remote)
        self.subscribers.append(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        if sub in self.subscribers:
            self.subscribers.remove(sub)
        sub.done.set()

    def publish(self, event: object) -> int:
        """Number `event`, keep it and hand it to every subscriber; returns its seq. Never blocks."""
        self.seq += 1
        item = (self.seq, event)
        self._buffer.append(item)
        for sub in list(self.subscribers):
            try:
                sub.queue.put_nowait(item)
            except asyncio.QueueFull:
                sub.overflow = Overflow(self.seq)
                self.unsubscribe(sub)
        return self.seq

    def since(self, seq: int) -> list[tuple[int, object]] | None:
        """The events after `seq`, or None when some of them are no longer in the buffer, or `seq` is one this bus never
        gave (a client of an earlier bus: it must start over from a snapshot)."""
        if seq > self.seq:
            return None
        if seq == self.seq:
            return []
        if not self._buffer or self._buffer[0][0] > seq + 1:
            return None
        return [item for item in self._buffer if item[0] > seq]

    async def drained(self, *, remote_timeout: float = REMOTE_DRAIN) -> None:
        """Wait until every current subscriber has handled what it was given (or went away). The local ones are waited
        for as long as it takes; the remote ones together for at most `remote_timeout` seconds."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + remote_timeout
        for sub in list(self.subscribers):
            join = asyncio.ensure_future(sub.queue.join())
            gone = asyncio.ensure_future(sub.done.wait())
            try:
                timeout = max(0.0, deadline - loop.time()) if sub.remote else None
                await asyncio.wait({join, gone}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            finally:
                join.cancel()
                gone.cancel()
