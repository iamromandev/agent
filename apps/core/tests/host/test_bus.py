"""The event bus: sequence numbers, the replay buffer and slow subscribers."""

from __future__ import annotations

import asyncio

from core.engine.events import Notice
from core.host.bus import EventBus, Overflow


def _n(text: str) -> Notice:
    return Notice("info", text)


async def test_events_reach_every_subscriber_in_order_with_rising_seq() -> None:
    bus = EventBus()
    a, b = bus.subscribe(), bus.subscribe()
    for text in ("one", "two", "three"):
        bus.publish(_n(text))
    for sub in (a, b):
        got = [sub.queue.get_nowait() for _ in range(3)]
        assert got == [(1, _n("one")), (2, _n("two")), (3, _n("three"))]


async def test_since_returns_what_followed_a_seq_or_none_when_it_left_the_buffer() -> None:
    bus = EventBus(buffer=3)
    for n in range(5):
        bus.publish(_n(str(n)))
    assert bus.since(3) == [(4, _n("3")), (5, _n("4"))]
    assert bus.since(5) == []
    assert bus.since(2) == [(3, _n("2")), (4, _n("3")), (5, _n("4"))]
    assert bus.since(1) is None  # seq 2 is gone: the gap cannot be filled


async def test_a_full_subscriber_is_dropped_and_told_while_the_others_go_on() -> None:
    bus = EventBus(queue_size=2)
    slow, fast = bus.subscribe(), bus.subscribe()
    bus.publish(_n("a"))
    bus.publish(_n("b"))
    fast.queue.get_nowait()
    fast.queue.get_nowait()
    bus.publish(_n("c"))  # slow is full: it is dropped

    assert slow.dropped and not fast.dropped
    assert slow not in bus.subscribers and fast in bus.subscribers
    assert fast.queue.get_nowait() == (3, _n("c"))
    assert slow.overflow == Overflow(3)
    bus.publish(_n("d"))
    assert fast.queue.get_nowait() == (4, _n("d"))


async def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    bus.unsubscribe(sub)
    bus.publish(_n("x"))
    assert sub.queue.empty()


async def test_drained_waits_until_every_subscriber_marked_its_events_done() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    bus.publish(_n("x"))
    waiter = asyncio.create_task(bus.drained())
    await asyncio.sleep(0)
    assert not waiter.done()
    sub.queue.get_nowait()
    sub.queue.task_done()
    await waiter
