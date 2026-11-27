"""The ask broker: a question to whichever client answers first."""

from __future__ import annotations

import asyncio

import pytest
from core.engine.approval import ApprovalRequest, Choice, Decision
from core.host.asks import AskBroker, AskEvent, AskResolved, HostApprover
from core.host.bus import EventBus

REQUEST = ApprovalRequest("bash", "bash ls", None, allow_always=True)


def _events(bus: EventBus, sub) -> list[object]:
    out = []
    while not sub.queue.empty():
        out.append(sub.queue.get_nowait()[1])
    return out


async def _asked(broker: AskBroker, request: ApprovalRequest = REQUEST) -> tuple[asyncio.Task[object], str]:
    task = asyncio.create_task(broker.ask("approval", request))
    await asyncio.sleep(0)
    (pending,) = broker.pending()
    return task, pending.ask_id


async def test_the_first_answer_wins_and_is_published() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    broker = AskBroker(bus)
    task, ask_id = await _asked(broker)

    assert broker.answer(ask_id, Decision(Choice.ONCE), "a") == "ok"
    assert broker.answer(ask_id, Decision(Choice.DENY), "b") == "already_answered"
    assert await task == Decision(Choice.ONCE)
    events = _events(bus, sub)
    assert events[0] == AskEvent(ask_id, "approval", REQUEST)
    assert events[1] == AskResolved(ask_id, "a", Decision(Choice.ONCE))
    assert broker.pending() == []


async def test_two_answers_at_once_leave_exactly_one_ok() -> None:
    broker = AskBroker(EventBus())
    task, ask_id = await _asked(broker)

    async def answer(client: str) -> str:
        await asyncio.sleep(0)
        return broker.answer(ask_id, Decision(Choice.ONCE), client)

    results = await asyncio.gather(answer("a"), answer("b"))
    assert sorted(results) == ["already_answered", "ok"]
    await task


async def test_an_unknown_id_and_an_answer_the_ask_does_not_offer() -> None:
    broker = AskBroker(EventBus())
    task, ask_id = await _asked(broker, ApprovalRequest("bash", "bash ls", None, allow_always=False))
    assert broker.answer("nope", Decision(Choice.ONCE), "a") == "unknown"
    assert broker.answer(ask_id, Decision(Choice.ALWAYS), "a") == "invalid"  # Always is not offered
    assert broker.answer(ask_id, True, "a") == "invalid"  # a trust answer to an approval
    assert broker.answer(ask_id, Decision(Choice.DENY), "a") == "ok"
    assert await task == Decision(Choice.DENY)


async def test_cancel_all_cancels_the_askers_and_publishes_the_end() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    broker = AskBroker(bus)
    task, ask_id = await _asked(broker)
    broker.cancel_all("closing")
    with pytest.raises(asyncio.CancelledError):
        await task
    assert broker.pending() == []
    assert _events(bus, sub)[-1] == AskResolved(ask_id, "", None)
    assert broker.answer(ask_id, Decision(Choice.ONCE), "a") == "unknown"


async def test_a_cancelled_asker_withdraws_its_ask() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    broker = AskBroker(bus)
    task, ask_id = await _asked(broker)
    task.cancel()
    assert broker.answer(ask_id, Decision(Choice.ONCE), "a") == "unknown"  # the waiter is gone already
    with pytest.raises(asyncio.CancelledError):
        await task
    assert broker.pending() == [] and _events(bus, sub)[-1] == AskResolved(ask_id, "", None)


async def test_a_late_subscriber_sees_the_pending_ask() -> None:
    bus = EventBus()
    broker = AskBroker(bus)
    task, ask_id = await _asked(broker)
    assert [p.ask_id for p in broker.pending()] == [ask_id]  # what a client that connects now is shown
    broker.cancel_all("done")
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_the_host_approver_asks_a_protected_read_as_its_own_kind() -> None:
    broker = AskBroker(EventBus())
    request = ApprovalRequest("read", "read .env", None, allow_always=False, protected=True, protected_path="/w/.env")
    task = asyncio.create_task(HostApprover(broker).ask(request))
    await asyncio.sleep(0)
    (pending,) = broker.pending()
    assert pending.kind == "protected"
    assert broker.answer(pending.ask_id, Decision(Choice.ALWAYS), "a") == "invalid"
    broker.answer(pending.ask_id, Decision(Choice.ONCE), "a")
    assert await task == Decision(Choice.ONCE)
