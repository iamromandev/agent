"""The idle exit: with no client and no turn the daemon stops by itself; a running turn keeps it up (AC10)."""

from __future__ import annotations

import asyncio

from core.engine.provider.base import Done, TextDelta
from core.testing.fakes import Gate

REPLY = [TextDelta("ok"), Done(1, 1)]


async def test_with_no_client_the_daemon_stops_after_the_idle_time(serve) -> None:
    served = await serve(idle=0.3)
    await asyncio.wait_for(served.task, 5)  # it ends on its own
    assert served.daemon.stopping.is_set()


async def test_a_running_turn_keeps_it_up_and_it_stops_once_the_turn_ends(serve) -> None:
    gate = Gate()
    served = await serve([[gate, *REPLY]], idle=0.3)
    client = served.client()
    session_id = (await served.open(client))["session_id"]
    await client.post(f"/v1/sessions/{session_id}/messages", json={"text": "slow"})
    await asyncio.wait_for(gate.reached.wait(), 5)
    await asyncio.sleep(0.8)  # well past the idle time: the turn holds it
    assert not served.task.done()
    gate.release.set()
    await asyncio.wait_for(served.task, 5)
