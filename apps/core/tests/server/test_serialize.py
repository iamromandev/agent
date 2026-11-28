"""Turns of different sessions run at once; only `/model` waits for another session's turn (the provider is shared)."""

from __future__ import annotations

import asyncio

from core.engine.provider.base import Done, TextDelta
from core.host.session import OTHER_SESSION
from core.testing.fakes import Gate

REPLY = [TextDelta("ok"), Done(1, 1)]


async def test_compact_and_undo_in_a_second_session_are_not_refused_while_another_session_runs_a_turn(serve) -> None:
    # Bug caught: the old one-job-at-a-time refusal (OTHER_SESSION) still returned for compact and undo.
    gate = Gate()
    served = await serve([[gate, *REPLY], REPLY])
    a, b = served.client("A"), served.client("B")
    one = (await served.open(a))["session_id"]
    two = (await served.open(b))["session_id"]
    await a.post(f"/v1/sessions/{one}/messages", json={"text": "slow"})
    await asyncio.wait_for(gate.reached.wait(), 5)

    compact = (await b.post(f"/v1/sessions/{two}/compact")).json()
    undo = (await b.post(f"/v1/sessions/{two}/undo")).json()

    assert OTHER_SESSION not in str(compact) and OTHER_SESSION not in str(undo)
    assert undo["events"][0]["text"] == "Nothing to undo"
    gate.release.set()


async def test_a_model_switch_waits_for_another_sessions_turn_and_is_recorded_in_every_session(serve) -> None:
    import json

    from core.testing.helpers import wait_until

    gate = Gate()
    served = await serve([REPLY, [gate, *REPLY]])
    a, b = served.client("A"), served.client("B")
    one = (await served.open(a))["session_id"]
    two = (await served.open(b))["session_id"]
    stream_one = served.stream(a, one, after=0)
    await a.post(f"/v1/sessions/{one}/messages", json={"text": "first"})  # gives A a session file
    await stream_one.until(lambda s: "turn_outcome" in s.types())
    host_one = served.daemon.host(one)
    await wait_until(lambda: not host_one.turn_active)
    await a.post(f"/v1/sessions/{one}/messages", json={"text": "slow"})
    await asyncio.wait_for(gate.reached.wait(), 5)

    refused = (await b.post(f"/v1/sessions/{two}/model", json={"name": "other:7b"})).json()
    assert refused["model"] is None and refused["notices"] == [{"level": "warn", "text": OTHER_SESSION}]
    assert served.provider.model == "fake"  # A's turn keeps its model

    gate.release.set()
    await wait_until(lambda: not host_one.turn_active)
    assert (await b.post(f"/v1/sessions/{two}/model", json={"name": "other:7b"})).json()["model"] == "other:7b"
    path = host_one.agent.recorder.path
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["model"] for r in records if r.get("type") == "model"] == ["other:7b"]
