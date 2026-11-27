"""One job that runs tools at a time per daemon: the sessions share one checkpoint store, so two turns at once would
break its index and `/undo` in one could revert the other's files."""

from __future__ import annotations

import asyncio

from core.engine.provider.base import Done, TextDelta
from core.host.session import OTHER_SESSION
from core.testing.fakes import Gate

REPLY = [TextDelta("ok"), Done(1, 1)]


async def test_a_second_sessions_turn_compact_and_undo_are_refused_while_another_session_runs_one(serve) -> None:
    gate = Gate()
    served = await serve([[gate, *REPLY], REPLY])
    a, b = served.client("A"), served.client("B")
    one = (await served.open(a))["session_id"]
    two = (await served.open(b))["session_id"]
    stream_one = served.stream(a, one, after=0)
    stream_two = served.stream(b, two, after=0)
    await a.post(f"/v1/sessions/{one}/messages", json={"text": "slow"})
    await asyncio.wait_for(gate.reached.wait(), 5)

    refused = (await b.post(f"/v1/sessions/{two}/messages", json={"text": "mine"})).json()
    compact = (await b.post(f"/v1/sessions/{two}/compact")).json()
    undo = (await b.post(f"/v1/sessions/{two}/undo")).json()
    assert refused == {"status": "refused", "reason": OTHER_SESSION}
    assert compact == {"status": "refused", "reason": OTHER_SESSION}
    assert undo["prompt"] is None and undo["events"][0]["text"] == OTHER_SESSION
    assert len(served.provider.requests) == 1  # the second session's turn never reached the model

    gate.release.set()
    await stream_one.until(lambda s: "turn_outcome" in s.types())
    host_one = served.daemon.host(one)
    from core.testing.helpers import wait_until

    await wait_until(lambda: not host_one.turn_active)
    started = (await b.post(f"/v1/sessions/{two}/messages", json={"text": "now"})).json()
    assert started["status"] == "started"
    await stream_two.until(lambda s: "turn_outcome" in s.types())
    assert len(served.provider.requests) == 2


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
