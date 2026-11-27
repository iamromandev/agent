"""The daemon's HTTP API over a real Unix socket, driven with httpx against a daemon over a scripted provider."""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path

from core.engine.events import Notice, WorkflowLine
from core.engine.provider.base import Done, Message, TextDelta, ToolCall, ToolCallEvent
from core.engine.session import FileRecorder
from core.engine.session.lock import SessionLock
from core.server.serve import hello, stop
from core.testing.fakes import Gate
from core.testing.helpers import session_held_elsewhere, wait_until

REPLY = [TextDelta("ok"), Done(1, 1)]
SECRET = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"  # the shape of a GitHub token: redacted everywhere


async def test_hello_needs_no_client_and_every_response_names_the_protocol(serve) -> None:
    served = await serve()
    anon = served.client("A")
    del anon.headers["X-Agent-Client"]
    hello = await anon.get("/v1/hello")
    assert hello.status_code == 200 and hello.headers["x-agent-protocol"] == "1"
    assert hello.json()["v"] == 1 and hello.json()["workspace"] == str(served.workspace)
    missing = await anon.get("/v1/sessions")
    assert missing.status_code == 400 and missing.headers["x-agent-protocol"] == "1"


async def test_bad_json_an_unknown_session_and_an_unknown_ask(serve) -> None:
    served = await serve()
    client = served.client()
    bad = await client.post("/v1/sessions", content=b"{nope")
    assert bad.status_code == 400 and "JSON" in bad.json()["error"]
    assert (await client.post("/v1/sessions/NOPE/messages", json={"text": "x"})).status_code == 404
    assert (await client.post("/v1/asks/NOPE", json={"trust": True})).status_code == 404


async def test_only_the_unix_socket_is_listened_on(serve) -> None:
    served = await serve()
    families = {s.family for server in served.daemon.server.servers for s in server.sockets}
    assert families == {socket.AF_UNIX}


async def test_a_message_streams_its_entry_reply_and_outcome_and_the_claim_is_released(serve) -> None:
    served = await serve([REPLY])
    client = served.client()
    opened = await served.open(client)
    session_id = opened["session_id"]
    assert opened["snapshot"]["type"] == "snapshot" and opened["snapshot"]["claims"]["turn_active"] is False
    stream = served.stream(client, session_id, after=opened["snapshot"]["seq"])
    await stream.until(lambda s: True)
    sent = await client.post(f"/v1/sessions/{session_id}/messages", json={"text": "hi"})
    assert sent.json() == {"status": "started", "reason": None}
    await stream.until(lambda s: "turn_outcome" in s.types())

    kinds = stream.types()
    assert kinds.index("user_entry") < kinds.index("assistant_text") < kinds.index("turn_ended") < kinds.index("turn_outcome")
    assert stream.of("user_entry")[0]["client"] == "A"
    host = served.daemon.host(session_id)
    await wait_until(lambda: not host.turn_active)  # A's stream applied its own outcome: no 2 s wait


async def test_an_approval_is_answered_once_and_a_late_or_wrong_answer_is_refused(serve) -> None:
    bash = [ToolCallEvent(ToolCall("b1", "bash", {"command": "true"})), Done(1, 1)]
    served = await serve([bash, REPLY])
    a, b = served.client("A"), served.client("B")
    session_id = (await served.open(a))["session_id"]
    stream_a = served.stream(a, session_id, after=0)
    stream_b = served.stream(b, session_id, after=0)
    await a.post(f"/v1/sessions/{session_id}/messages", json={"text": "run it"})
    await stream_b.until(lambda s: bool(s.of("ask")))
    ask = stream_b.of("ask")[0]
    assert ask["kind"] == "approval" and ask["request"]["tool"] == "bash"

    wrong = await b.post(f"/v1/asks/{ask['ask_id']}", json={"trust": True})  # not what an approval takes
    assert wrong.status_code == 400
    first = await b.post(f"/v1/asks/{ask['ask_id']}", json={"choice": "once"})
    late = await a.post(f"/v1/asks/{ask['ask_id']}", json={"choice": "deny"})
    assert first.json() == {"result": "ok"}
    assert late.status_code == 409 and late.json()["result"] == "already_answered"
    await stream_a.until(lambda s: "turn_outcome" in s.types())
    resolved = stream_a.of("ask_resolved")
    assert [(r["by"], r["decision"]["choice"]) for r in resolved] == [("B", "once")]
    assert len(stream_a.of("tool_finished")) == 1  # the tool ran once


async def test_secrets_never_cross_the_wire(serve, monkeypatch) -> None:
    served = await serve([[TextDelta(f"the token is {SECRET}"), Done(1, 1)]])
    client = served.client()
    session_id = (await served.open(client))["session_id"]
    stream = served.stream(client, session_id, after=0)
    await client.post(f"/v1/sessions/{session_id}/messages", json={"text": f"remember {SECRET}"})
    await stream.until(lambda s: "turn_outcome" in s.types())
    again = await served.open(served.client("B"), resume=session_id)  # the snapshot's records too

    assert SECRET not in stream.raw.decode() and SECRET not in str(again["snapshot"]["records"])
    assert "[redacted:github-token]" in stream.raw.decode()


async def test_an_event_with_no_wire_form_is_skipped_and_the_stream_goes_on(serve) -> None:
    served = await serve()
    client = served.client()
    session_id = (await served.open(client))["session_id"]
    stream = served.stream(client, session_id, after=0)
    host = served.daemon.host(session_id)
    host.bus.publish(WorkflowLine("info", "local only"))
    host.bus.publish(Notice("info", "after it"))
    await stream.until(lambda s: any(m.get("text") == "after it" for m in s.of("notice")))
    assert not stream.ended.is_set()


async def test_a_reconnect_gets_exactly_what_it_missed_or_a_snapshot_when_the_gap_is_too_old(serve) -> None:
    served = await serve()
    client = served.client()
    session_id = (await served.open(client))["session_id"]
    host = served.daemon.host(session_id)
    for n in range(3):
        host.bus.publish(Notice("info", f"n{n}"))
    seq = host.bus.seq
    host.bus.publish(Notice("info", "missed"))
    stream = served.stream(client, session_id, after=seq)
    await stream.until(lambda s: bool(s.messages))
    assert [m["text"] for m in stream.of("notice")] == ["missed"] and stream.messages[0]["seq"] == seq + 1

    for n in range(2001):  # more than the buffer keeps
        host.bus.publish(Notice("info", str(n)))
    old = served.stream(served.client("B"), session_id, after=seq)
    await old.until(lambda s: bool(s.messages))
    assert old.messages[0]["type"] == "snapshot"


async def test_a_session_open_in_another_process_is_409(serve) -> None:
    served = await serve()
    folder = served.daemon._sessions
    recorder = FileRecorder(folder, served.workspace)
    recorder.message(Message("user", "elsewhere"), None)
    session_id = recorder.session_id
    recorder.close()
    with session_held_elsewhere(folder, session_id):
        refused = await served.client().post("/v1/sessions", json={"resume": session_id})
    assert refused.status_code == 409 and "locked by another process" in refused.json()["error"]


async def test_two_clients_resuming_one_session_at_once_share_one_host(serve) -> None:
    served = await serve()
    folder = served.daemon._sessions
    recorder = FileRecorder(folder, served.workspace)
    recorder.message(Message("user", "shared"), None)
    session_id = recorder.session_id
    recorder.close()

    (one, _), (two, _) = await asyncio.gather(served.daemon.open(session_id), served.daemon.open(session_id[:10]))

    assert one is two and list(served.daemon.hosts) == [session_id]
    await one.aclose()
    one.agent.close()
    assert not SessionLock.held_by_us(folder, session_id)  # no orphan holds a count of the lock


async def test_a_clients_switch_claim_goes_when_its_last_stream_closes(serve) -> None:
    served = await serve()
    client = served.client("A")
    session_id = (await served.open(client))["session_id"]
    host = served.daemon.host(session_id)
    stream = served.stream(client, session_id, after=0)
    await wait_until(lambda: served.daemon.connections.get((session_id, "A")) == 1)
    assert host.begin_switch("A")
    await stream.close()
    await wait_until(lambda: not host.switching)


async def test_a_config_change_warning_reaches_every_session(serve) -> None:
    served = await serve()
    a, b = served.client("A"), served.client("B")
    one = (await served.open(a))["session_id"]
    two = (await served.open(b))["session_id"]
    streams = [served.stream(a, one, after=0), served.stream(b, two, after=0)]
    await asyncio.to_thread(served.daemon._config_changed, "config changed by bash (x): .agent/settings.json")
    for stream in streams:
        await stream.until(lambda s: any("config changed by bash" in m["text"] for m in s.of("notice")))


async def test_stop_ends_the_daemon_and_says_when_there_was_none(serve, tmp_path: Path) -> None:
    served = await serve()
    assert await asyncio.to_thread(stop, served.workspace)
    await asyncio.wait_for(served.task, 10)
    assert not await asyncio.to_thread(stop, tmp_path / "nobody")
    assert await asyncio.to_thread(hello, served.workspace) is None


def _project_server(ws: Path) -> None:
    (ws / ".agent").mkdir()
    (ws / ".agent" / "mcp.json").write_text(json.dumps({"servers": {"tools": {"command": "true"}}}))


async def test_a_trust_prompt_waits_for_a_client_and_the_first_answer_applies_once(serve) -> None:
    served = await serve(setup=_project_server)
    a, b = served.client("A"), served.client("B")
    opened = await served.open(a)  # nobody was attached when it was asked: the snapshot has it
    (ask,) = opened["snapshot"]["asks"]
    assert ask["kind"] == "mcp_trust" and ask["request"]["accept"] == "Trust and start them"
    stream = served.stream(a, opened["session_id"], after=opened["snapshot"]["seq"])

    first, second = await asyncio.gather(
        b.post(f"/v1/asks/{ask['ask_id']}", json={"trust": True}),
        a.post(f"/v1/asks/{ask['ask_id']}", json={"trust": False}),
    )
    assert sorted([first.status_code, second.status_code]) == [200, 409]
    connector = served.daemon.rt.mcp._connector
    await wait_until(lambda: [name for name, _ in connector.connects] == ["tools"])
    await stream.until(lambda s: bool(s.of("ask_resolved")))
    assert served.daemon.pending_asks() == []


async def test_a_gated_turn_is_interrupted_by_any_client(serve) -> None:
    gate = Gate()
    served = await serve([[gate, *REPLY]])
    a, b = served.client("A"), served.client("B")
    session_id = (await served.open(a))["session_id"]
    stream = served.stream(a, session_id, after=0)
    await a.post(f"/v1/sessions/{session_id}/messages", json={"text": "slow"})
    await asyncio.wait_for(gate.reached.wait(), 5)
    queued = await b.post(f"/v1/sessions/{session_id}/messages", json={"text": "later"})
    assert queued.json()["status"] == "queued"
    assert (await b.post(f"/v1/sessions/{session_id}/interrupt")).status_code == 204
    await stream.until(lambda s: "turn_outcome" in s.types())
    (outcome,) = stream.of("turn_outcome")
    assert outcome["status"] == "interrupted" and outcome["interrupted_by"] == "B"
    assert outcome["restores"] == [{"client": "B", "text": "later"}]
