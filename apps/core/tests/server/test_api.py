"""The daemon's HTTP API over a real Unix socket, driven with httpx against a daemon over a scripted provider."""

from __future__ import annotations

import asyncio
import json
import os
import socket
from pathlib import Path

from core.engine.configguard import GlobalMarker, marker_path
from core.engine.events import Notice, WorkflowLine
from core.engine.lsp.config import BUILTINS
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


def _secret_hook(ws: Path) -> None:
    (ws / ".agent").mkdir()
    settings = {"hooks": {"PreToolUse": [{"command": f"echo {SECRET}"}]}, "permissions": {"allow": [f"bash(echo {SECRET})"]}}
    (ws / ".agent" / "settings.json").write_text(json.dumps(settings))


async def test_the_info_texts_are_redacted(serve) -> None:
    served = await serve(setup=_secret_hook)
    client = served.client()
    session_id = (await served.open(client))["session_id"]
    for name in ("hooks", "permissions", "mcp", "guardrails", "skills", "agents", "usage"):
        got = await client.get(f"/v1/sessions/{session_id}/info/{name}")
        assert got.status_code == 200 and SECRET not in got.text, name
    hooks = (await client.get(f"/v1/sessions/{session_id}/info/hooks")).json()["lines"][0]["text"]
    assert "[redacted:github-token]" in hooks


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


MINE = {"command": "/bin/sh", "args": ["-c", "x"], "extensions": [".m"], "env": {"NODE_OPTIONS": "-r y"}}
# the built-ins are switched off: which of them is on PATH is not this test's business
LSP_SERVER = {"lsp": {"servers": {**{b.name: False for b in BUILTINS}, "mine": MINE}}}


def _project_lsp(ws: Path) -> None:
    (ws / ".agent").mkdir()
    (ws / ".agent" / "settings.json").write_text(json.dumps(LSP_SERVER))


def _held_global_lsp(ws: Path) -> None:
    folder = Path(os.environ["XDG_CONFIG_HOME"]) / "agent"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "settings.json").write_text(json.dumps(LSP_SERVER))
    GlobalMarker(marker_path()).write(["~/.config/agent/settings.json"])


async def test_the_daemons_project_trust_ask_lists_the_lsp_server(serve) -> None:
    served = await serve(setup=_project_lsp)
    (ask,) = (await served.open(served.client("A")))["snapshot"]["asks"]
    assert ask["kind"] == "permission_trust"
    assert ask["request"]["lines"] == ["lsp mine: /bin/sh -c x  (env: NODE_OPTIONS)"]


async def test_the_daemons_global_trust_ask_lists_the_held_global_lsp_server(serve) -> None:
    served = await serve(setup=_held_global_lsp)
    asks = (await served.open(served.client("A")))["snapshot"]["asks"]
    (ask,) = [a for a in asks if a["kind"] == "global_trust"]
    assert "lsp mine: /bin/sh -c x  (env: NODE_OPTIONS)" in ask["request"]["lines"]


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


async def test_a_session_that_cannot_be_read_is_500_with_its_reason_not_locked(serve, monkeypatch) -> None:
    from core.engine.session import SessionError

    served = await serve()
    folder = served.daemon._sessions
    recorder = FileRecorder(folder, served.workspace)
    recorder.message(Message("user", "x"), None)
    session_id = recorder.session_id
    recorder.close()

    def broken(_: str) -> None:
        raise SessionError("Session X has an unreadable header")

    monkeypatch.setattr(served.daemon.rt.store, "load", broken)
    refused = await served.client().post("/v1/sessions", json={"resume": session_id})
    assert refused.status_code == 500 and "unreadable header" in refused.json()["error"]
    assert served.daemon.hosts == {} and not SessionLock.held_by_us(folder, session_id)


async def test_an_unknown_session_is_404(serve) -> None:
    served = await serve()
    missing = await served.client().post("/v1/sessions", json={"resume": "01ZZZZZZZZZZZZ"})
    assert missing.status_code == 404


async def test_a_session_goes_with_its_last_client_and_so_does_its_diff_and_lock(serve) -> None:
    served = await serve([REPLY])
    served.daemon.drop_grace = 0.05
    a = served.client("A")
    empty = (await served.open(a))["session_id"]
    used = (await served.open(a))["session_id"]
    streams = [served.stream(a, empty, after=0), served.stream(a, used, after=0)]
    await a.post(f"/v1/sessions/{used}/messages", json={"text": "hi"})
    await streams[1].until(lambda s: "turn_outcome" in s.types())
    await a.get(f"/v1/sessions/{used}/diff")
    for stream in streams:
        await stream.close()

    await wait_until(lambda: empty not in served.daemon.hosts and used not in served.daemon.hosts)
    await wait_until(lambda: served.daemon.diffs == {})
    await wait_until(lambda: not SessionLock.held_by_us(served.daemon._sessions, used))  # plain `agent` may open it
    with session_held_elsewhere(served.daemon._sessions, used):  # another process (plain `agent`) takes it
        pass
    again = await served.open(served.client("B"), resume=used)  # and so may the daemon again, from its file
    assert again["snapshot"]["records"]


async def test_a_session_is_kept_a_moment_for_a_client_whose_stream_is_not_up_yet(serve) -> None:
    served = await serve()
    served.daemon.drop_grace = 0.5
    a, b = served.client("A"), served.client("B")
    session_id = (await served.open(a))["session_id"]
    first = served.stream(a, session_id, after=0)
    await wait_until(lambda: served.daemon.connections.get((session_id, "A")) == 1)
    await first.close()
    await b.post("/v1/sessions", json={"resume": session_id})  # B joins before the grace is over
    second = served.stream(b, session_id, after=0)
    await wait_until(lambda: served.daemon.connections.get((session_id, "B")) == 1)
    await asyncio.sleep(0.7)
    assert session_id in served.daemon.hosts and not second.ended.is_set()


async def test_a_failing_snapshot_leaks_no_subscriber_or_connection(serve, monkeypatch) -> None:
    import core.server.app as app_module

    served = await serve()
    client = served.client()
    session_id = (await served.open(client))["session_id"]
    host = served.daemon.host(session_id)

    def broken(*args, **kwargs):
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(app_module, "snapshot", broken)
    stream = served.stream(client, session_id)  # no `after`: it starts with a snapshot
    await asyncio.wait_for(stream.ended.wait(), 5)
    assert host.bus.subscribers == [] and served.daemon.connections == {}


async def test_an_outcome_that_could_not_be_sent_is_not_acknowledged(serve, monkeypatch) -> None:
    import core.server.app as app_module
    from core.host.events import TurnOutcome

    real = app_module.encode
    monkeypatch.setattr(app_module, "encode", lambda item, **kw: None if isinstance(item, TurnOutcome) else real(item, **kw))
    served = await serve([REPLY])
    client = served.client()
    session_id = (await served.open(client))["session_id"]
    host = served.daemon.host(session_id)
    stream = served.stream(client, session_id, after=0)
    await client.post(f"/v1/sessions/{session_id}/messages", json={"text": "hi"})
    await stream.until(lambda s: "turn_ended" in s.types())
    await wait_until(lambda: host._ending != 0)
    await asyncio.sleep(0.2)  # the stream has handled the outcome by now
    assert host._ending != 0  # still held: only the 2 s timeout releases it


async def test_a_model_switch_in_one_session_is_told_to_every_session(serve) -> None:
    served = await serve()
    a, b = served.client("A"), served.client("B")
    one = (await served.open(a))["session_id"]
    two = (await served.open(b))["session_id"]
    stream_one = served.stream(a, one, after=0)
    stream_two = served.stream(b, two, after=0)
    await stream_one.until(lambda s: True)

    switched = (await b.post(f"/v1/sessions/{two}/model", json={"name": "other:7b"})).json()
    assert switched["model"] == "other:7b"
    await stream_one.until(lambda s: bool(s.of("model_changed")))
    assert stream_one.of("model_changed")[0]["model"] == "other:7b"
    assert any(m["text"] == "model switched to other:7b by another session" for m in stream_one.of("notice"))
    await stream_two.until(lambda s: bool(s.of("model_changed")))
    assert not any("by another session" in m["text"] for m in stream_two.of("notice"))  # its own notices say it
    again = await served.open(served.client("C"), resume=None)
    assert again["snapshot"]["model"] == "other:7b"


async def test_the_info_texts_are_the_same_the_app_shows_and_help_is_still_the_clients(serve) -> None:
    from core import info

    served = await serve()
    client = served.client()
    session_id = (await served.open(client))["session_id"]
    rt = served.daemon.rt
    expected = {
        "skills": info.skills_summary(rt.skills),
        "agents": info.agents_summary(rt.agents),
        "permissions": info.permissions_summary(rt.permissions.rules),
        "hooks": info.hooks_summary(rt.hooks),
        "mcp": info.mcp_summary(rt.mcp),
        "guardrails": info.guardrails_summary(rt.guard, rt.config_guard),
    }
    for name, text in expected.items():
        got = await client.get(f"/v1/sessions/{session_id}/info/{name}")
        assert got.status_code == 200 and got.json() == {"lines": [{"level": "info", "text": text}]}, name
    assert "commit (builtin)" in expected["skills"]
    assert (await client.get(f"/v1/sessions/{session_id}/info/help")).status_code == 501


async def test_an_undo_reaches_every_client_of_the_session(serve) -> None:
    served = await serve([REPLY])
    a, b = served.client("A"), served.client("B")
    session_id = (await served.open(a))["session_id"]
    stream_a, stream_b = served.stream(a, session_id, after=0), served.stream(b, session_id, after=0)
    await a.post(f"/v1/sessions/{session_id}/messages", json={"text": "hi"})
    await stream_b.until(lambda s: "turn_outcome" in s.types())
    host = served.daemon.host(session_id)
    await wait_until(lambda: not host.turn_active)
    undone = (await a.post(f"/v1/sessions/{session_id}/undo")).json()
    assert undone["prompt"] == "hi"
    for stream in (stream_a, stream_b):
        await stream.until(lambda s: bool(s.of("turn_undone")))
        assert stream.of("turn_undone")[0]["client"] == "A"
        await stream.until(lambda s: bool(s.of("todos_changed")))


async def test_the_file_index_is_rebuilt_after_a_turn(serve) -> None:
    served = await serve([REPLY])
    client = served.client()
    session_id = (await served.open(client))["session_id"]
    await wait_until(lambda: served.daemon.file_index.ready)
    (served.workspace / "fresh_file.txt").write_text("x")
    stream = served.stream(client, session_id, after=0)
    await client.post(f"/v1/sessions/{session_id}/messages", json={"text": "hi"})
    await stream.until(lambda s: "turn_outcome" in s.types())

    async def found() -> bool:
        got = (await client.get("/v1/files/suggest", params={"token": "fresh"})).json()
        return any(s["path"] == "fresh_file.txt" for s in got)

    for _ in range(100):
        if await found():
            break
        await asyncio.sleep(0.05)
    assert await found()


async def test_diff_is_refused_during_a_turn_and_a_file_is_found_by_its_index(serve) -> None:
    gate = Gate()
    served = await serve([[gate, *REPLY]])
    client = served.client()
    session_id = (await served.open(client))["session_id"]
    await client.post(f"/v1/sessions/{session_id}/messages", json={"text": "slow"})
    await asyncio.wait_for(gate.reached.wait(), 5)
    busy = await client.get(f"/v1/sessions/{session_id}/diff")
    assert busy.status_code == 409 and "busy" in busy.json()["error"]
    gate.release.set()
    missing = await client.get(f"/v1/sessions/{session_id}/diff/file", params={"index": "0"})
    assert missing.status_code in (404, 409)


async def test_model_length_reads_the_catalog_and_needs_a_name(serve) -> None:
    served = await serve()
    served.daemon.rt.catalog.lengths["fake"] = 8192
    client = served.client()
    assert (await client.get("/v1/model-length", params={"name": "fake"})).json() == {"context_length": 8192}
    assert (await client.get("/v1/model-length", params={"name": "nope"})).json() == {"context_length": None}
    assert (await client.get("/v1/model-length")).status_code == 400
