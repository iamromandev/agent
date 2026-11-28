"""Two clients of one daemon through `core.client.daemon.DaemonClient` (what `agent --shared` uses), over a real Unix
socket and a scripted provider: what one client does, the other sees (AC4, AC6, AC8, AC11, AC14)."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest
from core.client.daemon import DaemonClient, DaemonError, ProtocolMismatch
from core.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from core.testing.fakes import Gate
from core.testing.helpers import wait_until

REPLY = [TextDelta("ok"), Done(1, 1)]


class Watch:
    """A client's event stream read in the background, as decoded dicts."""

    def __init__(self, client: DaemonClient, session_id: str, after: int | None) -> None:
        self.messages: list[dict[str, Any]] = []
        self._task = asyncio.create_task(self._read(client, session_id, after))

    async def _read(self, client: DaemonClient, session_id: str, after: int | None) -> None:
        with contextlib.suppress(Exception):
            async for data in client.events(session_id, after):
                if data:  # the first one, `{}`, only says the stream is open
                    self.messages.append(data)

    def types(self) -> list[str]:
        return [m["type"] for m in self.messages]

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [m for m in self.messages if m["type"] == kind]

    async def until(self, kind: str, timeout: float = 5.0) -> None:
        await wait_until(lambda: kind in self.types(), timeout)

    async def close(self) -> None:
        self._task.cancel()
        with contextlib.suppress(BaseException):
            await self._task


@pytest.fixture
async def clients(serve):
    made: list[DaemonClient] = []
    watches: list[Watch] = []

    async def start(script=None, **kwargs):
        served = await serve(script, **kwargs)

        def client(name: str) -> DaemonClient:
            made.append(DaemonClient(served.socket, name))
            return made[-1]

        def watch(c: DaemonClient, session_id: str, after: int | None = 0) -> Watch:
            watches.append(Watch(c, session_id, after))
            return watches[-1]

        return served, client, watch

    yield start
    for w in watches:
        await w.close()
    for c in made:
        await c.aclose()


async def test_b_sees_as_message_reply_and_tool_blocks_in_the_same_order(clients) -> None:
    bash = [ToolCallEvent(ToolCall("b1", "bash", {"command": "true"})), Done(1, 1)]
    _served, client, watch = await clients([bash, REPLY])
    a, b = client("A"), client("B")
    await a.hello()
    session_id = (await a.open_session(None))["session_id"]
    seen_a, seen_b = watch(a, session_id), watch(b, session_id)
    await a.send(session_id, "run it")
    await seen_b.until("ask")
    assert await b.answer(seen_b.of("ask")[0]["ask_id"], {"choice": "once"}) == "ok"
    await seen_a.until("turn_outcome")
    await seen_b.until("turn_outcome")

    order = ["user_entry", "tool_started", "tool_finished", "assistant_text", "turn_ended", "turn_outcome"]
    for seen in (seen_a, seen_b):
        kinds = [k for k in seen.types() if k in order]
        assert [kinds.index(k) for k in order] == sorted(kinds.index(k) for k in order)
    assert seen_a.of("user_entry") == seen_b.of("user_entry")


async def test_an_ask_pending_with_no_client_is_in_the_snapshot_and_its_answer_continues_the_turn(clients) -> None:
    bash = [ToolCallEvent(ToolCall("b1", "bash", {"command": "true"})), Done(1, 1)]
    served, client, watch = await clients([bash, REPLY])
    a = client("A")
    session_id = (await a.open_session(None))["session_id"]
    await a.send(session_id, "run it")
    host = served.daemon.host(session_id)
    await wait_until(lambda: bool(host.asks.pending()))

    b = client("B")
    opened = await b.open_session(session_id)  # attaches to the live session
    (ask,) = [x for x in opened["snapshot"]["asks"] if x["kind"] == "approval"]
    seen = watch(b, session_id, opened["snapshot"]["seq"])
    assert await b.answer(ask["ask_id"], {"choice": "once"}) == "ok"
    await seen.until("turn_outcome")
    assert len(seen.of("tool_finished")) == 1


async def test_a_queued_message_and_a_mode_change_reach_every_client(clients) -> None:
    gate = Gate()
    _served, client, watch = await clients([[gate, *REPLY], REPLY])
    a, b = client("A"), client("B")
    session_id = (await a.open_session(None))["session_id"]
    seen_a, seen_b = watch(a, session_id), watch(b, session_id)
    await a.send(session_id, "slow")
    await asyncio.wait_for(gate.reached.wait(), 5)

    assert (await b.send(session_id, "later"))["status"] == "queued"
    await b.set_mode(session_id, "plan")
    for seen in (seen_a, seen_b):
        await seen.until("queue_changed")
        assert seen.of("queue_changed")[-1]["entries"] == [{"typed": "later", "label": "", "client": "B"}]
        await seen.until("mode_changed")
        assert seen.of("mode_changed")[-1]["mode"] == "plan"
    gate.release.set()


async def test_new_moves_only_the_client_that_asked(clients) -> None:
    served, client, watch = await clients([REPLY])
    a, b = client("A"), client("B")
    shared = (await a.open_session(None))["session_id"]
    await b.open_session(shared)
    seen_b = watch(b, shared)
    fresh = (await a.open_session(None))["session_id"]  # `/new` in A

    assert fresh != shared
    await b.send(shared, "still here")
    await seen_b.until("turn_outcome")
    assert served.daemon.host(shared).agent.history  # B's session got the turn
    assert not served.daemon.host(fresh).agent.history


async def test_a_client_that_goes_away_mid_turn_does_not_affect_the_turn(clients) -> None:
    gate = Gate()
    _served, client, watch = await clients([[gate, *REPLY]])
    a, b = client("A"), client("B")
    session_id = (await a.open_session(None))["session_id"]
    seen_a, seen_b = watch(a, session_id), watch(b, session_id)
    await a.send(session_id, "slow")
    await asyncio.wait_for(gate.reached.wait(), 5)
    await seen_a.close()  # A is killed
    await a.aclose()
    gate.release.set()
    await seen_b.until("turn_outcome")
    assert seen_b.of("turn_ended")[0]["status"] == "done"


async def test_a_daemon_of_another_protocol_is_named_and_an_error_status_is_a_daemon_error(clients, monkeypatch) -> None:
    import core.client.daemon as daemon_client

    _served, client, _ = await clients()
    a = client("A")
    with pytest.raises(DaemonError) as missing:
        await a.send("NOPE", "x")
    assert missing.value.status == 404
    monkeypatch.setattr(daemon_client, "PROTOCOL", "2")
    with pytest.raises(ProtocolMismatch, match="run agent stop"):
        await a.hello()


async def test_stream_latency_with_two_clients_stays_low(clients) -> None:
    """NFR3: 200 chunks reach two clients with a median latency under 50 ms and a p95 under 200 ms."""
    import statistics
    import time

    chunks = [TextDelta(f"<{n}>") for n in range(200)]
    served, client, _ = await clients([[*chunks, Done(1, 1)]])
    served.provider.delay = 0.005
    sent: dict[str, float] = {}
    real = served.provider.stream

    async def timed(messages, tools):
        async for event in real(messages, tools):
            if isinstance(event, TextDelta):
                sent[event.text] = time.monotonic()
            yield event

    served.provider.stream = timed
    a, b = client("A"), client("B")
    session_id = (await a.open_session(None))["session_id"]
    got: dict[str, dict[str, float]] = {"A": {}, "B": {}}

    async def read(name: str, c: DaemonClient) -> None:
        async for data in c.events(session_id, 0):
            if data.get("type") == "assistant_text":
                got[name].setdefault(data["delta"], time.monotonic())
            if data.get("type") == "turn_outcome":
                return

    readers = [asyncio.create_task(read("A", a)), asyncio.create_task(read("B", b))]
    await asyncio.sleep(0.1)
    await a.send(session_id, "go")
    await asyncio.wait_for(asyncio.gather(*readers), 30)
    latencies = sorted(got[name][text] - sent[text] for name in got for text in got[name] if text in sent)
    assert len(latencies) == 400
    assert statistics.median(latencies) < 0.05
    assert latencies[int(len(latencies) * 0.95)] < 0.2


def _lsp_setup(tmp_path, monkeypatch):
    import json
    import os
    from pathlib import Path

    from core.engine.lsp import server as server_module
    from core.testing.lsp_fakes import fake_lsp

    monkeypatch.setattr(server_module, "QUIET", 0.1)
    monkeypatch.setenv("FAKE_LSP_DUMP", str(tmp_path / "dump.jsonl"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = fake_lsp(bin_dir, "ty")

    def setup(_ws: Path) -> None:
        folder = Path(os.environ["XDG_CONFIG_HOME"]) / "agent"
        folder.mkdir(parents=True, exist_ok=True)
        servers = {"ty": {"command": str(fake), "extensions": [".py"]}}
        body = {"lsp": {"servers": servers}, "permissions": {"allow": ["write"]}}
        (folder / "settings.json").write_text(json.dumps(body))

    return setup


async def test_ac11_two_sessions_share_one_language_server_and_both_results_carry_the_block(
    clients, tmp_path, monkeypatch
) -> None:
    import json

    def write(call_id: str) -> list[Any]:
        return [ToolCallEvent(ToolCall(call_id, "write", {"path": "a.py", "content": "x = 2\n"})), Done(1, 1)]

    gate = Gate()
    # A's second request is held open, so B's whole turn runs inside A's
    served, client, watch = await clients(
        [write("w1"), [gate, *REPLY], write("w2"), REPLY], setup=_lsp_setup(tmp_path, monkeypatch)
    )
    a, b = client("A"), client("B")
    sid_a = (await a.open_session(None))["session_id"]
    sid_b = (await b.open_session(None))["session_id"]
    seen_a, seen_b = watch(a, sid_a), watch(b, sid_b)
    await a.send(sid_a, "edit")
    await asyncio.wait_for(gate.reached.wait(), 5)
    (served.workspace / ".agent").mkdir()
    (served.workspace / ".agent" / "settings.json").write_text("{}")  # a change made while both sessions run
    await b.send(sid_b, "edit")
    await seen_b.until("turn_outcome")
    gate.release.set()
    await seen_a.until("turn_outcome")

    for seen in (seen_a, seen_b):
        (finished,) = seen.of("tool_finished")
        assert "[diagnostics from ty:" in str(finished)
        assert str(seen.messages).count("config changed by") == 1  # one change, one warning
    dump = tmp_path / "dump.jsonl"
    assert len({json.loads(x)["pid"] for x in dump.read_text().splitlines()}) == 1
