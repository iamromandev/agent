"""`RemotePort` without the app, against a daemon in the test: the reconnect paths (a dropped stream, a new daemon, an
unsaved session that is gone, an overflow, a reconnect that fails) and a daemon that answers with a wrong shape."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
import src.tui.remote as remote
from core.client.daemon import DaemonClient, DaemonError
from core.engine import Notice
from core.engine.provider.base import Done, TextDelta
from core.host.bus import Overflow
from core.host.events import TurnOutcome
from core.testing.daemon import data_home, serve
from core.testing.helpers import wait_until
from src.tui.remote import LOST, UNSAVED_LOST, DaemonGone, Reconnected, RemotePort

__all__ = ["data_home", "serve"]

REPLY = [TextDelta("ok"), Done(1, 1)]


class Seen:
    """What the port hands the app."""

    def __init__(self) -> None:
        self.events: list[object] = []
        self.asks: list[object] = []

    async def on_event(self, event: object) -> None:
        self.events.append(event)

    def notices(self) -> list[str]:
        return [e.text for e in self.events if isinstance(e, Notice)]

    def of(self, kind: type) -> list[Any]:
        return [e for e in self.events if isinstance(e, kind)]


@pytest.fixture(autouse=True)
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote, "RECONNECT_WAIT", 3.0)
    monkeypatch.setattr(remote, "RETRY_STEP", 0.05)


async def _port(served, seen: Seen, resume: str | None = None) -> RemotePort:
    port = RemotePort(served.socket, served.workspace, client_id="A")
    await port.connect()
    await port.open(resume)
    port.subscribe(seen.on_event, seen.asks.append)
    return port


async def _restart(serve, served):
    served.daemon.request_stop()
    await asyncio.wait_for(served.task, 10)
    return await serve([REPLY])


async def test_a_new_daemon_resumes_the_session_from_its_file_and_the_client_redraws_from_a_snapshot(serve) -> None:
    served = await serve([REPLY])
    seen = Seen()
    port = await _port(served, seen)
    try:
        session_id = port.session_id
        assert (await port.send_message("hi")).status == "started"
        await wait_until(lambda: bool(seen.of(TurnOutcome)))
        old_pid = port._daemon_pid
        await _restart(serve, served)
        await wait_until(lambda: any(r.snapshot is not None for r in seen.of(Reconnected)), 8)
        (back,) = [r for r in seen.of(Reconnected) if r.snapshot is not None]
        assert LOST in seen.notices() and port.session_id == session_id
        assert back.snapshot.records and port._daemon_pid != old_pid and old_pid is not None
    finally:
        await port.close()


async def test_an_unsaved_session_the_daemon_lost_becomes_a_new_one_with_a_notice(serve) -> None:
    served = await serve()
    seen = Seen()
    port = await _port(served, seen)
    try:
        lost = port.session_id
        await _restart(serve, served)
        await wait_until(lambda: UNSAVED_LOST in seen.notices(), 8)
        assert port.session_id != lost and any(r.snapshot is not None for r in seen.of(Reconnected))
    finally:
        await port.close()


async def test_a_dropped_stream_comes_back_and_says_so(serve) -> None:
    served = await serve()
    seen = Seen()
    port = await _port(served, seen)
    try:
        host = served.daemon.host(port.session_id)
        await wait_until(lambda: bool(host.bus.subscribers))
        for sub in list(host.bus.subscribers):
            host.bus.unsubscribe(sub)
        await wait_until(lambda: Reconnected(None) in seen.events)
        assert seen.notices().index(LOST) >= 0
    finally:
        await port.close()


async def test_an_overflow_starts_over_from_a_snapshot(serve) -> None:
    served = await serve()
    seen = Seen()
    port = await _port(served, seen)
    try:
        host = served.daemon.host(port.session_id)
        await wait_until(lambda: bool(host.bus.subscribers))
        (sub,) = host.bus.subscribers
        sub.overflow = Overflow(host.bus.seq)
        host.bus.unsubscribe(sub)
        await wait_until(lambda: any(r.snapshot is not None for r in seen.of(Reconnected)))
    finally:
        await port.close()


async def test_a_reconnect_that_fails_ends_in_daemon_gone_with_its_reason_and_keep_waiting_tries_again(
    serve, monkeypatch
) -> None:
    served = await serve()
    seen = Seen()
    port = await _port(served, seen)
    try:
        tries = 0

        async def broken() -> bool:
            nonlocal tries
            tries += 1
            raise KeyError("pid")

        monkeypatch.setattr(port, "_come_back", broken)
        host = served.daemon.host(port.session_id)
        await wait_until(lambda: bool(host.bus.subscribers))
        for sub in list(host.bus.subscribers):
            host.bus.unsubscribe(sub)
        await wait_until(lambda: bool(seen.of(DaemonGone)))
        assert "reconnecting to the agent daemon failed" in seen.of(DaemonGone)[0].reason
        assert port._reader is not None and not port._reader.done()  # never dies silently
        port.keep_waiting()  # the daemon is fine: the stream comes back
        await wait_until(lambda: Reconnected(None) in seen.events)
        assert tries == 1
    finally:
        await port.close()


def _answering(body: bytes) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://agent")


@pytest.mark.parametrize("body", [b"{nope", b"[1, 2]", b'[{"id": "x"}]', b'{"id": 1}'])
async def test_a_daemon_answer_of_the_wrong_shape_is_a_daemon_error(tmp_path, body: bytes) -> None:
    port = RemotePort(tmp_path / "s.sock", tmp_path)
    client = DaemonClient(tmp_path / "s.sock", "A")
    await client._http.aclose()
    client._http = _answering(body)
    port._client = client
    try:
        with pytest.raises(DaemonError):
            await port.list_sessions()
    finally:
        await client.aclose()


async def test_another_daemon_that_already_has_the_session_live_is_followed_from_a_snapshot(serve) -> None:
    served = await serve([REPLY])
    seen = Seen()
    port = await _port(served, seen)
    try:
        session_id = port.session_id
        await port.send_message("hi")
        await wait_until(lambda: bool(seen.of(TurnOutcome)))
        held = asyncio.Event()
        client = port._http()
        real = client.hello

        async def slow_hello() -> dict:
            await held.wait()  # this client comes back only once the new daemon is well past its old numbering
            return await real()

        client.hello = slow_hello  # ty: ignore[invalid-assignment]
        fresh = await _restart(serve, served)
        other = DaemonClient(fresh.socket, "B")
        try:
            await other.open_session(session_id)
            host = fresh.daemon.host(session_id)
            for n in range(port._last_seq + 5):
                host.bus.publish(Notice("info", f"new daemon {n}"))
        finally:
            await other.aclose()
        held.set()
        await wait_until(lambda: any(r.snapshot is not None for r in seen.of(Reconnected)), 8)
        assert not any(t.startswith("new daemon") for t in seen.notices())  # no events of another numbering
    finally:
        await port.close()
