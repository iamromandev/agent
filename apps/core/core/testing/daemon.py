"""Fixtures for tests that run a daemon in the test's own process (`data_home`, `serve`); a suite's conftest imports
them. The daemon tests need a short data folder: a Unix socket path must fit in about 100 bytes, and pytest's temp
folders on macOS are long."""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import tempfile
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from core.config import get_settings
from core.engine.commands.builtins import BUILTINS
from core.runtime import build_runtime
from core.server.daemon import Daemon
from core.server.files import bind_socket
from core.server.paths import socket_path
from core.server.serve import run_daemon
from core.testing.fakes import FakeCatalog, FakeConnector, Gate, ScriptedProvider
from core.testing.helpers import wait_until

Script = list[list[Any]]


@pytest.fixture
def data_home(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A short `XDG_DATA_HOME` of its own (the run folder goes under it), removed afterwards."""
    folder = Path(tempfile.mkdtemp(prefix="ag")).resolve()
    monkeypatch.setenv("XDG_DATA_HOME", str(folder))
    get_settings.cache_clear()
    try:
        yield folder
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        get_settings.cache_clear()


class Stream:
    """One client's SSE stream: the decoded messages, read in the background."""

    def __init__(self, client: httpx.AsyncClient, url: str) -> None:
        self.messages: list[dict[str, Any]] = []
        self.raw = bytearray()
        self.ended = asyncio.Event()
        self._task = asyncio.create_task(self._read(client, url))

    async def _read(self, client: httpx.AsyncClient, url: str) -> None:
        try:
            async with client.stream("GET", url, timeout=None) as response:
                self.status = response.status_code
                async for line in response.aiter_lines():
                    self.raw.extend(line.encode() + b"\n")
                    if line.startswith("data: "):
                        self.messages.append(json.loads(line[6:]))
        except (httpx.HTTPError, asyncio.CancelledError):
            pass
        finally:
            self.ended.set()

    def types(self) -> list[str]:
        return [m["type"] for m in self.messages]

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [m for m in self.messages if m["type"] == kind]

    async def until(self, predicate: Callable[[Stream], bool], timeout: float = 5.0) -> None:
        await wait_until(lambda: predicate(self), timeout)

    async def close(self) -> None:
        self._task.cancel()
        with contextlib.suppress(BaseException):
            await self._task


@dataclass
class Served:
    daemon: Daemon
    workspace: Path
    socket: Path
    provider: ScriptedProvider
    task: asyncio.Task[None]
    clients: list[httpx.AsyncClient]
    streams: list[Stream]

    def client(self, name: str = "A") -> httpx.AsyncClient:
        client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=str(self.socket)),
            base_url="http://agent",
            headers={"X-Agent-Client": name},
            timeout=10.0,
        )
        self.clients.append(client)
        return client

    def stream(self, client: httpx.AsyncClient, session_id: str, after: int | None = None) -> Stream:
        query = "" if after is None else f"?after={after}"
        stream = Stream(client, f"/v1/sessions/{session_id}/events{query}")
        self.streams.append(stream)
        return stream

    async def open(self, client: httpx.AsyncClient, resume: str | None = None) -> dict[str, Any]:
        response = await client.post("/v1/sessions", json={"resume": resume})
        assert response.status_code == 200, response.text
        return response.json()


@pytest.fixture
async def serve(data_home: Path, tmp_path: Path) -> AsyncIterator[Callable[..., Any]]:
    """Start a daemon in this process on a temp socket, over a scripted provider; stopped and closed at the end."""
    started: list[Served] = []

    async def start(script: Script | None = None, *, idle: float = 600.0, setup: Callable[[Path], None] | None = None, checkpoints: bool = True
    ) -> Served:
        ws = tmp_path / "ws"
        ws.mkdir(exist_ok=True)
        if setup is not None:
            setup(ws)
        provider = ScriptedProvider(list(script or []))
        rt = build_runtime(
            ws, BUILTINS, terminals=False, provider=provider, catalog=FakeCatalog(["fake", "other:7b"]), connector=FakeConnector(),
            checkpoints=checkpoints,
        )
        daemon = Daemon(rt, idle=idle)
        path = socket_path(ws)
        sock = bind_socket(path)
        task = asyncio.create_task(run_daemon(daemon, sock))
        served = Served(daemon, ws.resolve(), path, provider, task, [], [])
        started.append(served)
        hello = served.client("probe")
        await wait_until(lambda: daemon.server is not None and daemon.server.started)
        await hello.aclose()
        return served

    yield start
    for served in started:
        for stream in served.streams:
            await stream.close()
        for client in served.clients:
            await client.aclose()
        served.daemon.request_stop()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(served.task, 15)


__all__ = ["Gate", "Served", "Stream", "data_home", "serve"]
