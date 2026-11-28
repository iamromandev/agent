"""The HTTP client of the per-workspace daemon (`agent --shared`): JSON calls and the SSE event stream over its Unix
socket. It only moves bytes and dicts; turning them into engine events is `core.host.wire`'s, and what a client does
with them is the TUI's (`src.tui.remote`)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

PROTOCOL = "1"
CONNECT_TIMEOUT = 5.0  # seconds to reach the socket
CALL_TIMEOUT = 30.0  # seconds a JSON call may take (an undo or a diff reads the disk)


class DaemonError(Exception):
    """The daemon answered with an error status; `text` is its `error` field (or the body)."""

    def __init__(self, status: int, text: str) -> None:
        super().__init__(f"{status}: {text}")
        self.status = status
        self.text = text


class DaemonUnavailable(Exception):
    """The daemon could not be reached (no socket, refused, the connection dropped)."""


class ProtocolMismatch(Exception):
    """The daemon speaks another protocol version."""


def _error_text(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text.strip()[:200] or response.reason_phrase
    if isinstance(data, dict) and isinstance(data.get("error"), str):
        return data["error"]
    return response.reason_phrase


class DaemonClient:
    """One client of one daemon: every request names it in `X-Agent-Client`. Make it inside the event loop that uses
    it (httpx's connection pool belongs to that loop)."""

    def __init__(self, socket: Path, client_id: str) -> None:
        self.socket = socket
        self.client_id = client_id
        self._http = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=str(socket)),
            base_url="http://agent",
            headers={"X-Agent-Client": client_id},
            timeout=httpx.Timeout(CALL_TIMEOUT, connect=CONNECT_TIMEOUT),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _call(self, method: str, path: str, *, ok: tuple[int, ...] = (200,), **kwargs: Any) -> httpx.Response:
        try:
            response = await self._http.request(method, path, **kwargs)
        except httpx.TransportError as exc:
            raise DaemonUnavailable(str(exc) or type(exc).__name__) from exc
        if response.status_code not in ok:
            raise DaemonError(response.status_code, _error_text(response))
        return response

    async def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self._call(method, path, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            raise DaemonError(response.status_code, "the answer is not JSON") from exc

    async def _object(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """A JSON object, else `DaemonError` (a wrong shape is the daemon's fault, never a crash of the client)."""
        data = await self._json(method, path, **kwargs)
        if not isinstance(data, dict):
            raise DaemonError(502, "the daemon sent a bad answer (not an object)")
        return data

    async def _objects(self, method: str, path: str, **kwargs: Any) -> list[dict[str, Any]]:
        """A JSON list of objects, else `DaemonError`."""
        data = await self._json(method, path, **kwargs)
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise DaemonError(502, "the daemon sent a bad answer (not a list of objects)")
        return data

    async def hello(self) -> dict[str, Any]:
        """`/v1/hello`; `ProtocolMismatch` when the daemon speaks another version."""
        response = await self._call("GET", "/v1/hello")
        try:
            data = response.json()
        except ValueError as exc:
            raise DaemonError(response.status_code, "the answer is not JSON") from exc
        given = response.headers.get("x-agent-protocol", "?")
        if given != PROTOCOL or not isinstance(data, dict) or data.get("v") != int(PROTOCOL):
            version = data.get("version", "?") if isinstance(data, dict) else "?"
            raise ProtocolMismatch(f"daemon version {version} (protocol {given}), client protocol {PROTOCOL}: run agent stop")
        return data

    async def open_session(self, resume: str | None, model: str | None = None, source: str | None = None) -> dict[str, Any]:
        return await self._object("POST", "/v1/sessions", json={"resume": resume, "model": model, "source": source})

    async def list_sessions(self) -> list[dict[str, Any]]:
        return await self._objects("GET", "/v1/sessions")

    async def send(
        self, session_id: str, text: str, *, label: str | None = None, init: bool = False, command: bool = False
    ) -> dict[str, Any]:
        body = {"text": text, "label": label, "init": init, "command": command}
        return await self._object("POST", f"/v1/sessions/{session_id}/messages", json=body)

    async def interrupt(self, session_id: str) -> None:
        await self._call("POST", f"/v1/sessions/{session_id}/interrupt", ok=(204,))

    async def set_mode(self, session_id: str, mode: str) -> None:
        await self._call("PUT", f"/v1/sessions/{session_id}/mode", json={"mode": mode}, ok=(204,))

    async def undo(self, session_id: str) -> dict[str, Any]:
        return await self._object("POST", f"/v1/sessions/{session_id}/undo")

    async def compact(self, session_id: str) -> dict[str, Any]:
        return await self._object("POST", f"/v1/sessions/{session_id}/compact")

    async def switch_model(self, session_id: str, name: str) -> dict[str, Any]:
        return await self._object("POST", f"/v1/sessions/{session_id}/model", json={"name": name})

    async def info(self, session_id: str, name: str) -> dict[str, Any]:
        return await self._object("GET", f"/v1/sessions/{session_id}/info/{name}")

    async def commands(self) -> tuple[list[dict[str, Any]], list[str]]:
        """The daemon's custom commands (`name`, `source`, `description`) and the warnings from loading them."""
        data = await self._object("GET", "/v1/commands")
        rows = [row for row in data.get("commands") or () if isinstance(row, dict)]
        return rows, [w for w in data.get("warnings") or () if isinstance(w, str)]

    async def diff(self, session_id: str) -> dict[str, Any]:
        return await self._object("GET", f"/v1/sessions/{session_id}/diff")

    async def diff_file(self, session_id: str, index: int) -> dict[str, Any]:
        """The file at `index` of the last list this client asked for."""
        return await self._object("GET", f"/v1/sessions/{session_id}/diff/file", params={"index": str(index)})

    async def models(self) -> list[dict[str, Any]]:
        return await self._objects("GET", "/v1/models")

    async def model_length(self, name: str) -> int | None:
        """The model's context length as the daemon reads it (`None`: unknown); `DaemonError` on a bad answer."""
        length = (await self._object("GET", "/v1/model-length", params={"name": name})).get("context_length")
        if length is not None and (not isinstance(length, int) or isinstance(length, bool)):
            raise DaemonError(502, "the daemon sent a bad answer (context_length is not a number)")
        return length

    async def suggest(self, token: str) -> list[dict[str, Any]]:
        return await self._objects("GET", "/v1/files/suggest", params={"token": token})

    async def answer(self, ask_id: str, answer: dict[str, Any]) -> str:
        """`ok`, `already_answered`, or the error (`invalid`, `unknown`)."""
        response = await self._call("POST", f"/v1/asks/{ask_id}", json=answer, ok=(200, 400, 404, 409))
        if response.status_code == 409:
            return "already_answered"
        if response.status_code == 400:
            return "invalid"
        if response.status_code == 404:
            return "unknown"
        return "ok"

    async def events(self, session_id: str, after: int | None) -> AsyncIterator[dict[str, Any]]:
        """The session's event stream, one decoded `data:` message at a time (heartbeat comments skipped), after one
        empty dict that says the stream is open. Ends when the daemon ends it; `DaemonUnavailable` when the connection
        drops, `DaemonError` for an error status."""
        params = {} if after is None else {"after": str(after)}
        timeout = httpx.Timeout(None, connect=CONNECT_TIMEOUT)
        try:
            async with self._http.stream(
                "GET", f"/v1/sessions/{session_id}/events", params=params, timeout=timeout
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise DaemonError(response.status_code, _error_text(response))
                yield {}
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue  # `id:`, a blank line, or the `: ping` heartbeat
                    try:
                        data = json.loads(line[6:])
                    except ValueError:
                        continue
                    if isinstance(data, dict):
                        yield data
        except httpx.TransportError as exc:
            raise DaemonUnavailable(str(exc) or type(exc).__name__) from exc
