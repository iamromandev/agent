"""The daemon's HTTP API on its Unix socket (Starlette). Every handler only translates: the work is the daemon's and the
host's, the shapes are the wire's. Each response carries `X-Agent-Protocol: 1`; every request but `/v1/hello` names
its client in `X-Agent-Client`."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from core.engine.compaction import turn_starts
from core.engine.mode import Mode
from core.engine.provider.models import CatalogError
from core.engine.turn_diff import DiffItem
from core.engine.usage import summary
from core.host.asks import answer_from_wire
from core.host.events import TurnOutcome
from core.host.session import SessionHost
from core.host.wire import encode
from core.server.daemon import Daemon, OpenFailed, SessionLockedElsewhere
from core.server.snapshot import redacted, snapshot
from core.wire import models as wm

PROTOCOL = "1"
VERSION = "0.0.1"
HEARTBEAT = 15.0  # seconds between two SSE comments on a quiet stream
CLIENT = re.compile(r"[A-Za-z0-9_-]{1,64}")
INFO_NAMES = frozenset({"usage", "permissions", "mcp", "hooks", "guardrails", "skills", "agents", "help"})

_ANSWER: TypeAdapter[wm.Answer] = TypeAdapter(wm.Answer)


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpenBody(_Body):
    resume: str | None = None
    model: str | None = None  # the daemon's session decides the model; the client is told when it differs


class MessageBody(_Body):
    text: str
    label: str | None = None
    init: bool = False


class ModeBody(_Body):
    mode: Mode


class ModelBody(_Body):
    name: str


class ApiError(Exception):
    def __init__(self, status: int, text: str) -> None:
        super().__init__(text)
        self.status = status
        self.text = text


def _error(status: int, text: str) -> JSONResponse:
    return JSONResponse({"error": text}, status_code=status)


class ProtocolHeader:
    """Adds `X-Agent-Protocol` to every response (pure ASGI, so a stream is never buffered)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def tagged(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append((b"x-agent-protocol", PROTOCOL.encode()))
            await send(message)

        await self.app(scope, receive, tagged)


def build_app(daemon: Daemon) -> ASGIApp:
    def client_of(request: Request) -> str:
        client = request.headers.get("x-agent-client", "")
        if not CLIENT.fullmatch(client):
            raise ApiError(400, "X-Agent-Client is missing or not a client id")
        return client

    async def body(request: Request, model: type[BaseModel]) -> Any:
        try:
            data = json.loads(await request.body() or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ApiError(400, "the body is not valid JSON") from None
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            raise ApiError(400, f"bad body: {exc.error_count()} errors") from None

    def session(request: Request) -> tuple[str, SessionHost]:
        session_id = request.path_params["id"]
        host = daemon.host(session_id)
        if host is None:
            raise ApiError(404, "unknown session")
        return session_id, host

    def wire(item: object, host: SessionHost) -> dict[str, Any] | None:
        return encode(item, session_id=host.session_id, seq=0, redact=daemon.redact)

    def snap(host: SessionHost, seq: int) -> dict[str, Any]:
        data = snapshot(host, seq=seq, redact=daemon.redact, extra_asks=daemon.pending_asks())
        return data.model_dump(mode="json")

    def handler(fn: Any) -> Any:
        async def run(request: Request) -> Response:
            try:
                return await fn(request)
            except ApiError as exc:
                return _error(exc.status, exc.text)

        return run

    async def hello(request: Request) -> Response:
        return JSONResponse(
            {"v": 1, "version": VERSION, "pid": os.getpid(), "instance": daemon.instance, "workspace": str(daemon.workspace)}
        )

    async def list_sessions(request: Request) -> Response:
        client_of(request)
        live = daemon.live_ids()
        infos = await asyncio.to_thread(daemon.rt.store.list)
        return JSONResponse(
            [
                {
                    "id": info.id,
                    "title": daemon.redact(info.title),
                    "created_at": info.created_at.isoformat(),
                    "updated_at": info.updated_at.isoformat(),
                    "message_count": info.message_count,
                    "open_elsewhere": info.open_elsewhere and info.id not in live,
                    "live": info.id in live,
                }
                for info in infos
            ]
        )

    async def open_session(request: Request) -> Response:
        client_of(request)
        opened: OpenBody = await body(request, OpenBody)
        try:
            host, created = await daemon.open(opened.resume)
        except SessionLockedElsewhere as exc:
            raise ApiError(409, f"locked by another process: {exc}") from None
        except OpenFailed as exc:
            raise ApiError(exc.status, str(exc)) from None
        # A session made now has only its first notices on its bus: the client streams them from 0.
        return JSONResponse({"session_id": host.session_id, "snapshot": snap(host, 0 if created else host.bus.seq)})

    async def events(request: Request) -> Response:
        client = client_of(request)
        session_id, host = session(request)
        raw = request.query_params.get("after")
        try:
            after = int(raw) if raw is not None else None
        except ValueError:
            raise ApiError(400, "after must be a number") from None

        def line(data: dict[str, Any], seq: int) -> bytes:
            return f"id: {seq}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()

        async def stream() -> AsyncIterator[bytes]:
            # Subscribed and counted only once the stream runs, and released in its `finally`: a client that went
            # before the body started, or a snapshot that failed, leaves no subscriber and no connection behind.
            sub = host.bus.subscribe(remote=True)  # first, then the backlog: nothing in between can be missed
            attached = False
            try:
                backlog = host.bus.since(after) if after is not None else None
                start_seq = host.bus.seq
                first = snap(host, start_seq) if backlog is None else None
                daemon.attach(session_id, client)
                attached = True
                last = start_seq if backlog is None else (after or 0)
                if first is not None:  # too far behind (or a first attach): the state, then live events
                    yield line(first, start_seq)
                else:
                    for seq, item in backlog or ():
                        data = encode(item, session_id=host.session_id, seq=seq, redact=daemon.redact)
                        if data is not None:  # an event with no wire form is skipped
                            yield line(data, seq)
                        last = seq
                stopping = asyncio.ensure_future(daemon.stopping.wait())
                gone = asyncio.ensure_future(sub.done.wait())
                try:
                    while True:
                        if sub.queue.empty():
                            if sub.dropped:  # its queue was full: it must catch up from a snapshot
                                yield line(wm.Overflow(session_id=host.session_id, seq=last).model_dump(mode="json"), last)
                                return
                            if daemon.stopping.is_set():
                                closed = wm.SessionClosed(session_id=host.session_id, seq=last, reason="daemon stopping")
                                yield line(closed.model_dump(mode="json"), last)
                                return
                            if sub.done.is_set():  # unsubscribed some other way: nothing more will come
                                return
                            get = asyncio.ensure_future(sub.queue.get())
                            done, _ = await asyncio.wait({get, stopping, gone}, timeout=HEARTBEAT, return_when=asyncio.FIRST_COMPLETED)
                            if get not in done:
                                get.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await get
                                if get.cancelled():
                                    if not done:
                                        yield b": ping\n\n"
                                    continue
                            seq, item = get.result()
                        else:
                            seq, item = sub.queue.get_nowait()
                        try:
                            if seq <= last:
                                continue  # already sent from the backlog
                            data = encode(item, session_id=host.session_id, seq=seq, redact=daemon.redact)
                            last = seq
                            if data is None:  # an event with no wire form (a workflow's) is skipped
                                continue
                            yield line(data, seq)
                            # Written: its own text is on the way to its box (one that could not be sent is not
                            # acknowledged, so the claim waits for the timeout instead of dropping the text).
                            if isinstance(item, TurnOutcome) and item.ack and item.client == client:
                                host.outcome_applied(item.ack, client)
                        finally:
                            sub.queue.task_done()
                finally:
                    stopping.cancel()
                    gone.cancel()
            finally:
                if attached:
                    daemon.detach(host, session_id, client, sub)
                else:
                    host.bus.unsubscribe(sub)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    async def send(request: Request) -> Response:
        client = client_of(request)
        _, host = session(request)
        message: MessageBody = await body(request, MessageBody)
        if message.init:
            result = host.send(await asyncio.to_thread(host.init_prompt), init=True, draft=message.text, client=client)
        else:
            result = host.send(message.text, label=message.label, client=client)
        return JSONResponse({"status": result.status, "reason": result.reason or None})

    async def interrupt(request: Request) -> Response:
        client = client_of(request)
        _, host = session(request)
        host.interrupt(client)
        return Response(status_code=204)

    async def set_mode(request: Request) -> Response:
        client_of(request)
        _, host = session(request)
        mode: ModeBody = await body(request, ModeBody)
        host.set_mode(mode.mode)
        return Response(status_code=204)

    async def undo(request: Request) -> Response:
        client = client_of(request)
        _, host = session(request)
        result = await host.undo(client)
        return JSONResponse({"prompt": result.prompt, "events": [d for e in result.events if (d := wire(e, host))]})

    async def compact(request: Request) -> Response:
        client_of(request)
        _, host = session(request)
        result = host.compact()
        return JSONResponse({"status": result.status, "reason": result.reason or None})

    async def switch_model(request: Request) -> Response:
        client = client_of(request)
        _, host = session(request)
        named: ModelBody = await body(request, ModelBody)
        result = await daemon.switch_model(host, named.name, client)
        notices = [{"level": level, "text": daemon.redact(text)} for level, text in result.notices]
        return JSONResponse({"notices": notices, "model": result.model, "window": result.window})

    async def info(request: Request) -> Response:
        client_of(request)
        _, host = session(request)
        name = request.path_params["name"]
        if name not in INFO_NAMES:
            raise ApiError(404, f"no info named {name}")
        if name != "usage":
            text = daemon.info_text(name)
            if text is None:
                raise ApiError(501, f"/{name} is not available in shared mode yet")
            return JSONResponse({"lines": [{"level": "info", "text": daemon.redact(text)}]})
        agent = host.agent
        data = summary(
            agent.usage,
            session_id=host.session_id,
            model=host.model,
            context_used=agent.estimate(),
            window=host.provider.context_window,
            turns=len(turn_starts(agent.history)),
            unrecorded=agent.usage_unrecorded,
            estimated=True,
        )
        return JSONResponse({"usage": redacted(asdict(data), daemon.redact)})

    diffs = daemon.diffs  # the last list each client asked for: its files are loaded from it

    def item_json(item: DiffItem) -> dict[str, Any]:
        return {
            "path": daemon.redact(item.path),
            "status": item.status,
            "added": item.added,
            "removed": item.removed,
            "note": item.note,
            "old_mode": item.old_mode,
            "new_mode": item.new_mode,
            "old_size": item.old_size,
            "new_size": item.new_size,
        }

    async def diff_list(request: Request) -> Response:
        client = client_of(request)
        session_id, host = session(request)
        if (reason := host.refusal("/diff")) is not None:
            raise ApiError(409, reason)
        diff = await host.diff_list()
        if diff is None:
            diffs.pop((session_id, client), None)
            return JSONResponse({"diff": None})
        diffs[(session_id, client)] = diff
        return JSONResponse(
            {"diff": {"items": [item_json(i) for i in diff.items], "more": diff.more, "fallback": diff.fallback}}
        )

    async def diff_file(request: Request) -> Response:
        client = client_of(request)
        session_id, host = session(request)
        if (reason := host.refusal("/diff")) is not None:
            raise ApiError(409, reason)
        diff = diffs.get((session_id, client))
        # By its place in the list: the path the client got is redacted, so it may not match the file's.
        raw = request.query_params.get("index", "")
        index = int(raw) if raw.isdigit() else -1
        item = diff.items[index] if diff is not None and 0 <= index < len(diff.items) else None
        if diff is None or item is None:
            raise ApiError(404, "no such file in the last diff (ask for the list first)")
        loaded = await host.diff_file(diff, item)
        return JSONResponse(
            {
                "item": item_json(loaded.item),
                "meta": [daemon.redact(m) for m in loaded.meta],
                "lines": [daemon.redact(line) for line in loaded.lines],  # file content: redacted like a tool result
                "cut": loaded.cut,
            }
        )

    async def models(request: Request) -> Response:
        client_of(request)
        hosts = list(daemon.hosts.values())
        catalog = daemon.rt.catalog
        try:
            infos = await (hosts[0].list_models() if hosts else catalog.list())
        except CatalogError as exc:
            raise ApiError(502, str(exc)) from None
        return JSONResponse([asdict(i) for i in infos])

    async def model_length(request: Request) -> Response:
        """`{"context_length": int | null}` for `?name=` (null: Ollama reports none); 502 when it cannot be read."""
        client_of(request)
        name = request.query_params.get("name", "")
        if not name:
            raise ApiError(400, "name is required")
        try:
            length = await daemon.rt.catalog.context_length(name)
        except CatalogError as exc:
            raise ApiError(502, str(exc)) from None
        return JSONResponse({"context_length": length})

    async def suggest(request: Request) -> Response:
        client_of(request)
        token = request.query_params.get("token", "")
        found = daemon.file_index.suggest(token) if daemon.file_index.ready else []
        return JSONResponse([{"path": s.path, "folder": s.folder} for s in found])

    async def answer(request: Request) -> Response:
        client = client_of(request)
        try:
            data = json.loads(await request.body() or b"null")
            given = _ANSWER.validate_python(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ApiError(400, "the body is not valid JSON") from None
        except ValidationError:
            raise ApiError(400, "not an answer") from None
        result = daemon.answer(request.path_params["ask_id"], answer_from_wire(given), client)
        match result:
            case "ok":
                return JSONResponse({"result": "ok"})
            case "already_answered":
                return JSONResponse({"result": "already_answered", "error": "already answered"}, status_code=409)
            case "invalid":
                raise ApiError(400, "that answer is not offered by the ask")
            case _:
                raise ApiError(404, "unknown ask")

    async def stop(request: Request) -> Response:
        asyncio.get_running_loop().call_soon(daemon.request_stop)  # after this response is on its way
        return JSONResponse({"stopping": True}, status_code=202)

    routes = [
        Route("/v1/hello", handler(hello), methods=["GET"]),
        Route("/v1/sessions", handler(list_sessions), methods=["GET"]),
        Route("/v1/sessions", handler(open_session), methods=["POST"]),
        Route("/v1/sessions/{id}/events", handler(events), methods=["GET"]),
        Route("/v1/sessions/{id}/messages", handler(send), methods=["POST"]),
        Route("/v1/sessions/{id}/interrupt", handler(interrupt), methods=["POST"]),
        Route("/v1/sessions/{id}/mode", handler(set_mode), methods=["PUT"]),
        Route("/v1/sessions/{id}/undo", handler(undo), methods=["POST"]),
        Route("/v1/sessions/{id}/compact", handler(compact), methods=["POST"]),
        Route("/v1/sessions/{id}/model", handler(switch_model), methods=["POST"]),
        Route("/v1/sessions/{id}/info/{name}", handler(info), methods=["GET"]),
        Route("/v1/sessions/{id}/diff", handler(diff_list), methods=["GET"]),
        Route("/v1/sessions/{id}/diff/file", handler(diff_file), methods=["GET"]),
        Route("/v1/models", handler(models), methods=["GET"]),
        Route("/v1/model-length", handler(model_length), methods=["GET"]),
        Route("/v1/files/suggest", handler(suggest), methods=["GET"]),
        Route("/v1/asks/{ask_id}", handler(answer), methods=["POST"]),
        Route("/v1/stop", handler(stop), methods=["POST"]),
    ]
    return ProtocolHeader(Starlette(routes=routes))
