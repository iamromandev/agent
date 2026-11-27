"""`RemotePort`: the app's port to a session that lives in the workspace's daemon (`agent --shared`). Calls go over
`core.client.daemon.DaemonClient`; the event stream is decoded back into the engine and host events the local app
draws, so the app handles both ports the same way. The state the app asks about synchronously (claims, queue, mode,
model, todos, pending asks) is kept here from the snapshot and the events."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from core.client.daemon import DaemonClient, DaemonError, DaemonUnavailable, ProtocolMismatch
from core.engine import Compacting, Notice, TodosChanged, TurnEnded, UsageUpdate
from core.engine.mentions_index import Suggestion
from core.engine.mode import Mode
from core.engine.provider.base import Image
from core.engine.provider.models import CatalogError, ModelInfo
from core.engine.session import LoadedSession, SessionInfo
from core.engine.session.ulid import new_ulid
from core.engine.todo import TodoItem
from core.engine.turn_diff import DiffItem, FileDiff, TurnDiff
from core.engine.undo import UndoTurn
from core.engine.usage import UsageSummary
from core.host.asks import Answer, AnswerResult, AskEvent, AskResolved, answer_to_wire
from core.host.events import ModeChanged, ModelChanged, QueueChanged, TurnOutcome, UserEntry
from core.host.queue import Queued
from core.host.replay import snapshot_items
from core.host.session import Attachments, ModelSwitch, OpenedSession, SendResult, SessionSource, UndoResult
from core.host.wire import decode
from core.wire import models as wm
from core.wire.codec import WireError

LOST = "lost connection to the agent daemon; reconnecting…"
UNSAVED_LOST = "the daemon no longer has this session and it was never saved: it is lost; this is a new session"
RECONNECT_WAIT = 10.0  # seconds a lost daemon is waited for before the app asks what to do
RETRY_STEP = 0.25
SUGGEST_DEBOUNCE = 0.05
NOT_SHARED = "not available in --shared mode yet (use agent without --shared)"

OnEvent = Callable[[object], Awaitable[None]]
OnAsk = Callable[[AskEvent], None]


@dataclass(frozen=True)
class DaemonGone:
    """The daemon did not come back within `RECONNECT_WAIT`: the app asks whether to quit or keep waiting
    (`RemotePort.keep_waiting`). The input box is never touched."""

    reason: str


@dataclass(frozen=True)
class Reconnected:
    """The stream is back after a loss (`snapshot`: the session's state, when the gap was too old to replay)."""

    snapshot: wm.Snapshot | None


def _shaped[T](build: Callable[[], T]) -> T:
    """`build()`, with a wrong shape from the daemon (a missing key, a wrong type, a bad date) as `DaemonError`."""
    try:
        return build()
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        raise DaemonError(502, f"the daemon sent a bad answer ({type(exc).__name__}: {exc})") from exc


def _instance(hello: dict[str, Any]) -> object:
    return hello.get("instance") or hello.get("pid")


def _level(value: object) -> Literal["info", "warn", "error"]:
    return value if value in ("info", "warn", "error") else "info"  # ty: ignore[invalid-return-type]


def _session_info(data: dict[str, Any]) -> SessionInfo:
    return SessionInfo(
        id=str(data["id"]),
        path=Path(),  # the daemon's file: never opened by a remote client
        title=str(data.get("title", "")),
        created_at=datetime.fromisoformat(str(data["created_at"])),
        updated_at=datetime.fromisoformat(str(data["updated_at"])),
        message_count=int(data.get("message_count", 0)),
        open_elsewhere=bool(data.get("open_elsewhere", False)),
        live=bool(data.get("live", False)),
    )


def _diff_item(data: dict[str, Any]) -> DiffItem:
    return DiffItem(
        path=str(data["path"]),
        status=data["status"],
        added=data.get("added"),
        removed=data.get("removed"),
        note=data.get("note"),
        old_mode=int(data.get("old_mode") or 0),
        new_mode=int(data.get("new_mode") or 0),
        old_size=data.get("old_size"),
        new_size=data.get("new_size"),
    )


class RemotePort:
    """The port of a `--shared` client. Make it before the app; `connect` and `subscribe` run in the app's loop."""

    local = False

    def __init__(self, socket: Path, workspace: Path, *, client_id: str | None = None) -> None:
        self.socket = socket
        self.workspace = workspace
        self.client_id = client_id or new_ulid()
        self._client: DaemonClient | None = None
        self._session_id = ""
        self._model = ""
        self._window = 0
        self._mode = Mode.BUILD
        self.turn_active = False
        self.undoing = False
        self.workflow_active = False
        self._switching = False
        self._queue: tuple[Queued, ...] = ()
        self._todos: tuple[TodoItem, ...] = ()
        self._used = 0
        self._asks: dict[str, AskEvent] = {}
        self._last_seq = 0
        self._reader: asyncio.Task[None] | None = None
        self._on_event: OnEvent | None = None
        self._on_ask: OnAsk | None = None
        self._retry: asyncio.Event | None = None  # made in the loop by `subscribe`
        self._closed = False
        self._suggested: dict[str, list[Suggestion]] = {}
        self._suggest_task: asyncio.Task[None] | None = None
        self.on_suggestions: Callable[[], None] | None = None  # the app redraws the path list when a refresh lands
        self._tasks: set[asyncio.Task[Any]] = set()
        self._daemon_pid: object = None  # the daemon's instance id (its pid with an older daemon)
        self._unknown = False  # the stream said 404: the daemon no longer has the session
        self._outcomes = 0  # turn outcomes seen (a send's answer must not mark a turn busy that already ended)

    # state

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def model(self) -> str:
        return self._model

    @property
    def window(self) -> int:
        return self._window

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def switching(self) -> bool:
        return self._switching

    @property
    def queueing(self) -> bool:
        return self.turn_active and not self.undoing and not self._switching and not self.workflow_active

    @property
    def session_starting(self) -> bool:
        return False  # the daemon runs no SessionStart hooks for a shared session

    @property
    def queue_entries(self) -> tuple[Queued, ...]:
        return self._queue

    @property
    def todos(self) -> tuple[TodoItem, ...]:
        return self._todos

    def estimate(self) -> int:
        return self._used

    def busy_text(self, command: str) -> str:
        hint = "a session is loading" if self._switching and not self.turn_active else f"press Esc to interrupt, then {command}"
        return f"busy — {hint}"

    def refusal(self, command: str) -> str | None:
        """What this client knows is in the way; the daemon checks again (and says why it refuses)."""
        return self.busy_text(command) if self.turn_active or self._switching else None

    def message_refusal(self) -> str | None:
        if not (self.turn_active or self._switching):
            return None
        return f"busy — {'press Esc to interrupt' if self.turn_active else 'a session is loading'}"

    def begin_switch(self) -> bool:
        if self.turn_active or self._switching:
            return False
        self._switching = True
        return True

    def end_switch(self) -> None:
        self._switching = False

    def begin_close(self) -> None:
        self._closed = True

    def instructions_notice(self) -> None:
        return None  # the daemon's agent announces AGENTS.md in its own notices

    def forget_instructions_notice(self) -> None:
        return None

    def is_current_diff(self, diff: TurnDiff) -> bool:
        return True  # the daemon builds the list for the newest turn when asked

    def pending_asks(self) -> list[AskEvent]:
        return list(self._asks.values())

    def outcome_applied(self, ack: int) -> bool:
        return False  # the daemon takes the outcome as applied once its line was written to this client

    # connection

    def _http(self) -> DaemonClient:
        if self._client is None:
            self._client = DaemonClient(self.socket, self.client_id)
        return self._client

    async def connect(self) -> dict[str, Any]:
        """`/v1/hello`: `ProtocolMismatch` for another protocol, `DaemonUnavailable` when none answers. The daemon's pid
        is remembered: after a reconnect to another one, the stream starts over from a snapshot."""
        hello = await self._http().hello()
        self._daemon_pid = _instance(hello)
        return hello

    async def open(self, resume: str | None, model: str | None = None) -> wm.Snapshot:
        """Open (or attach to) a session and make it this port's: `DaemonError` when the daemon refuses (404 unknown,
        409 locked by another process). The stream (once subscribed) continues from the snapshot."""
        data = await self._http().open_session(resume, model)
        snapshot = self._snapshot_of(data.get("snapshot"))
        self._session_id = _shaped(lambda: str(data["session_id"]))
        self._apply_snapshot(snapshot)
        return snapshot

    def _snapshot_of(self, data: object) -> wm.Snapshot:
        if not isinstance(data, dict):
            raise DaemonError(500, "the daemon sent no snapshot")
        try:
            message = decode(data)
        except WireError as exc:
            raise DaemonError(500, f"the daemon sent a bad snapshot: {exc}") from exc
        if not isinstance(message, wm.Snapshot):
            raise DaemonError(500, "the daemon sent no snapshot")
        return message

    def _apply_snapshot(self, snapshot: wm.Snapshot) -> None:
        self._model = snapshot.model
        if snapshot.window:
            self._window = snapshot.window
        with contextlib.suppress(ValueError):
            self._mode = Mode(snapshot.mode)
        claims = snapshot.claims
        self.turn_active = claims.turn_active
        self.undoing = claims.undoing
        self.workflow_active = claims.workflow_active
        self._queue = tuple(Queued(q.typed, q.typed, q.label or None, q.client) for q in snapshot.queue)
        self._todos = tuple(TodoItem(t.content, t.status) for t in snapshot.todos)
        self._asks = {}
        for ask in snapshot.asks:
            with contextlib.suppress(WireError):
                event = decode(ask.model_dump(mode="json"))
                if isinstance(event, AskEvent):
                    self._asks[event.ask_id] = event
        self._last_seq = snapshot.seq

    def snapshot_session(self, snapshot: wm.Snapshot, title: str = "", count: int = 0) -> LoadedSession:
        """The snapshot as the local app's loaded session: what `_show_session` redraws."""
        now = datetime.now().astimezone()
        info = SessionInfo(self._session_id, Path(), title or "(shared session)", now, now, count)
        return LoadedSession(info, [], {}, snapshot_items(snapshot.records), 0, self._todos)

    def subscribe(self, on_event: OnEvent, on_ask: OnAsk, attachments: Attachments | None = None) -> None:
        """Start reading the session's stream into the app's handlers (needs the app's running loop). Pasted images
        never go to the daemon (`attachments` is ignored)."""
        self._on_event = on_event
        self._on_ask = on_ask
        self._retry = asyncio.Event()
        self._restart_reader()

    def _restart_reader(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
        self._reader = asyncio.get_running_loop().create_task(self._read())

    def keep_waiting(self) -> None:
        """The user chose to keep waiting for a lost daemon: try again."""
        if self._retry is not None:
            self._retry.set()

    async def _emit(self, event: object) -> None:
        if self._on_event is not None:
            await self._on_event(event)

    async def _read(self) -> None:
        """Follow the stream; when it ends or drops, say so, wait for the daemon and go on. Never ends on its own but
        when the port closes: anything unexpected ends in `DaemonGone` with its reason, and Keep waiting tries again."""
        lost = False
        while not self._closed:
            try:
                async for data in self._http().events(self._session_id, self._last_seq):
                    if not data:  # the stream is open again (`DaemonClient.events` yields `{}` first)
                        if lost:
                            lost = False
                            await self._emit(Reconnected(None))
                        continue
                    await self._handle(data)
                if self._closed:
                    return
            except asyncio.CancelledError:
                raise
            except DaemonError as exc:
                if exc.status == 404:
                    self._unknown = True  # the daemon no longer has the session: it is opened again
                else:
                    await self._emit(Notice("error", f"agent daemon: {exc.text}"))
            except DaemonUnavailable:
                pass
            except Exception as exc:  # the reader must go on: a bad message is said once
                await self._emit(Notice("error", f"agent daemon stream failed: {exc}"))
            if self._closed:
                return
            if not lost:
                lost = True
                await self._emit(Notice("warn", LOST))
            try:
                back = await self._come_back()
                reason = "the agent daemon did not come back"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                back, reason = False, f"reconnecting to the agent daemon failed: {exc}"
            if back is None:  # redrawn from a snapshot: the app already says it is back
                lost = False
                continue
            if not back:
                await self._emit(DaemonGone(reason))
                assert self._retry is not None
                self._retry.clear()
                await self._retry.wait()

    async def _come_back(self) -> bool | None:
        """Wait for the daemon (up to `RECONNECT_WAIT`). True: reopen the stream where it stopped; None: the session was
        opened again and the app redrew it from the snapshot (a new daemon resumes it from its file; an unsaved one
        that is gone is replaced by a new session, with a notice); False: it did not come back in time."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + RECONNECT_WAIT
        while loop.time() < deadline and not self._closed:
            try:
                hello = await self._http().hello()
            except (DaemonUnavailable, DaemonError, ProtocolMismatch):
                await asyncio.sleep(RETRY_STEP)
                continue
            if _instance(hello) != self._daemon_pid:  # another daemon: its numbering starts over
                self._daemon_pid = _instance(hello)
                self._last_seq = -1
                self._unknown = True
            if not self._unknown and self._last_seq >= 0:
                return True  # the same daemon: the stream replays what was missed
            try:
                try:
                    snapshot = await self.open(self._session_id)
                except DaemonError as exc:
                    if exc.status != 404:
                        raise
                    snapshot = await self.open(None)
                    await self._emit(Notice("warn", UNSAVED_LOST))
            except (DaemonUnavailable, DaemonError, ProtocolMismatch):
                await asyncio.sleep(RETRY_STEP)
                continue
            self._unknown = False
            await self._emit(Reconnected(snapshot))
            return None
        return False

    async def _handle(self, data: dict[str, Any]) -> None:
        seq = data.get("seq")
        try:
            message = decode(data)
        except WireError:
            return  # a message this client does not know: skipped, like the daemon skips local-only events
        if isinstance(message, wm.Snapshot):
            self._apply_snapshot(message)
            await self._emit(Reconnected(message))
            return
        if isinstance(seq, int) and seq > self._last_seq:
            self._last_seq = seq
        if isinstance(message, wm.Overflow):
            self._last_seq = -1  # too far behind: the next stream starts with a snapshot
            raise DaemonUnavailable("this client fell behind")
        if isinstance(message, wm.SessionClosed):
            raise DaemonUnavailable(message.reason or "the session was closed")
        if isinstance(message, wm.Message):
            return  # nothing else of the protocol's own is drawn
        self._track(message)
        if isinstance(message, AskEvent):
            if self._on_ask is not None:
                self._on_ask(message)
            return
        await self._emit(message)

    def _track(self, event: object) -> None:
        match event:
            case UserEntry():
                self.turn_active = True
            case Compacting(active) if active:
                self.turn_active = True
            case TurnOutcome() as outcome if not outcome.continued:
                self._outcomes += 1
                self.turn_active = False
                self.workflow_active = False
            case QueueChanged(entries):
                self._queue = entries
            case ModeChanged(mode):
                self._mode = mode
            case ModelChanged(model, window):
                self._model, self._window = model, window
            case TodosChanged(items):
                self._todos = tuple(items)
            case UsageUpdate(used, window, _, _, _):
                self._used, self._window = used, window
            case AskEvent() as ask:
                self._asks[ask.ask_id] = ask
            case AskResolved(ask_id, _, _):
                self._asks.pop(ask_id, None)
            case TurnEnded():
                pass

    # actions

    def _spawn(self, work: Awaitable[Any]) -> None:
        task = asyncio.ensure_future(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _quiet(self, work: Awaitable[Any], what: str) -> None:
        try:
            await work
        except (DaemonError, DaemonUnavailable) as exc:
            await self._emit(Notice("warn", f"could not {what}: {exc}"))

    async def send_message(
        self,
        text: str,
        *,
        label: str | None = None,
        init: bool = False,
        draft: str | None = None,
        pasted: Sequence[Image] | None = None,
    ) -> SendResult:
        """Send `text` (for `/init`, the daemon builds the prompt; `draft` is what goes back). Pasted images are never
        sent in shared mode."""
        seen = self._outcomes
        try:
            data = await self._http().send(self._session_id, draft if init and draft else text, label=label, init=init)
        except (DaemonError, DaemonUnavailable) as exc:
            return SendResult("refused", f"the agent daemon did not take it: {exc}")
        status = data.get("status")
        if status == "started" and self._outcomes == seen:  # its outcome may have come first on the stream
            self.turn_active = True
        if status in ("started", "queued", "ignored", "refused"):
            return SendResult(status, data.get("reason") or "")
        return SendResult("refused", "the agent daemon gave an unknown answer")

    async def start_compact(self) -> SendResult:
        try:
            data = await self._http().compact(self._session_id)
        except (DaemonError, DaemonUnavailable) as exc:
            return SendResult("refused", f"the agent daemon did not take it: {exc}")
        if data.get("status") == "started":
            self.turn_active = True
        return SendResult(data.get("status", "refused"), data.get("reason") or "")

    def interrupt(self) -> None:
        if self._session_id:
            self._spawn(self._quiet(self._http().interrupt(self._session_id), "interrupt the turn"))

    def set_mode(self, mode: Mode) -> None:
        self._mode = mode
        self._spawn(self._quiet(self._http().set_mode(self._session_id, mode.value), "change the mode"))

    def answer(self, ask_id: str, answer: Answer) -> AnswerResult:
        """Sent in the background; a refused answer is said as a notice (the first answer of any client wins)."""
        self._asks.pop(ask_id, None)

        async def post() -> None:
            try:
                result = await self._http().answer(ask_id, answer_to_wire(answer).model_dump(mode="json"))
            except (DaemonError, DaemonUnavailable) as exc:
                await self._emit(Notice("warn", f"could not answer: {exc}"))
                return
            if result not in ("ok", "already_answered"):
                await self._emit(Notice("warn", f"could not answer the approval ({result})"))

        self._spawn(post())
        return "ok"

    async def undo(self) -> UndoResult:
        self.undoing = self.turn_active = True
        try:
            data = await self._http().undo(self._session_id)
        except (DaemonError, DaemonUnavailable) as exc:
            return UndoResult(None, [Notice("error", f"Undo failed: {exc}")])
        finally:
            self.undoing = self.turn_active = False
        events = []
        given = data.get("events")
        for item in given if isinstance(given, list) else ():
            with contextlib.suppress(WireError):
                event = decode(item)
                if isinstance(event, Notice):
                    events.append(event)
        prompt = data.get("prompt")
        return UndoResult(prompt if isinstance(prompt, str) else None, list(events))

    async def new_session(self, command: str = "/new") -> str | None:
        """`/new`: this client moves to a fresh session of the daemon (the others stay where they are)."""
        if self.turn_active or self._switching:
            return self.busy_text(command)
        try:
            await self.open(None)
        except (DaemonError, DaemonUnavailable) as exc:
            return f"could not start a new session: {exc}"
        self._switching = True  # released by `end_switch`, as for the local port
        self._restart_reader()  # from 0: the new session's first notices are in its stream
        return None

    async def switch_to(self, session_id: str) -> wm.Snapshot:
        """Make the saved or live session `session_id` this port's and follow its stream; `DaemonError` or
        `DaemonUnavailable` when it cannot be opened (the current one stays)."""
        snapshot = await self.open(session_id)
        self._restart_reader()
        return snapshot

    async def open_session(self, info: SessionInfo) -> OpenedSession:
        try:
            snapshot = await self.switch_to(info.id)
        except DaemonError as exc:
            return OpenedSession(None, [("error", exc.text)])
        except DaemonUnavailable as exc:
            return OpenedSession(None, [("error", f"the agent daemon is not reachable: {exc}")])
        return OpenedSession(self.snapshot_session(snapshot, info.title, info.message_count), [])

    async def session_started(self, source: SessionSource) -> list[Any]:
        return []

    async def check(self) -> bool:
        return True  # the daemon asks the provider before every turn

    async def switch_model(self, name: str, looked: tuple[int | None, str | None] | None = None) -> ModelSwitch:
        try:
            data = await self._http().switch_model(self._session_id, name)
        except (DaemonError, DaemonUnavailable) as exc:
            return ModelSwitch([("error", f"Model switch failed: {exc}")])
        try:
            notices = _shaped(lambda: [(_level(n.get("level")), str(n.get("text", ""))) for n in data.get("notices") or ()])
        except DaemonError as exc:
            return ModelSwitch([("error", f"Model switch failed: {exc}")])
        model, window = data.get("model"), data.get("window")
        if isinstance(model, str) and isinstance(window, int):
            self._model, self._window = model, window
            return ModelSwitch(notices, model, window)
        return ModelSwitch(notices)

    async def list_models(self) -> list[ModelInfo]:
        try:
            data = await self._http().models()
        except (DaemonError, DaemonUnavailable) as exc:
            raise CatalogError(str(exc)) from exc
        try:
            return _shaped(
                lambda: [ModelInfo(str(d["name"]), d.get("size"), d.get("parameters"), d.get("context_length")) for d in data]
            )
        except DaemonError as exc:
            raise CatalogError(str(exc)) from exc

    async def model_length(self, name: str) -> int | None:
        return None  # the daemon reads the length itself when it switches

    async def list_sessions(self) -> list[SessionInfo]:
        """The folder's sessions as the daemon sees them; `DaemonError` or `DaemonUnavailable` (the app says why)."""
        data = await self._http().list_sessions()
        return _shaped(lambda: [_session_info(d) for d in data])

    async def diff_list(self) -> TurnDiff | None:
        data = (await self._http().diff(self._session_id)).get("diff")
        if data is None:
            return None

        def build() -> TurnDiff:
            items = tuple(_diff_item(i) for i in data.get("items") or ())
            return TurnDiff(items, int(data.get("more") or 0), bool(data.get("fallback")), None, self.workspace, UndoTurn(0, ""))

        return _shaped(build)

    async def diff_file(self, diff: TurnDiff, item: DiffItem) -> FileDiff:
        index = next((n for n, i in enumerate(diff.items) if i is item), -1)  # the daemon finds the file by its place
        data = await self._http().diff_file(self._session_id, index)
        return _shaped(
            lambda: FileDiff(
                _diff_item(data["item"]), tuple(data.get("meta") or ()), tuple(data.get("lines") or ()), int(data.get("cut") or 0)
            )
        )

    async def usage(self) -> UsageSummary:
        answer = await self._http().info(self._session_id, "usage")

        def build() -> UsageSummary:
            data = answer["usage"]
            return UsageSummary(
                tuple(tuple(row) for row in data.get("lines") or ()),
                tuple(tuple(row) for row in data.get("models") or ()),
                data.get("note"),
                bool(data.get("empty")),
            )

        return _shaped(build)

    async def info(self, name: str) -> list[tuple[Literal["info", "warn", "error"], str]]:
        """The lines of a listing command from the daemon (its runtime's rules, hooks, servers, ...)."""
        try:
            data = await self._http().info(self._session_id, name)
        except DaemonError as exc:
            return [("warn", exc.text)]
        except DaemonUnavailable as exc:
            return [("warn", f"the agent daemon is not reachable: {exc}")]
        try:
            return _shaped(lambda: [(_level(line.get("level")), str(line.get("text", ""))) for line in data.get("lines") or ()])
        except DaemonError as exc:
            return [("warn", exc.text)]

    def suggest_paths(self, token: str) -> list[Suggestion]:
        """The last answer for this token (or the longest shorter one it starts with), refreshed in the background."""
        cached = self._suggested.get(token)
        if cached is None:
            prefix = max((t for t in self._suggested if token.startswith(t)), key=len, default=None)
            cached = self._suggested.get(prefix, []) if prefix is not None else []
        if self._suggest_task is not None:
            self._suggest_task.cancel()
        with contextlib.suppress(RuntimeError):  # no running loop (a test): nothing to refresh
            self._suggest_task = asyncio.get_running_loop().create_task(self._refresh_paths(token))
        return cached

    async def _refresh_paths(self, token: str) -> None:
        await asyncio.sleep(SUGGEST_DEBOUNCE)
        try:
            data = await self._http().suggest(token)
        except (DaemonError, DaemonUnavailable):
            return
        try:
            found = _shaped(lambda: [Suggestion(str(d["path"]), bool(d["folder"])) for d in data])
        except DaemonError:
            return
        if self._suggested.get(token) == found:
            return
        if len(self._suggested) > 200:
            self._suggested.clear()
        self._suggested[token] = found
        if self.on_suggestions is not None:
            self.on_suggestions()

    async def close(self) -> None:
        self._closed = True
        for task in (self._reader, self._suggest_task, *self._tasks):
            if task is not None:
                task.cancel()
        for task in (self._reader, *list(self._tasks)):
            if task is not None:
                with contextlib.suppress(BaseException):
                    await task
        if self._client is not None:
            await self._client.aclose()
