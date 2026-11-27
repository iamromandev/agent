"""The daemon of one workspace: one runtime, the live sessions (`SessionHost`s) its clients share, the launch trust
prompts as daemon-wide asks, config-change and MCP notices for every session, and the idle exit."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from core.engine.events import Notice
from core.engine.loop import close_recorder
from core.engine.mentions_index import FileIndex
from core.engine.session import FileRecorder, SessionError, SessionInfo, new_ulid, sessions_root, workspace_dir
from core.engine.session.lock import SessionLock
from core.host.asks import Answer, AnswerResult, AskBroker, AskEvent
from core.host.bus import Subscriber
from core.host.session import SessionHost
from core.trust import TrustOutcome, apply_trust, trust_lines

if TYPE_CHECKING:
    from core.runtime import Runtime

log = logging.getLogger("agent.daemon")

KEPT_NOTICES = 50  # daemon-wide notices a session opened later still gets (launch warnings, trust results)
WATCH_STEP = 0.25  # seconds between two looks of the idle watchdog (at most)


class SessionLockedElsewhere(Exception):
    """The session is open in another agent process."""


class UnknownSession(Exception):
    """No saved session matches."""


class _Fanout:
    """A publisher for every live session's bus (the daemon's own asks and notices)."""

    def __init__(self, daemon: Daemon) -> None:
        self._daemon = daemon

    def publish(self, event: object) -> None:
        for host in list(self._daemon.hosts.values()):
            host.bus.publish(event)


class Daemon:
    def __init__(self, rt: Runtime, *, idle: float | None = None) -> None:
        self.rt = rt
        self.workspace = rt.root
        self.idle = idle if idle is not None else rt.settings.daemon_idle
        self.hosts: dict[str, SessionHost] = {}  # by session id (chosen up front for a new session)
        self.asks = AskBroker(_Fanout(self))  # the launch trust prompts, shown in every session
        self.file_index = FileIndex(rt.root, rt.guard)
        self.connections: dict[tuple[str, str], int] = {}  # (session id, client) -> open event streams
        self._opening: dict[str, asyncio.Lock] = {}  # by session id: resumes of one session run one at a time
        self.stopping = asyncio.Event()
        self._notices: deque[Notice] = deque(maxlen=KEPT_NOTICES)
        self._stop_hooks: list[Callable[[], None]] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self.server: Any = None  # the uvicorn server, once `run_daemon` made it
        self._sessions = workspace_dir(sessions_root(rt.settings.sessions_dir), rt.root)

    # text

    def redact(self, text: str) -> str:
        """What every text on the wire goes through (`Redactor.redact` returns the text and a count)."""
        return self.rt.guard.redactor.redact(text)[0]

    # life cycle

    def on_stop(self, hook: Callable[[], None]) -> None:
        self._stop_hooks.append(hook)

    def request_stop(self) -> None:
        """`agent stop`, the idle exit or a signal: the event streams end and the server stops accepting."""
        if self.stopping.is_set():
            return
        self.stopping.set()
        for hook in self._stop_hooks:
            with contextlib.suppress(Exception):
                hook()

    async def start(self) -> None:
        """Begin: notices from the config watch and MCP, the servers, the file index, the trust prompts, the watchdog."""
        loop = self._loop = asyncio.get_running_loop()
        rt = self.rt
        for text in rt.warnings:
            self._notice("warn", text)
        for text in rt.model_notices:
            self._notice("info", text)
        rt.config_guard.attach_loop(loop)
        rt.config_guard.listen(self._config_changed)
        rt.mcp.listen(self._mcp_event)
        rt.mcp.start()  # never the untrusted ones
        self._tasks.append(loop.create_task(asyncio.to_thread(self.file_index.build)))
        self._tasks.append(loop.create_task(self._trust()))
        self._tasks.append(loop.create_task(self._watch_idle()))

    async def close(self) -> None:
        """Stop everything, in order: the turns, the asks, the sessions (recorders flushed, locks released), the
        runtime. Never raises."""
        if self._closed:
            return
        self._closed = True
        self.request_stop()
        with contextlib.suppress(Exception):
            self.rt.config_guard.unlisten(self._config_changed)
        for host in list(self.hosts.values()):
            with contextlib.suppress(Exception):
                host.interrupt()
        self.asks.cancel_all("closing")
        for host in list(self.hosts.values()):
            try:
                await host.aclose()
            except Exception:
                log.exception("closing a session failed")
            host.agent.close()  # flushes the recorder and releases the session lock
        self.file_index.cancel()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(BaseException):
                await task
        try:
            await self.rt.close()
        except Exception:
            log.exception("closing the runtime failed")
        finally:
            self.rt.kill_all()

    async def _watch_idle(self) -> None:
        """Exit after `idle` seconds with no event stream open and no turn or hook running."""
        loop = asyncio.get_running_loop()
        quiet_since: float | None = None
        step = min(WATCH_STEP, self.idle / 4)
        while not self.stopping.is_set():
            busy = any(self.connections.values()) or any(h.turn_active or h.busy_tasks() for h in self.hosts.values())
            now = loop.time()
            if busy:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = now
            elif now - quiet_since >= self.idle:
                log.info("no client for %g s: stopping", self.idle)
                self.request_stop()
                return
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.stopping.wait(), step)

    # notices to every session

    def _notice(self, level: str, text: str) -> None:
        notice = Notice(level, text)  # ty: ignore[invalid-argument-type]
        self._notices.append(notice)
        for host in list(self.hosts.values()):
            host.bus.publish(notice)

    def _soon(self, level: str, text: str) -> None:
        """`_notice` from any thread."""
        loop = self._loop
        if loop is None or loop.is_closed():
            raise RuntimeError("the daemon is not running")
        loop.call_soon_threadsafe(self._notice, level, text)

    def _config_changed(self, text: str) -> None:
        """A config watch warning (already escaped), for every session (R14)."""
        self._soon("warn", text)

    def _mcp_event(self, event: tuple[str, str, str]) -> None:
        kind, server, text = event
        match kind:
            case "ready":
                level, line = "info", f"mcp {server} ready · {text}"
            case "failed":
                level, line = "warn", f"mcp {server} failed: {text}"
            case "exited":
                level, line = "warn", f"mcp {server} exited: {text}"
            case _:
                level, line = "warn", f"mcp {server}: {text}" if server else text
        with contextlib.suppress(RuntimeError):
            self._soon(level, line)

    # trust prompts (the launch's, asked once, in today's order: global, MCP, permissions)

    async def _trust(self) -> None:
        rt = self.rt
        rules, hooks, mcp, guard = rt.permissions.rules, rt.hooks, rt.mcp, rt.config_guard
        if guard.marker_set and not rules.global_active and not rules.global_revoked:
            servers = mcp.untrusted("global")
            for line in rt.held.notices():
                self._notice("warn", line)
            marker = await asyncio.to_thread(guard.marker_info)
            ask = trust_lines(
                "global_trust",
                rules=rules.global_allow,
                hooks=[entry for entry in hooks.entries if entry.source == "global"],
                servers=servers,
                marker=marker,
                deny=rules.global_deny,
                held=rt.held,
            )
            yes = await self.asks.ask("global_trust", ask)
            self._show(
                await asyncio.to_thread(
                    apply_trust,
                    "global_trust",
                    bool(yes),
                    guard=guard,
                    servers=[s.name for s in servers],
                    held_any=rt.held.any,
                )
            )
        servers = mcp.untrusted("project")
        if servers:
            yes = await self.asks.ask("mcp_trust", trust_lines("mcp_trust", servers=servers))
            self._show(
                await asyncio.to_thread(
                    apply_trust, "mcp_trust", bool(yes), mcp_trust=rt.setup.remember, servers=[s.name for s in servers]
                )
            )
        allow, project_hooks = rules.untrusted_allow, hooks.untrusted()
        if allow or project_hooks:
            yes = await self.asks.ask("permission_trust", trust_lines("permission_trust", rules=allow, hooks=project_hooks))
            self._show(
                await asyncio.to_thread(
                    apply_trust,
                    "permission_trust",
                    bool(yes),
                    permissions=rules,
                    has_rules=bool(allow),
                    has_hooks=bool(project_hooks),
                )
            )

    def _show(self, outcome: TrustOutcome) -> None:
        for level, text in outcome.notices:
            self._notice(level, text)
        if outcome.start:
            self.rt.mcp.start(outcome.start)

    # sessions

    def _new_host(self, session_id: str) -> SessionHost:
        rt = self.rt
        folder = self._sessions

        def recorder() -> FileRecorder:
            return FileRecorder(folder, rt.root, session_id=new_ulid())

        host = SessionHost(
            rt.agent_factory,
            rt.provider,
            rt.root,
            window_cap=rt.settings.num_ctx,
            store=rt.store,
            recorder_factory=recorder,
            catalog=rt.catalog,
            vision=rt.vision,
            guard=rt.guard,
            permissions=rt.permissions.rules,
            hooks=rt.hooks,
        )
        host.connected = lambda client: self.connections.get((host.session_id, client), 0) > 0
        old = host.agent.reset(recorder=FileRecorder(folder, rt.root, session_id=session_id))
        if old is not None:
            close_recorder(old)  # never written: the runtime's lazy recorder
        for notice in self._notices:  # what the daemon said before this session existed
            host.bus.publish(notice)
        return host

    async def open(self, resume: str | None) -> tuple[SessionHost, bool]:
        """The live session for a client, and whether it was made now: a new one (`resume` None), the latest
        (`"latest"`; a new one when there is none), or the saved one with that id or id prefix. A session already live
        in this daemon is shared. `UnknownSession`, or `SessionLockedElsewhere` when another process has it open."""
        info: SessionInfo | None = None
        if resume is not None:
            try:
                info = await asyncio.to_thread(self.rt.store.latest if resume == "latest" else lambda: self.rt.store.find(resume))
            except SessionError as exc:
                raise UnknownSession(str(exc)) from exc
        if info is None:
            session_id = new_ulid()
            host = self._new_host(session_id)
            self.hosts[session_id] = host
            return host, True
        # One opener per session at a time: a second one waits and then finds the first one's host (this process's
        # lock registry would let both resume, and the loser would be an orphan holding a count of the lock).
        opening = self._opening.setdefault(info.id, asyncio.Lock())
        async with opening:
            live = self.hosts.get(info.id)  # looked at again after every await
            if live is not None:
                return live, False
            if SessionLock.held_elsewhere(info.path.parent, info.id):
                raise SessionLockedElsewhere(f"session {info.id} is open in another agent process")
            host = self._new_host(new_ulid())
            opened = await host.open_session(info, "daemon")
            if opened.loaded is None:
                await host.aclose()
                host.agent.close()
                text = opened.notices[0][1] if opened.notices else "the session could not be opened"
                raise SessionLockedElsewhere(text)
            self.hosts[info.id] = host
            return host, True

    def host(self, session_id: str) -> SessionHost | None:
        return self.hosts.get(session_id)

    def attach(self, session_id: str, client: str) -> None:
        key = (session_id, client)
        self.connections[key] = self.connections.get(key, 0) + 1

    def detach(self, host: SessionHost, session_id: str, client: str, sub: Subscriber) -> None:
        """An event stream ended: when it was the client's last one on the session, what it held is released."""
        host.bus.unsubscribe(sub)
        key = (session_id, client)
        left = self.connections.get(key, 1) - 1
        if left > 0:
            self.connections[key] = left
            return
        self.connections.pop(key, None)
        host.client_gone(client)

    def pending_asks(self) -> list[AskEvent]:
        return self.asks.pending()

    def answer(self, ask_id: str, answer: Answer, client: str) -> AnswerResult:
        """The first valid answer wins, wherever the ask lives (the daemon's trust prompts, or a session's
        approvals)."""
        result = self.asks.answer(ask_id, answer, client)
        if result != "unknown":
            return result
        for host in list(self.hosts.values()):
            result = host.asks.answer(ask_id, answer, client)
            if result != "unknown":
                return result
        return "unknown"

    def live_ids(self) -> set[str]:
        return {host.session_id for host in self.hosts.values()} | set(self.hosts)
