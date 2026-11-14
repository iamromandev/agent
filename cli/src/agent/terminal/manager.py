"""Terminals the agent runs: each a process on a pty, with its recent output, for the whole run."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from src.agent.env import child_environment

from .buffer import LineBuffer
from .pty import PtyProcess

NAME = re.compile(r"[A-Za-z0-9_-]{1,32}")
GRACE = 3.0  # seconds between SIGTERM and SIGKILL
type Status = tuple[Literal["running"], None] | tuple[Literal["exited"], int]  # int: exit code, <0 = signal
type Event = Literal["started", "exited", "closed"]

MAX_TERMINALS = 16  # started and kept at once, exited ones included
_READ = 65_536
_POLL = 0.02  # how often a stop checks the leader and the group
_POLL_MAX = 1.0  # the reaper of a leader that closed its stdio but runs on backs off from _POLL to this
_KILL_WAIT = 2.0  # how long a SIGKILLed leader or group gets to be reaped or to disappear


class TerminalError(Exception):
    """A terminal could not be started (invalid or used name, no workspace, no pty)."""


class Terminal:
    """One named process on a pty. Created by `TerminalManager.start`; everything runs on the event loop."""

    def __init__(
        self,
        name: str,
        command: str | None,
        process: PtyProcess,
        buffer: LineBuffer,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.name = name
        self.command = command  # None: the user's shell
        self.process = process
        self.buffer = buffer
        self._loop = loop
        self._subscribers: list[Callable[[bytes], None]] = []
        self._exit_listeners: list[Callable[[Terminal], None]] = []
        self._code: int | None = None  # set once the pty is drained and the leader reaped
        self._done = asyncio.Event()
        self._reaper: asyncio.Task[None] | None = None
        self._stops = 0  # stops in progress: each signals the group and kills what stays after the grace
        loop.add_reader(process.fd, self._on_readable)
        # Watch the leader from the start: on Linux a child that ignores SIGHUP keeps the pty open after the
        # leader exits, so EOF never comes and the exit would otherwise go unnoticed.
        self._ensure_reaper()

    @property
    def status(self) -> Status:
        return ("running", None) if self._code is None else ("exited", self._code)

    def subscribe(self, fn: Callable[[bytes], None]) -> Callable[[], None]:
        """Call `fn` with every chunk of raw output; the returned function unsubscribes."""
        self._subscribers.append(fn)

        def unsubscribe() -> None:
            if fn in self._subscribers:
                self._subscribers.remove(fn)

        return unsubscribe

    def on_exit(self, fn: Callable[[Terminal], None]) -> None:
        """Call `fn(terminal)` when the process has exited and its output is read (at once if it has)."""
        if self._code is None:
            self._exit_listeners.append(fn)
        else:
            self._loop.call_soon(self._call, fn, self)

    def write(self, data: bytes) -> None:
        """Send input to the process (dropped once it exited). Raises `pty.WriteOverflow` if it reads nothing."""
        if self._code is None:
            self.process.write(data)

    def resize(self, cols: int, rows: int) -> None:
        if self._code is None:
            self.process.resize(cols, rows)

    async def stop(self, grace: float = GRACE) -> Status:
        """SIGTERM the process group (also SIGHUP for the user's shell), SIGKILL it after `grace` seconds, and reap.

        Whatever else of the terminal is still alive when the leader is gone (a child that ignores SIGTERM and
        SIGHUP, a job with a group of its own, a `nohup` background job of the shell) is SIGKILLed when the
        grace is over; `Terminal.survivors` says whether anything outlived even that. A leader that SIGKILL does
        not take down in time is reported as still running; when it is reaped later, what is left of its
        group is killed then.
        """
        self._stops += 1
        try:
            deadline = self._loop.time() + grace
            if self._code is None:
                self.process.signal_tree(signal.SIGTERM)
                if self.command is None:  # an interactive shell ignores SIGTERM; its terminal closing sends SIGHUP
                    self.process.signal_tree(signal.SIGHUP)
                if not await self._exited_within(grace):
                    self.process.signal_tree(signal.SIGKILL)
                    if not await self._exited_within(_KILL_WAIT):
                        self._ensure_reaper()  # stuck in the kernel: reap it whenever it goes
                        return self.status
            await self._clear_group(deadline)
            if self._reaper is not None and not self._reaper.done():
                self._reaper.cancel()
            return self.status
        finally:
            self._stops -= 1

    def survivors(self) -> bool:
        """True while any process of the terminal (the leader's group or session) is still alive."""
        return self.process.group_alive()

    async def _exited_within(self, seconds: float) -> bool:
        """Wait until the leader is reaped and the terminal finished; False if it is not within `seconds`."""
        end = self._loop.time() + seconds
        while self._code is None:
            code = self.process.poll()
            if code is not None:
                self._finish(code)
                break
            left = end - self._loop.time()
            if left <= 0:
                return False
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._done.wait(), min(_POLL, left))
        return True

    async def _clear_group(self, deadline: float) -> None:
        """The leader is reaped: give the rest of its group until `deadline` to go, then SIGKILL it."""
        while self.process.group_alive() and self._loop.time() < deadline:
            await asyncio.sleep(_POLL)
        if self.process.group_alive():
            self.process.signal_tree(signal.SIGKILL)
            end = self._loop.time() + _KILL_WAIT
            while self.process.group_alive() and self._loop.time() < end:
                await asyncio.sleep(_POLL)

    def kill(self) -> None:
        """SIGKILL the group and reap the leader, with nothing awaited and no event loop needed.

        The last resort for when the loop that runs the terminals is gone. The leader is reaped within a second
        (SIGKILL is not delivered to a process that cannot be killed at all, which is left to the system).
        """
        if self.process.group_alive():
            self.process.signal_tree(signal.SIGKILL)
        deadline = time.monotonic() + _KILL_WAIT / 2
        while self.process.poll() is None and time.monotonic() < deadline:
            time.sleep(_POLL / 4)
        with contextlib.suppress(Exception):  # the loop may be closed
            self._loop.remove_reader(self.process.fd)
        self.process.close()

    def _on_readable(self) -> None:
        try:
            data = os.read(self.process.fd, _READ)
        except BlockingIOError:
            return
        except OSError:  # EIO: every other end of the pty is closed
            data = b""
        if data:
            self._output(data)
        else:
            self._loop.remove_reader(self.process.fd)
            self._ensure_reaper()

    def _output(self, data: bytes) -> None:
        self.buffer.feed(data)
        for fn in list(self._subscribers):
            self._call(fn, data)

    def _ensure_reaper(self) -> None:
        if self._reaper is None or self._reaper.done():
            self._reaper = self._loop.create_task(self._reap())

    async def _reap(self) -> None:
        """The pty is closed: wait for the leader to exit and be reaped, then finish."""
        delay = _POLL
        while (code := self.process.poll()) is None:  # a leader that closed its stdio may run for a long time
            await asyncio.sleep(delay)
            delay = min(delay * 2, _POLL_MAX)
        self._finish(code)

    def _finish(self, code: int) -> None:
        if self._code is not None:
            return
        self._loop.remove_reader(self.process.fd)
        while True:  # what is still queued in the pty
            try:
                data = os.read(self.process.fd, _READ)
            except OSError:  # nothing more now, or EIO
                break
            if not data:
                break
            self._output(data)
        if not self._stops and self.process.group_alive():
            self.process.signal_tree(signal.SIGKILL)  # the leader is gone on its own: no member outlives it
        self.process.close()
        self._code = code
        self._done.set()
        listeners, self._exit_listeners = self._exit_listeners, []
        for fn in listeners:
            self._call(fn, self)

    def _call(self, fn: Callable[..., object], *args: object) -> None:
        """Run a callback; one that fails is reported to the loop and never stops the reader."""
        try:
            fn(*args)
        except Exception as exc:
            self._loop.call_exception_handler({"message": f"terminal {self.name}: callback failed", "exception": exc})


def _shell() -> str:
    shell = os.environ.get("SHELL")
    return (shell and shutil.which(shell)) or "/bin/sh"


class TerminalManager:
    """The terminals of one run, by name. They belong to the run, not to a conversation."""

    def __init__(
        self,
        workspace: Path,
        *,
        cols: int = 80,
        rows: int = 24,
        grace: float = GRACE,
        scrollback: int = 5_000,
    ) -> None:
        self._workspace = workspace
        self._cols = cols
        self._rows = rows
        self._grace = grace
        self._scrollback = scrollback
        self._terminals: dict[str, Terminal] = {}
        self._listeners: list[Callable[[str, Event], None]] = []
        self._closed = False

    def start(self, name: str, command: str | None) -> Terminal:
        """Run `command` through the shell (None: an interactive shell) in the workspace, as terminal `name`.

        Must be called on the running event loop. An exited terminal of the same name is replaced; when
        `MAX_TERMINALS` are kept, the oldest exited one is dropped to make room. Listeners hear "closed" for
        a terminal that is replaced or dropped after the new one is registered, then "started".
        """
        self.check_name(name)
        old = self._terminals.get(name)
        if not self._workspace.is_dir():
            raise TerminalError(f"workspace {self._workspace} is not a folder")
        loop = asyncio.get_running_loop()  # before the spawn: without a loop no child or pty may be left behind
        shell = _shell()
        argv = [shell, "-i"] if command is None else [shell, "-c", command]
        env = {k: v for k, v in child_environment().items() if k not in ("COLUMNS", "LINES")}
        env["TERM"] = "xterm-256color"
        try:
            process = PtyProcess.spawn(argv, self._workspace, env, self._cols, self._rows)
        except (OSError, ValueError) as exc:
            raise TerminalError(f"could not start terminal: {exc}") from exc
        terminal = Terminal(name, command, process, LineBuffer(self._scrollback), loop)
        terminal.on_exit(lambda _: self._emit(name, "exited"))
        dropped = None
        if old is None and len(self._terminals) >= MAX_TERMINALS:
            dropped = self._oldest_exited()
        if dropped is not None:
            del self._terminals[dropped]
        self._terminals[name] = terminal
        if old is not None:
            self._emit(name, "closed")
        elif dropped is not None:
            self._emit(dropped, "closed")
        self._emit(name, "started")
        return terminal

    def _oldest_exited(self) -> str | None:
        return next((name for name, t in self._terminals.items() if t.status[0] == "exited"), None)

    def check_name(self, name: str) -> None:
        """Raise TerminalError unless a terminal called `name` could be started now."""
        if self._closed:
            raise TerminalError("terminals are closed")
        if not NAME.fullmatch(name):
            raise TerminalError(f"invalid terminal name {name!r}: use 1-32 letters, digits, - or _")
        old = self._terminals.get(name)
        if old is not None and old.status[0] == "running":
            raise TerminalError(f"terminal {name} is already running")
        if old is None and len(self._terminals) >= MAX_TERMINALS and self._oldest_exited() is None:
            raise TerminalError(f"too many terminals: {MAX_TERMINALS} are running, stop one first")

    @property
    def grace(self) -> float:
        return self._grace

    def get(self, name: str) -> Terminal | None:
        return self._terminals.get(name)

    def names(self) -> list[str]:
        return list(self._terminals)

    def listen(self, fn: Callable[[str, Event], None]) -> None:
        """Call `fn(name, event)` when a terminal is started, has exited, or is closed and forgotten."""
        self._listeners.append(fn)

    async def close(self, name: str) -> None:
        """Stop terminal `name` if it runs, and forget it. It stays listed until it is stopped."""
        terminal = self._terminals.get(name)
        if terminal is not None:
            await terminal.stop(self._grace)
            self._forget(terminal)

    async def close_all(self) -> None:
        """Stop every terminal at once (SIGTERM to all, one grace period, SIGKILL to the rest); start no more.

        Also covers a terminal that a `close` is still stopping: it stays listed until it is stopped.
        """
        self._closed = True
        terminals = list(self._terminals.values())
        await asyncio.gather(*(t.stop(self._grace) for t in terminals))
        for terminal in terminals:
            self._forget(terminal)

    def kill_all(self) -> None:
        """SIGKILL every terminal's group and forget them, with no event loop needed; start no more.

        For when the app failed before `close_all` could run: it does not wait for a grace period.
        """
        self._closed = True
        for terminal in list(self._terminals.values()):
            terminal.kill()
        self._terminals.clear()

    def _forget(self, terminal: Terminal) -> None:
        """Drop `terminal` from the list and report it closed, unless that already happened."""
        if self._terminals.get(terminal.name) is terminal:
            del self._terminals[terminal.name]
            self._emit(terminal.name, "closed")

    def _emit(self, name: str, event: Event) -> None:
        for fn in list(self._listeners):
            try:
                fn(name, event)
            except Exception as exc:
                asyncio.get_running_loop().call_exception_handler(
                    {"message": f"terminal {name}: listener failed", "exception": exc}
                )
