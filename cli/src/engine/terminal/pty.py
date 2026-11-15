"""A process on a pseudo-terminal, in its own session and process group."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import fcntl
import os
import select
import signal
import struct
import subprocess
import termios
import threading
import warnings
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Self

_SPAWN_WAIT = 5.0  # how long a child gets to exec before the spawn is given up
MAX_PENDING = 1 << 20  # bytes of input queued for a process that does not read; more is refused
_live: weakref.WeakSet[PtyProcess] = weakref.WeakSet()

# `spawn` forks while asyncio's executor threads may exist, which Python warns about (on stderr, which would
# corrupt the screen of the TUI). The child only calls async-signal-safe-like functions (see `spawn`), so the
# warning is switched off once, here, for exactly that message: no per-call, thread-unsafe filter changes.
# This filter is intentionally process-wide: it applies to every fork in the program, and to nothing but that message.
warnings.filterwarnings("ignore", message=r"This process \(pid=\d+\) is multi-threaded", category=DeprecationWarning)


class WriteOverflow(Exception):
    """Input was refused because the process has not read the last megabyte sent to it."""


def _wait_readable(fd: int, timeout: float) -> bool:
    """Wait for `fd` to become readable (or hang up); poll, unlike select, takes descriptors above 1023."""
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    return bool(poller.poll(timeout * 1000))


def _all_pids() -> list[int]:
    """Every process id on the machine: from /proc where it exists, else from `ps`. Empty if neither works."""
    try:
        return [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        pass
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid="], capture_output=True, text=True, timeout=2, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(word) for word in out.split() if word.isdigit()]


def _session_groups(sid: int) -> set[int]:
    """The process groups of every process in session `sid`, whatever job control did to them."""
    groups: set[int] = set()
    for pid in _all_pids():
        try:
            if os.getsid(pid) == sid:
                groups.add(os.getpgid(pid))
        except (ProcessLookupError, PermissionError):  # gone meanwhile, or not ours (so not in our session)
            continue
    return groups


def _kill_and_reap(pid: int) -> None:
    """SIGKILL a child that was just forked (its group may not exist yet) and wait for it."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)
    with contextlib.suppress(ChildProcessError):
        os.waitpid(pid, 0)


def _exists(pgid: int) -> bool:
    """True when the process group has a member. macOS answers EPERM for a group that holds only zombies."""
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _set_size(fd: int, cols: int, rows: int) -> None:
    try:
        size = struct.pack("HHHH", rows, cols, 0, 0)
    except struct.error as exc:
        raise ValueError(f"invalid terminal size {cols}x{rows}") from exc
    fcntl.ioctl(fd, termios.TIOCSWINSZ, size)


class PtyProcess:
    """A child whose stdin, stdout and stderr are one pty, which it also has as its controlling terminal.

    `fd` is the master side: non-blocking and close-on-exec, so other children never inherit it.
    The child is a session leader, so `signal_group` reaches it and everything it started that stayed
    in its group.
    """

    pid: int
    fd: int

    def __init__(self, pid: int, fd: int) -> None:
        self.pid = pid
        self.fd = fd
        self._returncode: int | None = None
        self._closed = False
        self._gone = False  # the leader is reaped and no member is left: the group id may be reused, never signal it
        self._foreground: set[int] = set()  # groups of the pty's foreground jobs (job control gives them their own)
        self._pending = bytearray()  # input the process has not taken yet
        self._watching: asyncio.AbstractEventLoop | None = None  # the loop that flushes `_pending`
        self._lock = threading.Lock()  # one thread reaps; the loser must not see ECHILD as an error
        _live.add(self)

    @classmethod
    def spawn(
        cls, argv: Sequence[str], cwd: Path, env: Mapping[str, str], cols: int, rows: int
    ) -> Self:
        """Start `argv` in `cwd` with exactly `env`.

        Uses fork: asyncio's executor threads may exist, but the child only calls `login_tty`, `chdir` and
        `execvpe` (no Python locks are taken) and always ends in `_exit`, so a failure never runs the
        parent's code twice. A program that cannot start exits with 127; one that has not exec'd after
        `_SPAWN_WAIT` seconds is killed and TimeoutError (an OSError) is raised. ValueError is raised for a
        size a terminal cannot have. This blocks the caller, and so the event loop, until the child has exec'd:
        normally a few milliseconds, at most `_SPAWN_WAIT` seconds.
        """
        master, slave = os.openpty()  # all of these are non-inheritable: they close on exec
        ready_read = ready_write = -1
        pid = -1
        try:
            ready_read, ready_write = os.pipe()
            _set_size(slave, cols, rows)
            args, environment = list(argv), dict(env)
            pid = os.fork()
            if pid == 0:  # the child
                try:
                    os.login_tty(slave)  # new session, the pty as controlling terminal, and as 0, 1, 2
                    for name in ("SIGPIPE", "SIGXFSZ"):
                        signal.signal(getattr(signal, name), signal.SIG_DFL)  # Python ignores these
                    os.chdir(cwd)
                    os.execvpe(args[0], args, environment)
                except BaseException:
                    pass
                finally:
                    os._exit(127)
            os.close(ready_write)
            ready_write = -1
            # EOF arrives when the child has exec'd (the pipe closes with it) or died: from then on its
            # session and group exist, so a signal sent at once cannot miss it.
            if not _wait_readable(ready_read, _SPAWN_WAIT):
                raise TimeoutError(f"{args[0]} did not start within {_SPAWN_WAIT:g} seconds")
            os.read(ready_read, 1)
        except BaseException:  # also Ctrl-C: no child, and no pty end, is left behind
            os.close(master)
            if pid > 0:
                _kill_and_reap(pid)
            raise
        finally:
            os.close(slave)
            for fd in (ready_read, ready_write):
                if fd >= 0:
                    os.close(fd)
        os.set_blocking(master, False)
        return cls(pid, master)

    def resize(self, cols: int, rows: int) -> None:
        if self._closed:
            return
        with contextlib.suppress(OSError, ValueError):
            _set_size(self.fd, cols, rows)
        self.signal_group(signal.SIGWINCH)

    def write(self, data: bytes) -> None:
        """Send `data` to the process; what a full pty queue does not take now is sent as the process reads.

        Never blocks. Input for a process that has gone is dropped. At most `MAX_PENDING` bytes wait: past
        that `WriteOverflow` is raised before anything is sent, so `data` is dropped whole. Waiting needs the
        running event loop.
        """
        if self._closed or not data:
            return
        if len(self._pending) + len(data) > MAX_PENDING:
            raise WriteOverflow(f"input dropped: the process has not read the last {MAX_PENDING} bytes sent to it")
        if self._pending:  # keep the order
            self._queue(data)
            return
        view = memoryview(data)
        try:
            while view:
                view = view[os.write(self.fd, view) :]
        except BlockingIOError:
            pass
        except OSError:  # EIO after exit
            return
        if view:  # the pty queue is full
            self._queue(bytes(view))

    def _queue(self, data: bytes) -> None:
        self._pending += data
        if self._watching is None:
            loop = asyncio.get_running_loop()
            loop.add_writer(self.fd, self._flush)
            self._watching = loop

    def _flush(self) -> None:
        try:
            while self._pending:
                del self._pending[: os.write(self.fd, self._pending)]
        except BlockingIOError:
            return
        except OSError:  # EIO after exit
            self._pending.clear()
        self._unwatch()

    def _unwatch(self) -> None:
        loop, self._watching = self._watching, None
        if loop is not None:
            with contextlib.suppress(RuntimeError, ValueError, OSError):  # a loop already closed
                loop.remove_writer(self.fd)

    def signal_group(self, sig: int) -> None:
        """Send `sig` to the leader's process group; a group that is gone is not an error."""
        if self._gone:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.pid, sig)

    def signal_tree(self, sig: int) -> None:
        """`signal_group`, and every other group of the leader's session.

        Job control gives the jobs of an interactive shell (foreground or `&`) groups of their own, and
        `nohup` keeps them in the session when the shell quits, so the session is searched for them (a job that
        calls `setsid` is out of reach). The pty's foreground group is also remembered at each call (it cannot
        be asked once the leader has gone), and is signalled and watched by `group_alive` until it is empty.
        """
        if self._gone:
            return
        if not self._closed:
            with contextlib.suppress(OSError):
                group = os.tcgetpgrp(self.fd)
                if group > 1 and group not in (self.pid, os.getpgrp()):
                    self._foreground.add(group)
        self.signal_group(sig)
        for group in self._foreground | _session_groups(self.pid):
            if group > 1 and group not in (self.pid, os.getpgrp()):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(group, sig)

    def group_alive(self) -> bool:
        """True while a process of the leader's group, of a remembered foreground group, or of its session exists.

        Once the leader is reaped and nothing is left this stays False for good, and signals stop: the id
        (which is the group id and the session id) may be handed to an unrelated process.
        """
        if self._gone:
            return False
        self._foreground = {group for group in self._foreground if _exists(group)}
        if self._foreground or _exists(self.pid) or _session_groups(self.pid):
            return True
        if self._returncode is not None:
            self._gone = True
        return False

    def poll(self) -> int | None:
        """The exit code (negative: the signal that ended it) once the leader has exited, else None."""
        with self._lock:
            if self._returncode is None:
                try:
                    pid, status = os.waitpid(self.pid, os.WNOHANG)
                except ChildProcessError:
                    self._returncode = 255  # reaped elsewhere: the status is lost
                else:
                    if pid:
                        self._returncode = os.waitstatus_to_exitcode(status)
            return self._returncode

    def close(self) -> None:
        """Close the master side (once)."""
        if not self._closed:
            self._closed = True
            self._pending.clear()
            self._unwatch()
            with contextlib.suppress(OSError):
                os.close(self.fd)


def _kill_live_groups() -> None:
    """Last resort at interpreter exit: no group of a terminal outlives the agent, closed or not."""
    for process in list(_live):
        if process.group_alive():
            process.signal_tree(signal.SIGKILL)


atexit.register(_kill_live_groups)
