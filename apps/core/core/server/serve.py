"""Running the daemon: `serve` in the foreground (`agent serve`), `start_background` (for `--shared`), `stop`."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Generator
from pathlib import Path
from types import FrameType
from typing import Protocol

import httpx
import uvicorn

from core.server.app import build_app
from core.server.daemon import Daemon
from core.server.files import AlreadyRunning, PidLock, bind_socket, pid_held, read_pid
from core.server.paths import RunDirError, existing_socket_path, log_path, pid_path, socket_path
from core.server.protocol import PeerCheckedH11

log = logging.getLogger("agent.daemon")

START_WAIT = 10.0  # seconds `start_background` waits for the new daemon to answer
STOP_WAIT = 10.0  # seconds `stop` waits for the daemon's process to exit
POLL = 0.05
GRACEFUL = 5  # seconds uvicorn waits for open connections at shutdown (the streams end at once)
CLIENT_ID = "agent-cli"


class _Server(uvicorn.Server):
    """uvicorn's server, whose SIGINT/SIGTERM stops the daemon: the streams end, then `Daemon.close` runs. The signal is
    not raised again afterwards (uvicorn would, and the default action would kill the process before the cleanup)."""

    def __init__(self, config: uvicorn.Config, daemon: Daemon) -> None:
        super().__init__(config)
        self.daemon = daemon
        self.loop: asyncio.AbstractEventLoop | None = None

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        self.should_exit = True
        loop = self.loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self.daemon.request_stop)


async def run_daemon(daemon: Daemon, sock: object) -> None:
    """Serve `daemon` on the listening socket `sock` until it is asked to stop, then close it (awaited)."""
    config = uvicorn.Config(
        build_app(daemon),
        http=PeerCheckedH11,
        lifespan="off",
        log_level="warning",
        timeout_graceful_shutdown=GRACEFUL,
    )
    server = _Server(config, daemon)
    server.loop = asyncio.get_running_loop()
    daemon.server = server  # its `servers` say what it listens on (tests: only the Unix socket)
    daemon.on_stop(lambda: setattr(server, "should_exit", True))
    with _signals_ignored():  # uvicorn's handlers replace these while it serves, and put them back after
        try:
            await daemon.start()
            await server.serve(sockets=[sock])  # ty: ignore[invalid-argument-type]
        finally:
            await daemon.close()


@contextlib.contextmanager
def _signals_ignored() -> Generator[None]:
    """SIGINT and SIGTERM do nothing outside uvicorn's own handling: a second `agent stop` (its SIGTERM fallback) or
    ctrl+c during `Daemon.close` must not kill the process before the sessions are flushed, the child processes
    stopped and the socket and pid file removed. Restored on exit. Only the main thread can set handlers."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    before = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in before.items():
            signal.signal(sig, handler)


def serve(workspace: Path, model: str | None = None) -> int:
    """`agent serve`: the daemon for `workspace` in the foreground, logging to stderr. 1 when another daemon runs."""
    from core.engine.commands.builtins import BUILTINS
    from core.runtime import build_runtime

    ws = workspace.resolve()
    try:
        lock = PidLock.acquire(pid_path(ws))
    except AlreadyRunning as exc:
        print(f"agent daemon already running for {ws} (pid {exc.pid if exc.pid is not None else '?'})", file=sys.stderr)
        return 1
    except (RunDirError, OSError) as exc:
        print(f"agent daemon: {exc}", file=sys.stderr)
        return 1
    path = socket_path(ws)
    sock: socket.socket | None = None
    try:
        try:
            sock = bind_socket(path)
        except OSError as exc:
            print(f"agent daemon: {exc}", file=sys.stderr)
            return 1
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
        log.info("agent daemon for %s on %s (pid %d)", ws, path, os.getpid())
        rt = build_runtime(ws, BUILTINS, terminals=False, images=False, checkpoints=True, model=model)  # ctrl+c still aborts here
        with _signals_ignored():  # from here to the last file removed, a signal cannot skip the cleanup
            try:
                asyncio.run(run_daemon(Daemon(rt), sock))
            finally:
                try:
                    rt.kill_all()  # idempotent once `close` ran
                finally:
                    _remove(sock, path, lock)
                    sock = None
                    log_undelivered(rt.config_guard)  # what the quit checks saw after the streams were gone
    finally:
        if sock is not None:  # the runtime could not be built
            _remove(sock, path, lock)
        else:
            lock.release()  # idempotent
    return 0


class _Undelivered(Protocol):
    def undelivered(self) -> list[str]: ...


def log_undelivered(guard: _Undelivered) -> None:
    """The config watch warnings no client heard (said after the streams ended), into the daemon's log. Never raises."""
    try:
        texts = guard.undelivered()
    except Exception:
        log.exception("could not read the config watch's warnings")
        return
    for text in texts:
        log.warning("warning: %s", text)


def _remove(sock: socket.socket, path: Path, lock: PidLock) -> None:
    """Close the socket and remove it and the pid file (the lock goes with it). Never raises."""
    with contextlib.suppress(OSError):
        sock.close()
    with contextlib.suppress(OSError):
        path.unlink()
    lock.release()


def _client(path: Path, timeout: float) -> httpx.Client:
    return httpx.Client(
        transport=httpx.HTTPTransport(uds=str(path)),
        base_url="http://agent",
        timeout=timeout,
        headers={"X-Agent-Client": CLIENT_ID},
    )


def hello(workspace: Path, timeout: float = 1.0) -> dict[str, object] | None:
    """The running daemon's `/v1/hello`, or None when none answers (the run folder is never made here). `RunDirError`
    when the run folder is not ours alone."""
    path = existing_socket_path(workspace.resolve())
    if path is None or not path.exists():
        return None
    try:
        with _client(path, timeout) as client:
            response = client.get("/v1/hello")
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def daemon_argv(workspace: Path, model: str | None = None) -> list[str]:
    """The daemon's command line. `-P`: the interpreter does not put its working folder (the workspace, untrusted) first
    on `sys.path`, so a `core/`, `uvicorn.py` or `argparse.py` there is never imported. (`-I` would also drop
    `PYTHONPATH` and the user's site folder, which a user's own setup may rely on; the workspace is the threat.)"""
    argv = [sys.executable, "-P", "-m", "core.server", "serve", "--workspace", str(workspace)]
    return [*argv, "--model", model] if model else argv


def start_background(workspace: Path, model: str | None = None) -> Path:
    """Start the daemon for `workspace` as its own session (stdout and stderr to its log file) and wait until it
    answers; returns the socket path. `TimeoutError` when it did not come up in time. A thread waits for the process,
    so it never lingers as a zombie of this one once it exits."""
    ws = workspace.resolve()
    log_file = log_path(ws)
    fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        proc = subprocess.Popen(
            daemon_argv(ws, model),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=fd,
            stderr=fd,
            cwd=ws,
        )
    finally:
        os.close(fd)
    threading.Thread(target=proc.wait, name="agent-daemon-reaper", daemon=True).start()
    deadline = time.monotonic() + START_WAIT
    while time.monotonic() < deadline:
        if hello(ws) is not None:
            return socket_path(ws)
        time.sleep(POLL)
    raise TimeoutError(f"the agent daemon did not start within {START_WAIT:g} s (see {log_file})")


def ensure_daemon(workspace: Path, model: str | None = None) -> Path:
    """The socket of `workspace`'s daemon: the running one (whatever its model), else one started now on `model`
    (`start_background`)."""
    ws = workspace.resolve()
    if hello(ws) is not None:
        return socket_path(ws)
    return start_background(ws, model)


def stop(workspace: Path) -> bool:
    """`RunDirError` when the run folder is not ours alone. Ask the daemon for `workspace` to stop and wait until it let go of its pid file; False when none was running.

    "Gone" is the pid file's lock, not the pid: a daemon that exited but was not reaped yet (a zombie) still answers
    `kill(pid, 0)`, and a reused pid would answer for someone else. Only while the lock is held, so the pid in the file
    is the holder's, does a daemon that did not stop in time get SIGTERM (which runs the same cleanup)."""
    ws = workspace.resolve()
    if hello(ws) is None:  # also when there is no run folder: it is not made just to find nothing in it
        return False
    path = socket_path(ws)
    pids = pid_path(ws)
    with contextlib.suppress(httpx.HTTPError), _client(path, 5.0) as client:
        client.post("/v1/stop")
    deadline = time.monotonic() + STOP_WAIT
    while pid_held(pids):
        if time.monotonic() >= deadline:
            pid = read_pid(pids)
            if pid is not None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGTERM)
            break
        time.sleep(POLL)
    return True
