from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import pytest
from src.agent.terminal.pty import PtyProcess

from tests.helpers import gone

SpawnFn = Callable[..., PtyProcess]


@pytest.fixture
def pty_env() -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "TERM": "xterm"}


@pytest.fixture
def spawn_pty(tmp_path: Path, pty_env: dict[str, str]) -> Iterator[SpawnFn]:
    """Spawn processes on a pty; whatever a test leaves running is killed, reaped and closed afterwards."""
    spawned: list[PtyProcess] = []

    def spawn(
        argv: Sequence[str],
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        cols: int = 80,
        rows: int = 24,
    ) -> PtyProcess:
        process = PtyProcess.spawn(argv, cwd or tmp_path, env or pty_env, cols, rows)
        spawned.append(process)
        return process

    yield spawn
    for process in spawned:
        process.signal_group(signal.SIGKILL)
    deadline = time.monotonic() + 5
    while any(p.poll() is None for p in spawned) and time.monotonic() < deadline:
        time.sleep(0.01)
    for process in spawned:
        process.close()
    leaked = [p.pid for p in spawned if not gone(p.pid)]
    assert leaked == [], f"processes left running: {leaked}"


def read_until(process: PtyProcess, needle: bytes, timeout: float = 5.0) -> bytes:
    """Everything the process wrote to its pty up to and including `needle`."""
    import select

    poller = select.poll()  # not select.select: descriptors above 1023 are out of its range
    poller.register(process.fd, select.POLLIN)
    seen = b""
    deadline = time.monotonic() + timeout
    while needle not in seen:
        left = deadline - time.monotonic()
        assert left > 0, f"{needle!r} not seen in time; got {seen!r}"
        if poller.poll(left * 1000):
            try:
                chunk = os.read(process.fd, 65_536)
            except BlockingIOError:
                continue
            except OSError:  # EIO: the other side closed
                break
            if not chunk:
                break
            seen += chunk
    assert needle in seen, f"{needle!r} not seen; got {seen!r}"
    return seen


def wait_exit(process: PtyProcess, timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    while (code := process.poll()) is None:
        assert time.monotonic() < deadline, "process did not exit in time"
        time.sleep(0.01)
    return code


@pytest.fixture(autouse=True)
def _posix_sh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Terminals run $SHELL; pin it so the tests do not depend on the developer's shell."""
    monkeypatch.setenv("SHELL", "/bin/sh")

