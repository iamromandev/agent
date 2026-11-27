from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from pathlib import Path

import pytest
from core.engine.terminal import pty as pty_module
from core.engine.terminal.pty import PtyProcess
from core.testing.helpers import gone, wait_until

from tests.engine.terminal.conftest import SpawnFn, read_until, wait_exit


async def test_a_large_write_is_delivered_whole(spawn_pty: SpawnFn) -> None:
    # raw mode: in canonical mode the line discipline holds back input that has no newline
    process = spawn_pty(["sh", "-c", "stty raw -echo; echo ready; head -c 200000 > /dev/null; echo done-reading"])
    read_until(process, b"ready")

    process.write(b"x" * 200_000)  # far more than the pty queue takes: the rest waits for the loop to flush

    seen = bytearray()

    def finished() -> bool:
        seen.extend(_drain(process))
        return b"done-reading" in seen

    await wait_until(finished)


def _drain(process: PtyProcess) -> bytes:
    """What the process has written so far (nothing when it has written nothing)."""
    with contextlib.suppress(OSError):  # nothing to read now, or EIO
        return os.read(process.fd, 65_536)
    return b""


async def test_input_beyond_the_cap_is_refused_with_a_clear_error(
    spawn_pty: SpawnFn, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pty_module, "MAX_PENDING", 10_000)
    process = spawn_pty(["sh", "-c", "stty raw -echo; echo ready; sleep 30"])
    read_until(process, b"ready")

    with pytest.raises(pty_module.WriteOverflow, match="input dropped"):
        for _ in range(100):
            process.write(b"b" * 4_000)


async def test_closing_drops_the_queued_input_and_stops_watching(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sh", "-c", "stty raw -echo; echo ready; sleep 30"])
    read_until(process, b"ready")
    process.write(b"z" * 100_000)

    process.close()

    process.write(b"late")  # dropped quietly
    await asyncio.sleep(0.05)  # a stale writer on a closed descriptor would blow up the loop


def test_writing_after_the_process_exited_is_dropped(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sh", "-c", "exit 0"])
    wait_exit(process)

    process.write(b"late input\n")


def test_the_exit_hook_kills_the_groups_still_alive(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sleep", "30"])

    pty_module._kill_live_groups()

    assert wait_exit(process) == -signal.SIGKILL


def test_a_gone_group_is_never_signalled_again(spawn_pty: SpawnFn, monkeypatch: pytest.MonkeyPatch) -> None:
    process = spawn_pty(["sh", "-c", "exit 0"])
    wait_exit(process)
    assert not process.group_alive()  # reaped and empty: the id may now belong to somebody else
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))

    pty_module._kill_live_groups()
    process.signal_group(signal.SIGKILL)
    process.signal_tree(signal.SIGKILL)

    assert sent == []


# --- spawn failing ---------------------------------------------------------------------------------


def _open_fds() -> int:
    return len(os.listdir("/dev/fd"))


def _record_forks(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    pids: list[int] = []
    real_fork = os.fork

    def fork() -> int:
        pid = real_fork()
        if pid:
            pids.append(pid)
        return pid

    monkeypatch.setattr(os, "fork", fork)
    return pids


def test_a_child_that_never_execs_is_killed_and_reaped_after_the_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pty_env: dict[str, str]
) -> None:
    monkeypatch.setattr(pty_module, "_SPAWN_WAIT", 0.3)
    monkeypatch.setattr(os, "execvpe", lambda *_: time.sleep(60))  # the child hangs before exec
    pids = _record_forks(monkeypatch)
    fds = _open_fds()

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="did not start"):
        PtyProcess.spawn(["sleep", "30"], tmp_path, pty_env, 80, 24)

    assert time.monotonic() - started < 3
    assert len(pids) == 1
    assert gone(pids[0])
    assert _open_fds() == fds


def test_an_interrupted_spawn_kills_and_reaps_the_child_and_closes_its_fds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pty_env: dict[str, str]
) -> None:
    pids = _record_forks(monkeypatch)
    fds = _open_fds()

    def interrupt(fd: int, timeout: float) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(pty_module, "_wait_readable", interrupt)

    with pytest.raises(KeyboardInterrupt):
        PtyProcess.spawn(["sleep", "30"], tmp_path, pty_env, 80, 24)

    assert len(pids) == 1
    assert gone(pids[0])
    assert _open_fds() == fds


def test_a_failing_pipe_does_not_leak_the_pty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pty_env: dict[str, str]) -> None:
    def broken() -> tuple[int, int]:
        raise OSError("no pipes")

    monkeypatch.setattr(os, "pipe", broken)
    fds = _open_fds()

    with pytest.raises(OSError, match="no pipes"):
        PtyProcess.spawn(["sleep", "30"], tmp_path, pty_env, 80, 24)

    assert _open_fds() == fds
