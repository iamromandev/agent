"""Helpers shared by the process tests: wait for a condition, check that a PID is gone."""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import os
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


async def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.02)


async def wait_for_pids(*files: Path) -> list[int]:
    await wait_until(lambda: all(f.exists() and f.read_text().strip() for f in files))
    return [int(f.read_text().strip()) for f in files]


def _hold_lock(directory: str, session_id: str, ready: Any, release: Any) -> None:  # in the spawned process
    from core.engine.session.lock import SessionLock

    lock = SessionLock.acquire(Path(directory), session_id)
    ready.set()
    release.wait(30)
    lock.release()


@contextlib.contextmanager
def session_held_elsewhere(directory: Path, session_id: str) -> Generator[None]:
    """Another process holds the session's lock (as another agent with the session open would) until the block ends."""
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    proc = ctx.Process(target=_hold_lock, args=(str(directory), session_id, ready, release))
    proc.start()
    try:
        assert ready.wait(30), "the other process did not take the lock"
        yield
    finally:
        release.set()
        proc.join(10)


async def assert_all_gone(*pids: int) -> None:
    # A killed orphan stays a zombie until init reaps it, so allow a moment.
    await wait_until(lambda: all(gone(p) for p in pids), timeout=3.0)
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
