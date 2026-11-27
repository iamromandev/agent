"""Helpers shared by the process tests: wait for a condition, check that a PID is gone."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from pathlib import Path

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


async def assert_all_gone(*pids: int) -> None:
    # A killed orphan stays a zombie until init reaps it, so allow a moment.
    await wait_until(lambda: all(gone(p) for p in pids), timeout=3.0)
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
