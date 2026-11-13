"""NFR1: journaling a 1 MB write costs under 20 ms; undoing 50 files of 100 KB takes under 1 s."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest
from src.agent.undo import UndoLog, restore_turn

JOURNAL_OVERHEAD_SECONDS = 0.020
UNDO_SECONDS = 1.0


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root.resolve()


def _timed_write(path: Path, data: bytes, log: UndoLog | None) -> float:
    start = time.perf_counter()
    if log is not None:
        log.before(path)
    path.write_bytes(data)
    if log is not None:
        log.after(path)
    return time.perf_counter() - start


def test_journaling_a_1mb_write_adds_under_20_ms(workspace: Path) -> None:
    target = workspace / "big.bin"
    target.write_bytes(b"a" * 1_000_000)
    new = b"b" * 1_000_000
    plain: list[float] = []
    journaled: list[float] = []
    for round_ in range(5):
        plain.append(_timed_write(target, bytes([65 + round_]) * 1_000_000, None))
        log = UndoLog(workspace)
        log.begin(0, "p")
        journaled.append(_timed_write(target, new if round_ % 2 else b"c" * 1_000_000, log))

    overhead = statistics.median(journaled) - statistics.median(plain)
    print(f"NFR1 journal overhead={overhead * 1000:.1f}ms")
    assert overhead < JOURNAL_OVERHEAD_SECONDS


def test_undoing_50_files_of_100kb_takes_under_a_second(workspace: Path) -> None:
    log = UndoLog(workspace)
    log.begin(0, "p")
    for index in range(50):
        path = workspace / f"f{index:02}.bin"
        path.write_bytes(bytes([index]) * 100_000)
        log.before(path)
        path.write_bytes(b"new" * 33_000)
        log.after(path)
    log.finish()
    turn = log.pop()
    assert turn is not None

    start = time.perf_counter()
    result = restore_turn(turn, workspace)
    seconds = time.perf_counter() - start

    print(f"NFR1 undo of 50x100KB={seconds:.3f}s")
    assert [e[0] for e in result.entries] == ["restored"] * 50
    assert seconds < UNDO_SECONDS
