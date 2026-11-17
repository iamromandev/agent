"""NFR1: the first snapshot of 5,000 small files in 50 folders is under 2 s, an unchanged one under 0.3 s."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest
from src.engine.checkpoints.store import CheckpointStore
from src.engine.guardrails.setup import Guardrails

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

FIRST_SECONDS = 2.0
UNCHANGED_SECONDS = 0.3


def _fill(workspace: Path) -> None:
    for folder in range(50):
        base = workspace / f"pkg{folder:02}"
        base.mkdir()
        for index in range(100):
            (base / f"file{index:03}.py").write_text(f"print({folder}, {index})\n")


async def _measure(tmp_path: Path, attempt: int) -> tuple[float, float]:
    workspace = (tmp_path / f"ws{attempt}").resolve()
    workspace.mkdir()
    _fill(workspace)
    store = CheckpointStore(
        workspace, tmp_path / f"data{attempt}" / "checkpoints", Guardrails.off(workspace), dict(os.environ), timeout=30
    )
    try:
        start = time.perf_counter()
        first = await store.snapshot()
        first_seconds = time.perf_counter() - start
        start = time.perf_counter()
        second = await store.snapshot()
        second_seconds = time.perf_counter() - start
        assert first is not None and second is not None and first.commit == second.commit
        return first_seconds, second_seconds
    finally:
        store.close()


async def test_a_first_snapshot_of_5000_files_is_under_2_s_and_an_unchanged_one_under_0_3_s(tmp_path: Path) -> None:
    for attempt in range(2):  # retry once, like the other perf tests: a loaded machine is not a regression
        first, second = await _measure(tmp_path, attempt)
        print(f"NFR1 checkpoints: first={first:.2f}s unchanged={second:.3f}s")
        if first < FIRST_SECONDS and second < UNCHANGED_SECONDS:
            return
    assert first < FIRST_SECONDS
    assert second < UNCHANGED_SECONDS
