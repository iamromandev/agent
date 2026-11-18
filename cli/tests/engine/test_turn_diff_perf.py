"""NFR1: the file list of a 200-file turn in a 5,000-file workspace within 300 ms; one 2,000-line file within 200 ms."""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from src.engine.checkpoints.store import CheckpointStore
from src.engine.guardrails.setup import Guardrails
from src.engine.turn_diff import build_diff, load_file, render_lines
from src.engine.undo import UndoTurn

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

LIST_SECONDS = 0.3
FILE_SECONDS = 0.2


async def best_of_two(run: Callable[[], Awaitable[object]]) -> float:
    """The faster of two runs: a loaded machine fails a timing test once, not twice."""
    times = []
    for _ in range(2):
        start = time.perf_counter()
        await run()
        times.append(time.perf_counter() - start)
    return min(times)


async def test_listing_200_changed_files_in_a_5000_file_workspace_takes_under_300_ms(tmp_path: Path) -> None:
    ws = (tmp_path / "ws").resolve()
    for n in range(5_000):
        folder = ws / f"d{n % 50}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"f{n}.txt").write_text(f"line a {n}\nline b\nline c\n")
    store = CheckpointStore(ws, tmp_path / "data", Guardrails.off(ws), dict(os.environ), timeout=60)
    first = await store.snapshot()
    for n in range(200):
        (ws / f"d{n % 50}" / f"f{n}.txt").write_text(f"line a {n}\nCHANGED\nline c\nextra\n")
    second = await store.snapshot()
    assert first is not None and second is not None
    turn = UndoTurn(0, "p", before=first.commit, after=second.commit)

    seconds = await best_of_two(lambda: build_diff(turn, store, ws))

    diff = await build_diff(turn, store, ws)
    assert len(diff.items) == 200
    print(f"list of 200 files: {seconds * 1000:.0f} ms")
    assert seconds < LIST_SECONDS


async def test_a_2000_line_file_loads_and_renders_in_under_200_ms(tmp_path: Path) -> None:
    ws = (tmp_path / "ws").resolve()
    ws.mkdir()
    (ws / "big.py").write_text("".join(f"line {n}\n" for n in range(2_000)))
    store = CheckpointStore(ws, tmp_path / "data", Guardrails.off(ws), dict(os.environ))
    first = await store.snapshot()
    (ws / "big.py").write_text("".join(f"line {n}{' edited' if n % 3 == 0 else ''}\n" for n in range(2_000)))
    second = await store.snapshot()
    assert first is not None and second is not None
    diff = await build_diff(UndoTurn(0, "p", before=first.commit, after=second.commit), store, ws)

    async def one() -> int:
        return len(render_lines(await load_file(diff, diff.items[0])))

    seconds = await best_of_two(one)

    assert await one() > 2_000
    print(f"2000-line file: {seconds * 1000:.0f} ms")
    assert seconds < FILE_SECONDS
