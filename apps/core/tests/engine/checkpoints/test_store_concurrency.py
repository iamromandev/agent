"""Concurrent CheckpointStore.snapshot calls (two sessions of one daemon): each is correct, none corrupts the index."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from core.engine.checkpoints.store import CheckpointStore
from core.engine.guardrails.setup import Guardrails

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def shadow(store: CheckpointStore, *args: str) -> str:
    done = subprocess.run(
        ["git", "--git-dir", str(store.folder), *args], capture_output=True, text=True, check=True, env={"PATH": os.environ["PATH"]}
    )
    return done.stdout


def make_store(tmp_path: Path, timeout: float | None = None) -> tuple[CheckpointStore, Path]:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    workspace = workspace.resolve()
    guard, env = Guardrails.off(workspace), dict(os.environ)
    if timeout is None:
        store = CheckpointStore(workspace, tmp_path / "data", guard, env)
    else:
        store = CheckpointStore(workspace, tmp_path / "data", guard, env, timeout=timeout)
    return store, workspace


async def test_cold_start_pair_and_concurrent_snapshots_are_each_correct(tmp_path: Path) -> None:
    # Bug caught: two snapshots racing on one index (`index.lock` failures turn checkpoints off) or two cold starts
    # both initialising the shadow repo.
    store, workspace = make_store(tmp_path)
    for i in range(5):
        (workspace / f"f{i}.txt").write_text("0")

    async def one(tag: str, n: int) -> list[tuple[str, dict[str, str]]]:
        seen = []
        for i in range(n):
            (workspace / f"{tag}.txt").write_text(str(i))
            snap = await store.snapshot()
            assert snap is not None, store.off_reason
            seen.append((snap.commit, {tag: str(i)}))
        return seen

    results = await asyncio.gather(one("a", 30), one("b", 30))

    assert store.enabled, store.off_reason
    for commit, expected in (r for res in results for r in res):
        for name, value in expected.items():
            assert shadow(store, "show", f"{commit}:{name}.txt") == value
    shadow(store, "fsck", "--strict")


async def test_waiting_for_the_lock_does_not_use_up_the_snapshot_time_limit(tmp_path: Path) -> None:
    # Bug caught: the time limit wrapping the wait too, so a snapshot queued behind a slow one turned checkpoints off.
    store, _ = make_store(tmp_path, timeout=0.2)
    async with store._lock:
        waiting = asyncio.ensure_future(store.snapshot())
        await asyncio.sleep(0.5)  # longer than the limit
        assert not waiting.done()

    snap = await asyncio.wait_for(waiting, 10)

    assert snap is not None and store.enabled, store.off_reason


async def test_cancelling_a_snapshot_that_waits_for_the_lock_is_safe_and_the_lock_stays_usable(tmp_path: Path) -> None:
    # Bug caught: a cancelled waiter leaving the lock held or the store off (Esc during `/undo` waits on this lock).
    store, workspace = make_store(tmp_path)
    (workspace / "f.txt").write_text("1")
    async with store._lock:
        waiting = asyncio.ensure_future(store.snapshot())
        await asyncio.sleep(0.05)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
    assert not store._lock.locked() and store.enabled

    snap = await store.snapshot()

    assert snap is not None, store.off_reason


async def test_a_waiter_gets_none_once_the_snapshot_ahead_of_it_timed_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug caught: a queued snapshot running (and failing late) after a timeout already turned checkpoints off.
    store, _ = make_store(tmp_path, timeout=0.2)
    real = store._snapshot
    calls = 0

    async def slow_first() -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(30)
        return await real()

    monkeypatch.setattr(store, "_snapshot", slow_first)
    first = asyncio.ensure_future(store.snapshot())
    await asyncio.sleep(0)
    second = asyncio.ensure_future(store.snapshot())

    assert await asyncio.wait_for(first, 10) is None
    assert await asyncio.wait_for(second, 10) is None
    assert calls == 1 and not store.enabled and "longer than" in (store.off_reason or "")
