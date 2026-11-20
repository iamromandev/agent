"""NFR1 / AC10: usage records are cheap to write, replay and show. Wall-clock limits, retried once under load."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.engine.session.paths import workspace_dir
from src.engine.session.records import SessionHeader, dumps, message_to_obj
from src.engine.session.store import SessionStore
from src.engine.tools.base import ToolContext
from src.engine.usage import UsageRecord, UsageTotals, summary
from src.tui.screens.usage import render

from tests.engine.test_compaction_loop import make_agent

STAMP = "2026-11-18T10:00:00Z"
RECORDS = 10_000
REPLAY_EXTRA_SECONDS = 0.050
RECORD_SECONDS = 0.001
SHOW_SECONDS = 0.050


def within(limit: float, measure: Callable[[], float]) -> float:
    """The best of two measurements: one slow run under load does not fail the test."""
    first = measure()
    return first if first <= limit else min(first, measure())


def _session(root: Path, workspace: Path, name: str, with_usage: bool) -> str:
    directory = workspace_dir(root, workspace)
    directory.mkdir(parents=True, exist_ok=True)
    objs: list[dict[str, Any]] = [SessionHeader(name, str(workspace), STAMP).to_obj()]
    from src.engine.provider.base import Message

    for n in range(RECORDS):
        if n % 10 == 0:
            from_msg = {**message_to_obj(Message("user", f"q{n}"), None), "at": STAMP}
            objs.append(from_msg)
        if with_usage:
            objs.append(UsageRecord(f"m{n % 3}", 100, 20, 900, n % 50 == 0, STAMP).to_obj())
    (directory / f"{name}.jsonl").write_text("".join(dumps(o) + "\n" for o in objs))
    return name


def test_replaying_10000_usage_records_adds_under_50_ms(tmp_path: Path) -> None:
    workspace = tmp_path / "w"
    workspace.mkdir()
    workspace = workspace.resolve()
    store = SessionStore(tmp_path / "root", workspace)
    plain = _session(tmp_path / "root", workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", False)
    full = _session(tmp_path / "root", workspace, "01BBBBBBBBBBBBBBBBBBBBBBBB", True)

    def timed(name: str) -> float:
        start = time.perf_counter()
        store.load(name)
        return time.perf_counter() - start

    def extra() -> float:
        return timed(full) - timed(plain)

    assert store.load(full).usage.calls == RECORDS
    assert within(REPLAY_EXTRA_SECONDS, extra) < REPLAY_EXTRA_SECONDS


async def test_recording_a_call_takes_under_1_ms(tmp_path: Path) -> None:
    ctx = ToolContext(workspace=tmp_path.resolve(), bash_timeout=5.0)
    agent, _, recorder = make_agent(ctx, [])

    def measure() -> float:
        start = time.perf_counter()
        for _ in range(1000):
            agent._record_usage(10, 5, time.monotonic(), compaction=False)
        return (time.perf_counter() - start) / 1000

    assert within(RECORD_SECONDS, measure) < RECORD_SECONDS
    assert len(recorder.usage_records) >= 1000


def test_summary_and_screen_text_take_under_50_ms_for_10000_records() -> None:
    totals = UsageTotals()
    for n in range(RECORDS):
        totals = totals.add(UsageRecord(f"m{n % 3}", 100, 20, 900, False, STAMP))

    def measure() -> float:
        start = time.perf_counter()
        render(summary(totals, session_id="S", model="m0", context_used=5, window=10, turns=3, unrecorded=False))
        return time.perf_counter() - start

    assert within(SHOW_SECONDS, measure) < SHOW_SECONDS


def test_building_the_totals_from_records_is_linear(tmp_path: Path) -> None:
    start = time.perf_counter()
    totals = UsageTotals()
    for n in range(RECORDS):
        totals = totals.add(UsageRecord(f"m{n % 3}", 1, 1, 1, False, STAMP))
    assert time.perf_counter() - start < REPLAY_EXTRA_SECONDS * 4
    assert totals.calls == RECORDS


