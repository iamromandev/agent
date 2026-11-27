"""NFR1 / AC10: usage records are cheap to write, replay and show. Wall-clock limits, retried once under load."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.engine.session.paths import workspace_dir
from core.engine.session.records import SessionHeader, dumps, message_to_obj
from core.engine.usage import UsageRecord, UsageTotals, summary
from src.tui.screens.usage import render

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
    from core.engine.provider.base import Message

    for n in range(RECORDS):
        if n % 10 == 0:
            from_msg = {**message_to_obj(Message("user", f"q{n}"), None), "at": STAMP}
            objs.append(from_msg)
        if with_usage:
            objs.append(UsageRecord(f"m{n % 3}", 100, 20, 900, n % 50 == 0, STAMP).to_obj())
    (directory / f"{name}.jsonl").write_text("".join(dumps(o) + "\n" for o in objs))
    return name


def test_summary_and_screen_text_take_under_50_ms_for_10000_records() -> None:
    totals = UsageTotals()
    for n in range(RECORDS):
        totals = totals.add(UsageRecord(f"m{n % 3}", 100, 20, 900, False, STAMP))

    def measure() -> float:
        start = time.perf_counter()
        render(summary(totals, session_id="S", model="m0", context_used=5, window=10, turns=3, unrecorded=False))
        return time.perf_counter() - start

    assert within(SHOW_SECONDS, measure) < SHOW_SECONDS
