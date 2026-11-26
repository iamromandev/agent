"""Aggregate the runs of an eval into a table and a JSON report."""

from __future__ import annotations

import json
import statistics
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.record import RunRecord

REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Row:
    task: str
    passes: int
    runs: int
    steps: float  # medians
    tokens: float | None  # None: no run reported usage
    seconds: float
    tools: str  # "name k/N": in how many runs the tool was used


@dataclass(frozen=True)
class Meta:
    model: str
    num_ctx: int
    runs: int
    tasks: tuple[str, ...]
    timeout: float
    commit: str
    started: str


def summarize(records: list[RunRecord]) -> list[Row]:
    by_task: dict[str, list[RunRecord]] = {}
    for record in records:
        by_task.setdefault(record.task, []).append(record)
    rows: list[Row] = []
    for task, runs in by_task.items():
        used: Counter[str] = Counter()
        for record in runs:
            used.update(name for name, count in record.tools.items() if count > 0)
        ordered = sorted(used.items(), key=lambda item: (-item[1], item[0]))
        rows.append(
            Row(
                task,
                sum(1 for r in runs if r.passed),
                len(runs),
                statistics.median(r.steps for r in runs),
                statistics.median(known) if (known := [r.tokens for r in runs if r.tokens is not None]) else None,
                statistics.median(r.seconds for r in runs),
                ", ".join(f"{name} {k}/{len(runs)}" for name, k in ordered),
            )
        )
    return rows


def _number(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:.1f}"


def _tokens(value: float | None) -> str:
    if value is None:
        return "?"
    return f"{value / 1000:.1f}k" if value >= 1000 else _number(value)


def _time(seconds: float) -> str:
    whole = round(seconds)
    return f"{whole // 60}m{whole % 60:02d}s" if whole >= 120 else f"{whole}s"


def table(rows: list[Row], header: str) -> str:
    cells = [("task", "pass", "steps(med)", "tokens(med)", "time(med)", "tools used")]
    cells += [
        (r.task, f"{r.passes}/{r.runs}", _number(r.steps), _tokens(r.tokens), _time(r.seconds), r.tools) for r in rows
    ]
    widths = [max(len(row[i]) for row in cells) for i in range(5)]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row[:5], widths, strict=True)) + "  " + row[5] for row in cells]
    return "\n".join([header, *(line.rstrip() for line in lines)])


def to_json(records: list[RunRecord], meta: Meta) -> dict[str, Any]:
    return {
        "model": meta.model,
        "num_ctx": meta.num_ctx,
        "runs": meta.runs,
        "tasks": list(meta.tasks),
        "timeout": meta.timeout,
        "commit": meta.commit,
        "started": meta.started,
        "results": [
            {
                "task": r.task,
                "run": r.run,
                "passed": r.passed,
                "reasons": r.reasons,
                "steps": r.steps,
                "tokens": r.tokens,
                "seconds": round(r.seconds, 2),
                "tools": dict(r.tools),
                "approvals": r.approvals,
                "error": r.error,
                "notes": r.notes,
            }
            for r in records
        ],
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def commit_hash(folder: Path = REPO) -> str:
    """The git commit of the repo at `folder`, or "unknown"."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=folder, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return done.stdout.strip() or "unknown"
