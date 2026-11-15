"""Aggregation, table and JSON report (AC7)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evals.record import RunRecord
from evals.report import Meta, commit_hash, summarize, table, to_json, write_json


def _rec(task: str, run: int, passed: bool, steps: int, tokens: int, seconds: float, tools: dict[str, int]) -> RunRecord:
    record = RunRecord(task, run, passed=passed, steps=steps, tokens=tokens, seconds=seconds, tools=Counter(tools))
    if not passed:
        record.reasons.append("nope")
    return record


RECORDS = [
    _rec("find", 1, True, 6, 5000, 41.0, {"task": 1}),
    _rec("find", 2, False, 4, 5100, 30.0, {"task": 1, "grep": 2}),
    _rec("find", 3, True, 8, 9000, 130.0, {"task": 1}),
    _rec("rules", 1, True, 3, 2000, 18.0, {"bash": 2}),
]


def test_summarize_gives_passes_medians_and_tool_usage_per_task() -> None:
    find, rules = summarize(RECORDS)
    assert (find.task, find.passes, find.runs) == ("find", 2, 3)
    assert (find.steps, find.tokens, find.seconds) == (6, 5100, 41.0)
    assert find.tools == "task 3/3, grep 1/3"
    assert (rules.passes, rules.runs, rules.tools) == (1, 1, "bash 1/1")


def test_the_table_shows_the_header_and_one_row_per_task() -> None:
    text = table(summarize(RECORDS), "model m · ctx 8192 · 3 runs/task")
    lines = text.splitlines()
    assert lines[0] == "model m · ctx 8192 · 3 runs/task"
    assert lines[1].split() == ["task", "pass", "steps(med)", "tokens(med)", "time(med)", "tools", "used"]
    assert lines[2].split() == ["find", "2/3", "6", "5.1k", "41s", "task", "3/3,", "grep", "1/3"]
    assert lines[3].split()[:6] == ["rules", "1/1", "3", "2.0k", "18s", "bash"]


def test_long_times_use_minutes() -> None:
    row = summarize([_rec("x", 1, True, 1, 10, 130.0, {})])
    assert "2m10s" in table(row, "h")


META = Meta(model="m", num_ctx=8192, runs=3, tasks=("find", "rules"), timeout=300.0, commit="abc123", started="2026-11-16T10-02-11")


def test_json_has_the_documented_keys() -> None:
    data = to_json(RECORDS, META)
    assert set(data) == {"model", "num_ctx", "runs", "tasks", "timeout", "commit", "started", "results"}
    assert data["commit"] == "abc123"
    first = data["results"][0]
    assert set(first) == {"task", "run", "passed", "reasons", "steps", "tokens", "seconds", "tools", "approvals", "error", "notes"}
    assert first["tools"] == {"task": 1}
    assert len(data["results"]) == 4


def test_write_json_creates_the_folder_and_the_file(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "r.json"
    write_json(path, to_json(RECORDS, META))
    assert json.loads(path.read_text())["model"] == "m"


def test_commit_hash_of_this_repo_is_a_hash() -> None:
    assert len(commit_hash()) == 40


def test_commit_hash_is_unknown_outside_a_repo(tmp_path: Path) -> None:
    assert commit_hash(tmp_path) == "unknown"


def test_unknown_tokens_show_as_a_question_mark_and_null() -> None:
    record = RunRecord("find", 1, passed=True, steps=2, seconds=5.0)
    (row,) = summarize([record])
    assert row.tokens is None
    assert "?" in table([row], "h").splitlines()[2]
    assert to_json([record], META)["results"][0]["tokens"] is None
