from __future__ import annotations

from typing import Any

import pytest
from src.engine.session.records import RecordError
from src.engine.usage import UsageRecord, UsageTotals, format_duration, summary, usage_from_obj


def rec(model="m", i=100, o=20, ms=1000, compaction=False) -> UsageRecord:
    return UsageRecord(model, i, o, ms, compaction, "2026-11-18T10:00:00Z")


def make(*records: UsageRecord, **kw):
    totals = UsageTotals()
    for r in records:
        totals = totals.add(r)
    args: dict[str, Any] = dict(session_id="S1", model="m", context_used=500, window=1000, turns=2, unrecorded=False)
    args.update(kw)
    return summary(totals, **args)


def rows(s) -> dict[str, str]:
    return dict(s.lines)


@pytest.mark.parametrize(
    "patch",
    [{"compaction": 1},
     {"at": 5}],
)
def test_bad_fields_raise(patch):
    with pytest.raises(RecordError):
        usage_from_obj({**rec().to_obj(), **patch})


def test_summary_rows():
    s = make(rec(), rec(), rec(i=50, o=10, ms=500, compaction=True))
    r = rows(s)
    assert r["Input tokens"] == "250" and r["Output tokens"] == "50" and r["Total"] == "300"
    assert r["Model calls"] == "3 (1 compaction: 60 tokens)"
    assert r["Turns in context"] == "2"
    assert r["Model time"] == "2.5s" and r["Tokens/s (replies)"] == "20"
    assert r["Context"] == "500 / 1,000 (50%)"
    assert rows(s)["Session"] == "S1 · m"
    assert s.note is None and not s.empty and s.models == ()


def test_unknown_counts_when_all_zero():
    r = rows(make(rec(i=0, o=0), rec(i=0, o=0)))
    assert r["Input tokens"] == r["Output tokens"] == r["Total"] == "unknown (the model reports no counts)"


@pytest.mark.parametrize(
    ("ms", "text"), [(161000, "2m 41s")]
)
def test_format_duration(ms, text):
    assert format_duration(ms) == text
