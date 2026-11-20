from __future__ import annotations

from typing import Any

import pytest
from src.engine.session.records import RecordError
from src.engine.usage import UsageAccumulator, UsageRecord, UsageTotals, format_duration, summary, usage_from_obj


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


def test_to_obj_and_back():
    r = rec()
    assert r.to_obj() == {
        "type": "usage", "model": "m", "input_tokens": 100, "output_tokens": 20,
        "duration_ms": 1000, "compaction": False, "at": "2026-11-18T10:00:00Z",
    }
    assert usage_from_obj(r.to_obj()) == r


@pytest.mark.parametrize(
    "patch",
    [{"input_tokens": -1}, {"output_tokens": "5"}, {"duration_ms": True}, {"model": 3}, {"compaction": 1},
     {"model": "x" * 1001}, {"at": 5}, {"input_tokens": None}],
)
def test_bad_fields_raise(patch):
    with pytest.raises(RecordError):
        usage_from_obj({**rec().to_obj(), **patch})


def test_totals_sum_by_model_in_first_use_order():
    t = UsageTotals().add(rec("b")).add(rec("a")).add(rec("b", i=1, o=2))
    assert [n for n, _ in t.by_model] == ["b", "a"]
    assert t.total.input_tokens == 201 and t.total.output_tokens == 42
    assert t.calls == 3


def test_compaction_counted_apart_from_rate():
    t = UsageTotals().add(rec(ms=2000)).add(rec(i=50, o=10, ms=500, compaction=True))
    u = t.total
    assert u.compactions == 1 and u.compaction_tokens == 60
    assert u.out_for_rate == 20 and u.rate_ms == 2000 and u.all_duration_ms == 2500


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


def test_thousands_separators():
    assert rows(make(rec(i=48210, o=3904)))["Input tokens"] == "48,210"


def test_per_model_rows():
    s = make(rec("a"), rec("b", i=5, o=6))
    assert len(s.models) == 2
    assert s.models[0][0] == "a" and s.models[1][0] == "b"


def test_unknown_counts_when_all_zero():
    r = rows(make(rec(i=0, o=0), rec(i=0, o=0)))
    assert r["Input tokens"] == r["Output tokens"] == r["Total"] == "unknown (the model reports no counts)"


def test_empty():
    s = make()
    assert s.empty
    assert [k for k, _ in s.lines] == ["Session", "Context"]


def test_rate_dash_without_duration():
    assert rows(make(rec(ms=0)))["Tokens/s (replies)"] == "—"


def test_unrecorded_note():
    assert make(rec(), unrecorded=True).note == "some of this session has no usage records"


def test_names_escaped():
    s = make(rec("m\x1b[31m"), session_id="S\x1b")
    text = "".join(a + b for a, b in s.lines) + "".join("".join(m) for m in s.models)
    assert "\x1b" not in text
    assert "\x1b" not in s.lines[0][1] + s.lines[0][0]


@pytest.mark.parametrize(
    ("ms", "text"), [(0, "0.0s"), (8100, "8.1s"), (161000, "2m 41s"), (3720000, "1h 2m"), (59999, "60.0s")]
)
def test_format_duration(ms, text):
    assert format_duration(ms) == text


def test_accumulator_equals_adding_one_by_one():
    records = [rec("a"), rec("b", i=5, o=6, ms=0), rec("a", i=7, o=8, ms=300, compaction=True), rec("b")]
    one_by_one = UsageTotals()
    acc = UsageAccumulator()
    for r in records:
        one_by_one = one_by_one.add(r)
        acc.add(r)
    assert acc.totals() == one_by_one
