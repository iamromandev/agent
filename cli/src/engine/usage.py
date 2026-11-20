"""Token usage of a session: the records, their running totals and what `/usage` shows. UI-free."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from src.engine.session.records import RecordError
from src.engine.text import escape_display

MODEL_LIMIT = 1_000
UNKNOWN_COUNTS = "unknown (the model reports no counts)"
NO_RECORDS_NOTE = "some of this session has no usage records"
EMPTY_TEXT = "No model calls yet"
NO_MODEL = "unknown model"


@dataclass(frozen=True)
class UsageRecord:
    model: str
    input_tokens: int
    output_tokens: int
    duration_ms: int
    compaction: bool
    at: str  # ISO 8601 UTC

    def to_obj(self) -> dict[str, object]:
        return {
            "type": "usage",
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "duration_ms": self.duration_ms,
            "compaction": self.compaction,
            "at": self.at,
        }


@dataclass(frozen=True)
class ModelUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0  # non-compaction calls only
    out_for_rate: int = 0  # output tokens of non-compaction calls with duration > 0
    rate_ms: int = 0  # their durations
    compactions: int = 0
    compaction_tokens: int = 0  # input + output of compaction calls
    all_duration_ms: int = 0  # every call (model time)

    def add(self, record: UsageRecord) -> ModelUsage:
        compaction = record.compaction
        timed = not compaction and record.duration_ms > 0
        return ModelUsage(
            calls=self.calls + 1,
            input_tokens=self.input_tokens + record.input_tokens,
            output_tokens=self.output_tokens + record.output_tokens,
            duration_ms=self.duration_ms + (0 if compaction else record.duration_ms),
            out_for_rate=self.out_for_rate + (record.output_tokens if timed else 0),
            rate_ms=self.rate_ms + (record.duration_ms if timed else 0),
            compactions=self.compactions + (1 if compaction else 0),
            compaction_tokens=self.compaction_tokens + (record.input_tokens + record.output_tokens if compaction else 0),
            all_duration_ms=self.all_duration_ms + record.duration_ms,
        )

    def merged(self, other: ModelUsage) -> ModelUsage:
        fields = ModelUsage.__dataclass_fields__
        return ModelUsage(**{f: getattr(self, f) + getattr(other, f) for f in fields})


@dataclass(frozen=True)
class UsageTotals:
    by_model: tuple[tuple[str, ModelUsage], ...] = ()  # first-use order

    def add(self, record: UsageRecord) -> UsageTotals:
        for n, (name, usage) in enumerate(self.by_model):
            if name == record.model:
                updated = (*self.by_model[:n], (name, usage.add(record)), *self.by_model[n + 1 :])
                return replace(self, by_model=updated)
        return replace(self, by_model=(*self.by_model, (record.model, ModelUsage().add(record))))

    @property
    def total(self) -> ModelUsage:
        result = ModelUsage()
        for _, usage in self.by_model:
            result = result.merged(usage)
        return result

    @property
    def calls(self) -> int:
        return sum(usage.calls for _, usage in self.by_model)


class _Sums:
    """Mutable counters named like `ModelUsage`'s fields, so a field reorder cannot misplace one."""

    __slots__ = (
        "all_duration_ms",
        "calls",
        "compaction_tokens",
        "compactions",
        "duration_ms",
        "input_tokens",
        "out_for_rate",
        "output_tokens",
        "rate_ms",
    )

    def __init__(self) -> None:
        self.calls = self.input_tokens = self.output_tokens = self.duration_ms = 0
        self.out_for_rate = self.rate_ms = self.compactions = self.compaction_tokens = self.all_duration_ms = 0


class UsageAccumulator:
    """Sums many records without building a new frozen value for each: the replay of a long session uses it."""

    def __init__(self) -> None:
        self._sums: dict[str, _Sums] = {}

    def add(self, record: UsageRecord) -> None:
        sums = self._sums.get(record.model)
        if sums is None:
            sums = self._sums[record.model] = _Sums()
        tokens = record.input_tokens + record.output_tokens
        sums.calls += 1
        sums.input_tokens += record.input_tokens
        sums.output_tokens += record.output_tokens
        sums.all_duration_ms += record.duration_ms
        if record.compaction:
            sums.compactions += 1
            sums.compaction_tokens += tokens
        else:
            sums.duration_ms += record.duration_ms
            if record.duration_ms > 0:
                sums.out_for_rate += record.output_tokens
                sums.rate_ms += record.duration_ms

    def totals(self) -> UsageTotals:
        fields = ModelUsage.__dataclass_fields__
        return UsageTotals(
            tuple((name, ModelUsage(**{f: getattr(sums, f) for f in fields})) for name, sums in self._sums.items())
        )


def _count(obj: Mapping[str, object], key: str) -> int:
    value = obj.get(key)
    if type(value) is not int or value < 0:  # exact type: a bool is not a count
        raise RecordError(f"bad usage record: {key}")
    return value


def usage_from_obj(obj: Mapping[str, object]) -> UsageRecord:
    model, compaction, at = obj.get("model"), obj.get("compaction"), obj.get("at")
    if not isinstance(model, str) or len(model) > MODEL_LIMIT:
        raise RecordError("bad usage record: model")
    if not isinstance(compaction, bool):
        raise RecordError("bad usage record: compaction")
    if not isinstance(at, str):
        raise RecordError("bad usage record: at")
    return UsageRecord(
        model, _count(obj, "input_tokens"), _count(obj, "output_tokens"), _count(obj, "duration_ms"), compaction, at
    )


def format_duration(ms: int) -> str:
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    whole = int(seconds)
    if whole < 3600:
        return f"{whole // 60}m {whole % 60}s"
    return f"{whole // 3600}h {whole % 3600 // 60}m"


@dataclass(frozen=True)
class UsageSummary:
    lines: tuple[tuple[str, str], ...]  # (label, value) rows, plain text, names escaped
    models: tuple[tuple[str, str, str, str], ...]  # (name, input, output, calls) when more than one model
    note: str | None
    empty: bool


def _name(model: str) -> str:
    return escape_display(model) if model else NO_MODEL


def summary(
    totals: UsageTotals,
    *,
    session_id: str,
    model: str,
    context_used: int,
    window: int,
    turns: int,
    unrecorded: bool,
    estimated: bool = False,
) -> UsageSummary:
    total = totals.total
    shown_id = escape_display(session_id) if session_id else "new session"
    lines: list[tuple[str, str]] = [("Session", f"{shown_id} · {_name(model)}")]
    pct = f"{context_used * 100 // window}%" if window > 0 else "?"
    context = ("Context", f"{'~' if estimated else ''}{context_used:,} / {window:,} ({pct})")
    note = NO_RECORDS_NOTE if unrecorded else None
    if total.calls == 0:
        return UsageSummary((*lines, context), (), note, True)
    if total.input_tokens == 0 and total.output_tokens == 0:
        lines += [(label, UNKNOWN_COUNTS) for label in ("Input tokens", "Output tokens", "Total")]
    else:
        lines += [
            ("Input tokens", f"{total.input_tokens:,}"),
            ("Output tokens", f"{total.output_tokens:,}"),
            ("Total", f"{total.input_tokens + total.output_tokens:,}"),
        ]
    calls = f"{total.calls:,}"
    if total.compactions:
        word = "compaction" if total.compactions == 1 else "compactions"
        calls += f" ({total.compactions} {word}: {total.compaction_tokens:,} tokens)"
    rate = f"{total.out_for_rate * 1000 // total.rate_ms:,}" if total.rate_ms > 0 else "—"
    lines += [
        ("Model calls", calls),
        ("Turns in context", f"{turns:,}"),
        ("Model time", format_duration(total.all_duration_ms)),
        ("Tokens/s (replies)", rate),
        context,
    ]
    models: tuple[tuple[str, str, str, str], ...] = ()
    if len(totals.by_model) > 1:
        models = tuple(
            (_name(n), f"{u.input_tokens:,}", f"{u.output_tokens:,}", f"{u.calls:,}") for n, u in totals.by_model
        )
    return UsageSummary(tuple(lines), models, note, False)
