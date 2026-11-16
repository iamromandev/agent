"""The `hooks` key of a settings file: which commands run at which event. Pure, no I/O."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, get_args

from src.engine.permissions.rules import RuleError, Source, name_matcher
from src.engine.text import escape

Event = Literal["PreToolUse", "PostToolUse", "UserPromptSubmit", "Stop", "SessionStart"]
EVENTS: tuple[Event, ...] = get_args(Event)
TOOL_EVENTS: frozenset[str] = frozenset({"PreToolUse", "PostToolUse"})
DEFAULT_TIMEOUT = 60.0


@dataclass(frozen=True)
class HookEntry:
    event: Event
    command: str
    matcher: str | None  # None: every tool
    timeout: float
    source: Source

    @property
    def label(self) -> str:
        """`<Event> [<matcher>]`, with `[*]` when there is no matcher (every tool, or an event without tools). Raw: escape it."""
        return f"{self.event} [{self.matcher or '*'}]"

    def matches(self, tool: str) -> bool:
        """Does this entry apply to a call of `tool`? (The matcher was checked when the entry was read.)"""
        return self.matcher is None or name_matcher(self.matcher)(tool)


def read_hooks(data: Mapping[str, object] | None, source: Source) -> tuple[list[HookEntry], list[str]]:
    """The entries of `data["hooks"]` (a settings file's parsed object) and the launch warnings, escaped.

    A bad entry is one warning and is skipped; the others still load."""
    raw = data.get("hooks") if data is not None else None
    if raw is None:
        return [], []
    where = f"hooks ({source})"
    if not isinstance(raw, dict):
        return [], [f"{where}: hooks must be an object of event names to lists; no hooks loaded"]
    entries: list[HookEntry] = []
    warnings: list[str] = []
    for name, items in raw.items():
        if name not in EVENTS:
            warnings.append(escape(f'{where}: unknown event "{name}"; its entries are skipped'))
            continue
        if not isinstance(items, list):
            warnings.append(escape(f"{where}: {name} must be a list of entries; it adds no hooks"))
            continue
        for index, item in enumerate(items, 1):
            entry, warning = _entry(name, item, source)
            if warning is not None:
                warnings.append(escape(f"{where}: {name} #{index}: {warning}"))
            if entry is not None:
                entries.append(entry)
    return entries, warnings


def _entry(event: Event, item: object, source: Source) -> tuple[HookEntry | None, str | None]:
    """The entry for one item and a warning: no entry when it is unusable, both when a part of it is ignored."""
    if not isinstance(item, dict):
        return None, "entry must be an object; skipped"
    command = item.get("command")
    if not isinstance(command, str) or not command.strip():
        return None, "command must be a non-empty string; skipped"
    timeout = item.get("timeout", DEFAULT_TIMEOUT)
    if isinstance(timeout, bool) or not isinstance(timeout, int | float) or not math.isfinite(timeout) or timeout <= 0:
        return None, "timeout must be a positive number; skipped"
    matcher = item.get("matcher")
    warning = None
    if matcher is not None:
        if not isinstance(matcher, str):
            return None, "matcher must be a string; skipped"
        if event not in TOOL_EVENTS:
            matcher, warning = None, f"matcher only applies to {' and '.join(sorted(TOOL_EVENTS))}; ignored"
        else:
            try:
                name_matcher(matcher)
            except RuleError as exc:
                return None, f"matcher {matcher!r}: {exc}; skipped"
            if matcher == "*":
                matcher = None
    return HookEntry(event, command, matcher, float(timeout), source), warning
