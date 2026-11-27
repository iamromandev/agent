"""The queue panel's lines (the queue itself is `core.host.queue`)."""

from __future__ import annotations

from collections.abc import Sequence

from core.engine.text import escape_display
from core.host.queue import Queued

SHOWN_LINES = 5
SHOWN_CHARS = 80


def panel_lines(entries: Sequence[Queued]) -> list[str]:
    """One `<first line>` per entry, at most 5, then `and N more`. Only the shown entries are read."""
    lines = []
    for entry in entries[:SHOWN_LINES]:
        first, newline, _ = entry.typed.partition("\n")
        cut = len(first) > SHOWN_CHARS or bool(newline)
        lines.append(escape_display(first[:SHOWN_CHARS]) + ("…" if cut else ""))
    if len(entries) > SHOWN_LINES:
        lines.append(f"and {len(entries) - SHOWN_LINES} more")
    return lines
