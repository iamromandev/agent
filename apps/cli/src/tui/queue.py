"""Messages typed while a turn runs: the queue, how it merges into one turn, and the panel's lines (UI-free)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from core.engine.text import escape_display

SHOWN_LINES = 5
SHOWN_CHARS = 80


@dataclass(frozen=True)
class Queued:
    typed: str  # what the user typed: the message, or "/name args"
    prompt: str  # what is sent: the message itself, or the command's expanded prompt
    label: str | None  # "/name (source)" for a custom command, else None


@dataclass(frozen=True)
class Merged:
    prompt: str  # the entries' prompts joined by a blank line
    draft: str  # the entries' typed text joined by newlines (what /undo and a restore give back)
    label: str | None  # only when there is exactly one entry and it has one


class MessageQueue:
    def __init__(self) -> None:
        self._entries: list[Queued] = []

    def add(self, entry: Queued) -> None:
        self._entries.append(entry)

    def take(self) -> tuple[Queued, ...]:
        taken = tuple(self._entries)
        self._entries.clear()
        return taken

    @property
    def entries(self) -> tuple[Queued, ...]:
        return tuple(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)


def merge(entries: Sequence[Queued]) -> Merged:
    label = entries[0].label if len(entries) == 1 else None
    return Merged("\n\n".join(e.prompt for e in entries), "\n".join(e.typed for e in entries), label)


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
