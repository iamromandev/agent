"""Every style of the TUI, and helpers that build styled `Text` from plain strings.

Text from the model, files, commands or the user is only ever appended as plain text: it is never
parsed as markup. Colors are ANSI names, so they follow the terminal palette in light and dark themes.
"""

from __future__ import annotations

import re
from typing import Literal

from rich.cells import cell_len, set_cell_size
from rich.style import Style
from rich.text import Text

TOOL_NAME = Style(bold=True)
ACCENT = Style(color="cyan")
DIM = Style(dim=True)
OK = Style(color="green")
ERROR = Style(color="red")
NOTICE = {
    "info": Style(dim=True),
    "warn": Style(color="yellow"),
    "error": Style(color="red", bold=True),
}
ICONS = {"info": "ℹ", "warn": "⚠", "error": "✗"}  # noqa: RUF001 - the info sign is the intended glyph
MARK_OK, MARK_ERROR, MARK_ENDED = "✓", "✗", "⏹"

Level = Literal["info", "warn", "error"]
Outcome = Literal["ok", "error", "ended"] | None
_MARKS = {"ok": (MARK_OK, OK), "error": (MARK_ERROR, ERROR), "ended": (MARK_ENDED, DIM)}


def tool_title(label: str, summary: str, outcome: Outcome) -> Text:
    """`mark name argument · summary`; the mark only once the outcome is known."""
    text = Text()
    if outcome is not None:
        mark, style = _MARKS[outcome]
        text.append(mark, style)
        text.append(" ")
    name, space, rest = label.partition(" ")
    text.append(name, TOOL_NAME)
    if space:
        text.append(" ")
        text.append(rest, ACCENT)
    if summary:
        text.append(" · ", DIM)
        text.append(summary, DIM)
    return text


def notice_text(level: Level, text: str, icon: str | None = None) -> Text:
    return Text(f"{icon or ICONS[level]} {text}", style=NOTICE[level])


def session_row(title: str, when: str, count: int, title_width: int) -> Text:
    one_line = " ".join(title.split("\n")).replace("\r", "")
    shown = one_line if cell_len(one_line) <= title_width else set_cell_size(one_line, title_width - 1) + "…"
    row = Text(shown + " " * (title_width - cell_len(shown)) + "  ")
    row.append(when, DIM)
    row.append(f"  {count:>3}")
    return row


_CRLF = re.compile(r"\r+((?:\x1b\[[0-9;:]*m)*)\n")
_COLON = re.compile(r"\x1b\[[0-9;]*[34]8:[0-9;:]*m")
_COLOR = re.compile(r"([34]8):(?:2:(?:\d*:)?(\d*):(\d*):(\d*)|5:(\d*))")


def _semicolons(match: re.Match[str]) -> str:
    def color(m: re.Match[str]) -> str:
        kind, r, g, b, n = m.groups()
        return f"{kind};5;{n}" if n is not None else f"{kind};2;{r};{g};{b}"

    return _COLOR.sub(color, match.group())


def ansi_output(display: str) -> Text:
    """Command output with its colors as styles (`Text.from_ansi` never reads markup).

    A `\\r` before a newline is dropped first (CRLF, also with colors in between), as Rich would blank
    the line. Colon-form colors (`38:2::r:g:b`, `48:5:n`), which Rich ignores, become semicolon form.
    Rich keeps only the last `\\r` segment of a line (a progress bar redrawn in place shows its final
    state), while the model's copy keeps every segment.
    """
    return Text.from_ansi(_COLON.sub(_semicolons, _CRLF.sub(r"\1\n", display)))


def diff_theme(dark: bool) -> str:
    return "ansi_dark" if dark else "ansi_light"
