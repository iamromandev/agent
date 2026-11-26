"""Draw a pyte screen as Textual strips: one `Strip` per row, one segment per run of equal style."""

from __future__ import annotations

from pyte.screens import Char, Screen
from rich.cells import cell_len, set_cell_size
from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip

from src.tui.styles import terminal_style


def _cell_style(char: Char, cursor: bool) -> Style:
    return terminal_style(
        char.fg,
        char.bg,
        char.bold,
        char.italics,
        char.underscore,
        char.strikethrough,
        char.reverse != cursor,  # the cursor is the cell drawn the other way round
    )


def line_strip(screen: Screen, y: int, *, cursor: bool) -> Strip:
    """Row `y` of `screen`; `cursor` draws the cursor (when it is on this row and not hidden)."""
    if not 0 <= y < screen.lines:
        return Strip([Segment(" " * screen.columns)], screen.columns)
    line = screen.buffer[y]
    at = screen.cursor.x if cursor and not screen.cursor.hidden and screen.cursor.y == y else -1
    segments: list[Segment] = []
    run: list[str] = []
    run_style: Style | None = None
    for x in range(screen.columns):
        char = line[x]
        if not char.data:  # the second half of a wide character, already emitted with the first
            continue
        covered = x == at or (at == x + 1 and not line[at].data)  # the cursor on the placeholder covers the wide one
        style = _cell_style(char, covered)
        if style != run_style:
            if run and run_style is not None:
                segments.append(Segment("".join(run), run_style))
            run, run_style = [], style
        data = char.data
        width = 2 if x + 1 < screen.columns and not line[x + 1].data else 1  # what the emulator gave the cell
        if not data.isascii() and cell_len(data) != width:  # Rich and the emulator disagree: keep the columns
            data = set_cell_size(data, width)
        run.append(data)
    if run and run_style is not None:
        segments.append(Segment("".join(run), run_style))
    return Strip(segments, screen.columns)


def screen_strips(screen: Screen, *, cursor: bool) -> list[Strip]:
    return [line_strip(screen, y, cursor=cursor) for y in range(screen.lines)]
