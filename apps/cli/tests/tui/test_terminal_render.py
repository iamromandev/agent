"""Drawing a terminal screen as Textual strips: colors, attributes, the cursor, wide characters."""

from __future__ import annotations

import pyte
from rich.cells import cell_len
from rich.color import Color
from rich.style import Style
from src.tui.terminal_render import line_strip, screen_strips
from src.tui.terminal_screen import TerminalScreen
from textual.strip import Strip


def make(text: str, cols: int = 20, rows: int = 3) -> TerminalScreen:
    screen = TerminalScreen(cols, rows)
    pyte.ByteStream(screen).feed(text.encode())
    return screen


def strips(text: str, *, cursor: bool = False, cols: int = 20, rows: int = 3) -> list[Strip]:
    return screen_strips(make(text, cols, rows), cursor=cursor)


def plain(strip: Strip) -> str:
    return "".join(segment.text for segment in strip)


def styled(strip: Strip, text: str) -> Style:
    """The style of the first segment that holds `text`."""
    style = next(segment.style for segment in strip if text in segment.text)
    assert style is not None
    return style


def reversed_segments(strip: Strip) -> list[str]:
    """The text of the segments drawn reversed."""
    return [segment.text for segment in strip if segment.style is not None and segment.style.reverse]


def test_ansi_names_map_to_the_terminal_palette() -> None:
    first = strips("\x1b[33ma\x1b[94mb\x1b[42mc")[0]

    assert styled(first, "a").color == Color.parse("yellow")  # pyte calls it brown
    assert styled(first, "b").color == Color.parse("bright_blue")
    assert styled(first, "c").bgcolor == Color.parse("green")


def test_256_and_true_colors_are_rgb() -> None:
    first = strips("\x1b[38;2;1;2;3ma\x1b[48;5;196mb")[0]

    assert styled(first, "a").color == Color.parse("#010203")
    assert styled(first, "b").bgcolor == Color.parse("#ff0000")


def test_a_row_outside_the_screen_is_blank() -> None:
    screen = make("x", cols=5, rows=2)

    assert plain(line_strip(screen, 7, cursor=False)) == " " * 5


def test_a_character_wider_for_the_emulator_than_for_rich_takes_the_emulators_width() -> None:
    # wcwidth says 2 cells (pyte), Rich says 1: the cells after it must not shift left
    wide = "\U00018cd6"
    assert cell_len(wide) == 1
    strip = strips(f"a{wide}b")[0]

    assert strip.cell_length == 20
    assert cell_len(plain(strip).split("b")[0]) == 3
    assert plain(strip).startswith("a" + wide + " b")
