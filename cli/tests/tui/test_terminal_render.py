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


def test_there_is_one_strip_per_row_as_wide_as_the_screen() -> None:
    result = strips("hi")

    assert len(result) == 3
    assert [strip.cell_length for strip in result] == [20, 20, 20]
    assert plain(result[0]).rstrip() == "hi"


def test_red_text_is_red_and_the_rest_has_no_color() -> None:
    first = strips("\x1b[31mred\x1b[0m plain")[0]

    assert styled(first, "red").color == Color.parse("red")
    assert styled(first, "plain").color is None
    assert styled(first, "plain").bgcolor is None


def test_equal_neighbours_are_one_segment() -> None:
    first = strips("abc")[0]

    assert next(segment.text for segment in first).startswith("abc")
    assert len(list(first)) == 1


def test_ansi_names_map_to_the_terminal_palette() -> None:
    first = strips("\x1b[33ma\x1b[94mb\x1b[42mc")[0]

    assert styled(first, "a").color == Color.parse("yellow")  # pyte calls it brown
    assert styled(first, "b").color == Color.parse("bright_blue")
    assert styled(first, "c").bgcolor == Color.parse("green")


def test_256_and_true_colors_are_rgb() -> None:
    first = strips("\x1b[38;2;1;2;3ma\x1b[48;5;196mb")[0]

    assert styled(first, "a").color == Color.parse("#010203")
    assert styled(first, "b").bgcolor == Color.parse("#ff0000")


def test_bold_italic_underline_strike_and_reverse() -> None:
    first = strips("\x1b[1mB\x1b[0m\x1b[3mI\x1b[0m\x1b[4mU\x1b[0m\x1b[9mS\x1b[0m\x1b[7mR")[0]

    assert styled(first, "B").bold
    assert styled(first, "I").italic
    assert styled(first, "U").underline
    assert styled(first, "S").strike
    assert styled(first, "R").reverse


def test_the_cursor_is_drawn_reversed_when_asked() -> None:
    with_cursor = strips("ab", cursor=True)[0]
    without = strips("ab", cursor=False)[0]

    assert styled(with_cursor, " ").reverse  # the cell after "ab"
    assert reversed_segments(without) == []
    assert plain(with_cursor) == plain(without)


def test_the_cursor_on_a_reversed_cell_flips_it_back() -> None:
    screen = make("\x1b[7mab\x1b[1D")  # the cursor is on the "b", which is already reversed

    strip = line_strip(screen, 0, cursor=True)

    assert not styled(strip, "b").reverse
    assert styled(strip, "a").reverse


def test_a_hidden_cursor_is_not_drawn() -> None:
    result = strips("ab\x1b[?25l", cursor=True)[0]

    assert reversed_segments(result) == []


def test_the_cursor_is_only_on_its_own_row() -> None:
    result = strips("a\r\nb", cursor=True)

    assert reversed_segments(result[0]) == []
    assert reversed_segments(result[1]) != []


def test_a_wide_character_is_emitted_once_and_its_placeholder_skipped() -> None:
    first = strips("a中b")[0]

    text = plain(first)
    assert text.startswith("a中b")
    assert text.count("中") == 1
    assert cell_len(text) == 20  # 19 characters, one of them two cells wide
    assert first.cell_length == 20


def test_the_cursor_on_a_wide_character_covers_it() -> None:
    screen = make("a中b\x1b[1;2H")

    strip = line_strip(screen, 0, cursor=True)

    assert styled(strip, "中").reverse
    assert strip.cell_length == 20


def test_the_alternate_screen_is_drawn_and_then_the_main_one_again() -> None:
    screen = make("main\x1b[?1049h\x1b[Halt")
    assert plain(line_strip(screen, 0, cursor=False)).startswith("alt")

    pyte.ByteStream(screen).feed(b"\x1b[?1049l")

    assert plain(line_strip(screen, 0, cursor=False)).startswith("main")


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


def test_a_cell_wider_for_rich_than_for_the_emulator_is_clipped_to_the_emulators_width() -> None:
    screen = make("ab")
    screen.buffer[0][0] = screen.buffer[0][0]._replace(data="a\u0b53")  # one cell for pyte, two for Rich
    assert cell_len("a\u0b53") == 2

    strip = line_strip(screen, 0, cursor=False)

    assert strip.cell_length == 20
    assert plain(strip).startswith("ab")


def test_a_zero_width_joiner_or_emoji_selector_does_not_swallow_the_text_after_it() -> None:
    screen = make("a\u2764\ufe0fb\U0001f468\u200d\U0001f469c")

    text = screen.display[0]
    assert text.startswith("a\u2764b")
    assert text.rstrip().endswith("c")
