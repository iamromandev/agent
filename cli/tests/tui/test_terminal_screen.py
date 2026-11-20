"""The emulator screen: what pyte lacks (the alternate screen) and the modes the pane needs to read."""

from __future__ import annotations

import pyte
from src.tui.terminal_screen import TerminalScreen, TerminalStream


def feed(screen: pyte.Screen, text: str) -> None:
    pyte.ByteStream(screen).feed(text.encode())


def rows(screen: pyte.Screen) -> list[str]:
    return [line.rstrip() for line in screen.display]


def test_entering_the_alternate_screen_twice_keeps_the_first_main_screen() -> None:
    screen = TerminalScreen(20, 3)
    feed(screen, "keep\x1b[?1049h\x1b[?1049hx\x1b[?1049l")

    assert rows(screen)[0] == "keep"


def test_a_full_reset_leaves_the_alternate_screen() -> None:
    screen = TerminalScreen(20, 3)
    feed(screen, "keep\x1b[?1049hother")

    feed(screen, "\x1bc")

    assert rows(screen) == ["", "", ""]
    feed(screen, "new\x1b[?1049l")
    assert rows(screen)[0] == "new"


def test_without_a_reply_function_questions_are_ignored() -> None:
    screen = TerminalScreen(20, 5)

    feed(screen, "\x1b[6n")  # must not raise


def numbered(lines: int, columns: int = 20) -> TerminalScreen:
    """A screen showing `row 0` ... on every row, the cursor at the end of the last row."""
    screen = TerminalScreen(columns, lines)
    feed(screen, "\r\n".join(f"row {i}" for i in range(lines)))
    return screen


def test_the_main_screen_is_shrunk_by_its_own_cursor_while_in_the_alternate_screen() -> None:
    screen = numbered(5)
    feed(screen, "\x1b[?1049h")

    screen.resize(3, 20)
    feed(screen, "\x1b[?1049l")

    assert rows(screen) == ["row 2", "row 3", "row 4"]
    assert screen.cursor.y == 2


def test_growing_keeps_the_rows_and_adds_blank_ones_at_the_bottom() -> None:
    screen = numbered(3)

    screen.resize(5, 20)

    assert rows(screen) == ["row 0", "row 1", "row 2", "", ""]
    assert screen.cursor.y == 2


def test_a_hidden_cursor_stays_hidden_across_the_alternate_screen_in_both_directions() -> None:
    screen = TerminalScreen(20, 3)

    feed(screen, "\x1b[?25l\x1b[?1049h")
    assert screen.cursor.hidden
    feed(screen, "\x1b[?25h\x1b[?1049l")  # shown while in the alternate screen
    assert not screen.cursor.hidden

    feed(screen, "\x1b[?1049h\x1b[?25l\x1b[?1049l")  # hidden in the alternate screen
    assert screen.cursor.hidden


def stream_of(screen: pyte.Screen) -> pyte.ByteStream:
    return TerminalStream(screen)


def test_the_stream_keeps_working_after_a_bad_sequence_in_another_chunk() -> None:
    screen = TerminalScreen(20, 3)
    stream = stream_of(screen)

    stream.feed(b"a\x1b[;5C")
    stream.feed(b"b")

    assert rows(screen)[0].startswith("ab")
