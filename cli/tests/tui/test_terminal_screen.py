"""The emulator screen: what pyte lacks (the alternate screen) and the modes the pane needs to read."""

from __future__ import annotations

import pyte
from src.tui.terminal_screen import TerminalScreen, TerminalStream


def feed(screen: pyte.Screen, text: str) -> None:
    pyte.ByteStream(screen).feed(text.encode())


def rows(screen: pyte.Screen) -> list[str]:
    return [line.rstrip() for line in screen.display]


def test_the_alternate_screen_starts_blank_and_gives_back_the_main_screen() -> None:
    screen = TerminalScreen(20, 4)
    feed(screen, "main text\r\nsecond")

    feed(screen, "\x1b[?1049h")
    assert rows(screen) == ["", "", "", ""]
    feed(screen, "\x1b[Halt text")
    assert rows(screen)[0] == "alt text"

    feed(screen, "\x1b[?1049l")
    assert rows(screen) == ["main text", "second", "", ""]


def test_leaving_the_alternate_screen_puts_the_cursor_back() -> None:
    screen = TerminalScreen(20, 4)
    feed(screen, "ab\r\ncd")
    before = (screen.cursor.x, screen.cursor.y)

    feed(screen, "\x1b[?1049h\x1b[3;7Hzz\x1b[?1049l")

    assert (screen.cursor.x, screen.cursor.y) == before


def test_the_older_alternate_screen_modes_work_too() -> None:
    for mode in (47, 1047):
        screen = TerminalScreen(20, 3)
        feed(screen, "keep")

        feed(screen, f"\x1b[?{mode}hother\x1b[?{mode}l")

        assert rows(screen)[0] == "keep"


def test_entering_the_alternate_screen_twice_keeps_the_first_main_screen() -> None:
    screen = TerminalScreen(20, 3)
    feed(screen, "keep\x1b[?1049h\x1b[?1049hx\x1b[?1049l")

    assert rows(screen)[0] == "keep"


def test_leaving_when_not_in_the_alternate_screen_changes_nothing() -> None:
    screen = TerminalScreen(20, 3)
    feed(screen, "keep\x1b[?1049l")

    assert rows(screen)[0] == "keep"


def test_a_resize_in_the_alternate_screen_also_resizes_the_main_screen() -> None:
    screen = TerminalScreen(20, 4)
    feed(screen, "keep")
    feed(screen, "\x1b[?1049h")

    screen.resize(6, 12)
    assert (screen.lines, screen.columns) == (6, 12)
    feed(screen, "\x1b[?1049l")

    assert len(screen.display) == 6
    assert all(len(line) == 12 for line in screen.display)
    assert rows(screen)[0] == "keep"


def test_a_full_reset_leaves_the_alternate_screen() -> None:
    screen = TerminalScreen(20, 3)
    feed(screen, "keep\x1b[?1049hother")

    feed(screen, "\x1bc")

    assert rows(screen) == ["", "", ""]
    feed(screen, "new\x1b[?1049l")
    assert rows(screen)[0] == "new"


def test_application_cursor_and_bracketed_paste_modes_are_readable() -> None:
    screen = TerminalScreen(20, 3)
    assert (screen.application_cursor, screen.bracketed_paste) == (False, False)

    feed(screen, "\x1b[?1h\x1b[?2004h")
    assert (screen.application_cursor, screen.bracketed_paste) == (True, True)

    feed(screen, "\x1b[?1l\x1b[?2004l")
    assert (screen.application_cursor, screen.bracketed_paste) == (False, False)


def test_answers_to_the_program_go_to_the_reply_function() -> None:
    screen = TerminalScreen(20, 5)
    replies: list[str] = []
    screen.reply = replies.append

    feed(screen, "ab\r\n\x1b[6n")  # where is the cursor?

    assert replies == ["\x1b[2;1R"]


def test_without_a_reply_function_questions_are_ignored() -> None:
    screen = TerminalScreen(20, 5)

    feed(screen, "\x1b[6n")  # must not raise


def numbered(lines: int, columns: int = 20) -> TerminalScreen:
    """A screen showing `row 0` ... on every row, the cursor at the end of the last row."""
    screen = TerminalScreen(columns, lines)
    feed(screen, "\r\n".join(f"row {i}" for i in range(lines)))
    return screen


def test_shrinking_with_the_cursor_on_the_last_row_drops_rows_from_the_top() -> None:
    screen = numbered(5)
    assert screen.cursor.y == 4

    screen.resize(3, 20)

    assert rows(screen) == ["row 2", "row 3", "row 4"]
    assert (screen.cursor.x, screen.cursor.y) == (5, 2)  # still after the text it printed last


def test_shrinking_with_the_cursor_near_the_top_trims_the_bottom() -> None:
    screen = numbered(5)
    feed(screen, "\x1b[2;3H")  # row 2 (index 1), column 3

    screen.resize(3, 20)

    assert rows(screen) == ["row 0", "row 1", "row 2"]
    assert (screen.cursor.x, screen.cursor.y) == (2, 1)


def test_shrinking_drops_only_as_many_rows_as_the_cursor_needs() -> None:
    screen = numbered(6)
    feed(screen, "\x1b[5;1H")  # index 4

    screen.resize(3, 20)

    assert rows(screen) == ["row 2", "row 3", "row 4"]
    assert screen.cursor.y == 2


def test_the_cursor_stays_inside_the_screen_after_shrinking() -> None:
    screen = numbered(5)

    screen.resize(2, 4)

    assert 0 <= screen.cursor.y < 2
    assert 0 <= screen.cursor.x < 4
    assert rows(screen) == ["row", "row"]


def test_shrinking_in_the_alternate_screen_keeps_the_content_by_the_same_rule() -> None:
    screen = TerminalScreen(20, 5)
    feed(screen, "main")
    feed(screen, "\x1b[?1049h" + "\r\n".join(f"alt {i}" for i in range(5)))

    screen.resize(3, 20)

    assert rows(screen) == ["alt 2", "alt 3", "alt 4"]
    assert (screen.cursor.x, screen.cursor.y) == (5, 2)
    feed(screen, "\x1b[?1049l")
    assert rows(screen)[0] == "main"
    assert len(screen.display) == 3
    assert screen.cursor.y == 0  # the main screen's cursor was in its first row


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


def test_the_132_column_mode_does_not_resize_the_screen() -> None:
    screen = TerminalScreen(40, 5)
    feed(screen, "keep")

    feed(screen, "\x1b[?3h")
    assert (screen.lines, screen.columns) == (5, 40)
    feed(screen, "\x1b[?3l")

    assert (screen.lines, screen.columns) == (5, 40)
    assert rows(screen)[0] == "keep"  # and does not clear it


def test_other_private_modes_set_together_with_132_columns_still_apply() -> None:
    screen = TerminalScreen(40, 5)

    feed(screen, "\x1b[?3;1;2004h")

    assert (screen.application_cursor, screen.bracketed_paste) == (True, True)
    assert screen.columns == 40


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


def test_a_sequence_pyte_cannot_handle_is_skipped_and_the_rest_of_the_chunk_is_drawn() -> None:
    screen = TerminalScreen(20, 3)

    stream_of(screen).feed(b"before\x1b[;5Cafter\r\nnext")  # ESC [ ; 5 C makes pyte's handler raise a TypeError

    assert rows(screen)[0].startswith("before")
    assert "after" in rows(screen)[0]
    assert rows(screen)[1] == "next"


def test_a_cursor_position_request_in_origin_mode_is_skipped_and_the_rest_is_drawn() -> None:
    screen = TerminalScreen(20, 3)
    screen.reply = lambda _: None

    stream_of(screen).feed(b"\x1b[?6h\x1b[6nok")  # pyte asserts on it

    assert rows(screen)[0] == "ok"


def test_the_stream_keeps_working_after_a_bad_sequence_in_another_chunk() -> None:
    screen = TerminalScreen(20, 3)
    stream = stream_of(screen)

    stream.feed(b"a\x1b[;5C")
    stream.feed(b"b")

    assert rows(screen)[0].startswith("ab")
