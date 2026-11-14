from __future__ import annotations

from src.agent.terminal.buffer import LineBuffer


def _fed(*chunks: bytes, **options: int) -> LineBuffer:
    buffer = LineBuffer(**options)
    for chunk in chunks:
        buffer.feed(chunk)
    return buffer


def test_an_empty_buffer_has_no_lines() -> None:
    assert LineBuffer().tail(5) == []


def test_complete_lines_are_returned_in_order() -> None:
    assert _fed(b"one\ntwo\nthree\n").tail(10) == ["one", "two", "three"]


def test_tail_returns_only_the_last_n_lines() -> None:
    assert _fed(b"1\n2\n3\n4\n5\n6\n").tail(2) == ["5", "6"]


def test_tail_includes_the_partial_line_within_the_count() -> None:
    buffer = _fed(b"1\n2\n3\npart")

    assert buffer.tail(10) == ["1", "2", "3", "part"]
    assert buffer.tail(2) == ["3", "part"]


def test_an_empty_partial_line_is_not_returned() -> None:
    assert _fed(b"a\n").tail(3) == ["a"]


def test_tail_of_zero_or_less_is_empty() -> None:
    assert _fed(b"a\nb\n").tail(0) == []
    assert _fed(b"a\nb\n").tail(-1) == []


def test_a_line_split_across_feeds_is_joined() -> None:
    assert _fed(b"hel", b"lo wor", b"ld\n").tail(5) == ["hello world"]


def test_sgr_colors_are_removed() -> None:
    assert _fed(b"\x1b[31mred\x1b[0m plain\n").tail(1) == ["red plain"]


def test_other_escape_sequences_are_removed() -> None:
    data = b"\x1b[2J\x1b[Hscreen\x1b]0;title\x07 text\x1b[?25l\n"

    assert _fed(data).tail(1) == ["screen text"]


def test_an_escape_sequence_split_across_feeds_is_removed() -> None:
    assert _fed(b"a\x1b[3", b"1mb\x1b", b"[0mc\n").tail(1) == ["abc"]


def test_an_osc_split_across_feeds_is_removed() -> None:
    assert _fed(b"x\x1b]0;ti", b"tle\x07y\n").tail(1) == ["xy"]


def test_a_lone_escape_at_the_end_waits_for_the_next_feed() -> None:
    buffer = _fed(b"a\x1b")

    assert buffer.tail(1) == ["a"]
    buffer.feed(b"[0mb\n")
    assert buffer.tail(1) == ["ab"]


def test_other_control_characters_are_dropped() -> None:
    assert _fed(b"a\x07b\x00c\x7fd\te\n").tail(1) == ["abcd\te"]


def test_crlf_is_one_line_break() -> None:
    assert _fed(b"one\r\ntwo\r\n").tail(5) == ["one", "two"]


def test_a_crlf_split_across_feeds_is_one_line_break() -> None:
    assert _fed(b"one\r", b"\ntwo\n").tail(5) == ["one", "two"]


def test_a_lone_carriage_return_keeps_only_the_text_after_it() -> None:
    assert _fed(b"10%\r50%\r100%\ndone\n").tail(5) == ["100%", "done"]


def test_a_carriage_return_at_the_end_of_a_feed_redraws_the_line() -> None:
    buffer = _fed(b"loading 1\r", b"loading 2\r", b"loading 3\n")

    assert buffer.tail(5) == ["loading 3"]


def test_a_utf8_character_split_across_feeds_is_decoded() -> None:
    data = "héllo ✓\n".encode()
    cut = data.index("✓".encode()) + 1

    assert _fed(data[:cut], data[cut:]).tail(1) == ["héllo ✓"]


def test_invalid_utf8_becomes_replacement_characters() -> None:
    assert _fed(b"ok \xff\xfe end\n").tail(1) == ["ok �� end"]


def test_the_oldest_lines_are_dropped_at_the_line_count_cap() -> None:
    buffer = _fed(b"".join(b"line %d\n" % i for i in range(50)), max_lines=10)

    assert len(buffer) == 10
    assert buffer.tail(100) == [f"line {i}" for i in range(40, 50)]


def test_a_long_line_is_cut_at_the_byte_cap() -> None:
    buffer = _fed(b"x" * 10_000 + b"\nshort\n", max_line_bytes=100)

    assert buffer.tail(2) == ["x" * 100, "short"]


def test_a_long_line_of_wide_characters_is_cut_at_the_byte_cap_on_a_character() -> None:
    buffer = _fed(("é" * 500 + "\n").encode(), max_line_bytes=101)  # 2 bytes each

    (line,) = buffer.tail(1)
    assert line == "é" * 50


def test_a_partial_line_without_a_newline_stays_bounded() -> None:
    buffer = LineBuffer(max_line_bytes=64)
    for _ in range(1_000):
        buffer.feed(b"y" * 1_000)

    assert buffer.tail(1) == ["y" * 64]


def test_a_large_stream_stays_bounded_and_tail_is_correct() -> None:
    buffer = LineBuffer(max_lines=1_000)
    total = 50_000
    for start in range(0, total, 500):
        chunk = b"".join(b"\x1b[32mline %d\x1b[0m\r\n" % i for i in range(start, start + 500))
        buffer.feed(chunk)

    assert len(buffer) <= 1_000
    assert buffer.tail(5) == [f"line {i}" for i in range(total - 5, total)]


def test_an_empty_feed_does_not_end_a_carriage_return() -> None:
    assert _fed(b"abc\r", b"", b"\n", b"d\n").tail(5) == ["abc", "d"]


def test_a_feed_holding_only_part_of_an_escape_does_not_end_a_carriage_return() -> None:
    assert _fed(b"abc\r", b"\x1b[", b"0m\ndef\n").tail(5) == ["abc", "def"]


def test_escapes_that_are_not_csi_do_not_leak_into_the_text() -> None:
    assert _fed(b"\x1b(B\x1b[m done\x1b=\x1b>\x1b[>1u\n").tail(1) == [" done"]


def test_a_charset_designator_split_across_feeds_is_removed() -> None:
    assert _fed(b"a\x1b(", b"Bb\n").tail(1) == ["ab"]


def test_a_carriage_return_survives_a_feed_that_holds_part_of_an_escape() -> None:
    assert _fed(b"abc\r", b"\x1b[", b"Kdef\n").tail(5) == ["def"]


def test_a_carriage_return_survives_an_empty_feed() -> None:
    assert _fed(b"abc\r", b"", b"def\n").tail(5) == ["def"]
