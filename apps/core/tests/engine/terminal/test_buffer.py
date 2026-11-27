from __future__ import annotations

from core.engine.terminal.buffer import LineBuffer


def _fed(*chunks: bytes, **options: int) -> LineBuffer:
    buffer = LineBuffer(**options)
    for chunk in chunks:
        buffer.feed(chunk)
    return buffer


def test_tail_of_zero_or_less_is_empty() -> None:
    assert _fed(b"a\nb\n").tail(0) == []
    assert _fed(b"a\nb\n").tail(-1) == []


def test_an_osc_split_across_feeds_is_removed() -> None:
    assert _fed(b"x\x1b]0;ti", b"tle\x07y\n").tail(1) == ["xy"]


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
