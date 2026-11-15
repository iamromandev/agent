"""`escape`: text from a workspace file is shown with everything but printable ASCII written out."""

from __future__ import annotations

import pytest
from src.engine.text import escape


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("bash(ls*)", "bash(ls*)"),
        ("a\x1b[2Jb", "a\\x1b[2Jb"),
        ("a\nb\rc", "a\\x0ab\\x0dc"),
        ("\u202ex", "\\u202ex"),
        ("\U0001f600", "\\U0001f600"),
        ("é", "\\xe9"),
    ],
)
def test_everything_but_printable_ascii_is_written_out(text: str, expected: str) -> None:
    assert escape(text) == expected


@pytest.mark.parametrize("text", ["Привет мир", "日本語のテキスト", "naïve café — “quoted” …", "https://пример.рф/путь?q=ü"])
def test_escape_display_keeps_letters_and_punctuation(text: str) -> None:
    from src.engine.text import escape_display

    assert escape_display(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("\U000e0041", "\\U000e0041"),
        ("a\u202eb", "a\\u202eb"),
        ("a\u2028b\u2029", "a\\u2028b\\u2029"),
        ("a\x85b", "a\\x85b"),
        ("\x1b[2J", "\\x1b[2J"),
        ("\u200b\ufeff\u200e\u061c\u2066", "\\u200b\\ufeff\\u200e\\u061c\\u2066"),
        ("\ud800", "\\ud800"),
        ("\ue000", "\\ue000"),
        ("\u0378", "\\u0378"),
    ],
)
def test_escape_display_writes_out_control_and_invisible_characters(text: str, expected: str) -> None:
    from src.engine.text import escape_display

    assert escape_display(text) == expected
