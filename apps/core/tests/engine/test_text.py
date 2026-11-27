"""`escape`: text from a workspace file is shown with everything but printable ASCII written out."""

from __future__ import annotations

import pytest
from core.engine.text import escape


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("\U0001f600", "\\U0001f600"),
    ],
)
def test_everything_but_printable_ascii_is_written_out(text: str, expected: str) -> None:
    assert escape(text) == expected


def test_one_line_collapses_whitespace_and_cuts_at_the_limit_keeping_escapes_whole() -> None:
    from core.engine.text import one_line

    assert one_line("a\n  b\tc", 50) == "a b c"
    assert one_line("abcdefghij", 5) == "abcd…"
    assert one_line("ab‮cd", 5) == "ab…"  # the escape ‮ (6 chars) never gets cut in half
