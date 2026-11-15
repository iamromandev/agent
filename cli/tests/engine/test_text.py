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
        ("‮x", "\\u202ex"),
        ("\U0001f600", "\\U0001f600"),
        ("é", "\\xe9"),
    ],
)
def test_everything_but_printable_ascii_is_written_out(text: str, expected: str) -> None:
    assert escape(text) == expected
