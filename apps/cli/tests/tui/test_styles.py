"""The shared styles build `Text` from plain strings: styled, aligned and never parsed as markup."""

from __future__ import annotations

from rich.style import Style
from rich.text import Text
from src.tui.styles import (
    ansi_output,
)


def _styled(text: Text) -> list[tuple[str, Style | str]]:
    """The plain pieces of `text` with the style each carries."""
    return [(text.plain[s.start : s.end], s.style) for s in text.spans]


def test_ansi_output_reads_colon_form_colors() -> None:
    text = ansi_output("\x1b[38:2::255:0:0mred\x1b[0m \x1b[48:5:21mblue\x1b[0m")
    assert text.plain == "red blue"
    assert text.spans[0].style == Style.parse("#ff0000")
    assert text.spans[1].style == Style.parse("on color(21)")
