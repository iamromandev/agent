"""The shared styles build `Text` from plain strings: styled, aligned and never parsed as markup."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import pytest
from rich.style import Style
from rich.text import Text
from src.tui import styles
from src.tui.styles import (
    ACCENT,
    DIM,
    ERROR,
    ICONS,
    NOTICE,
    OK,
    TOOL_NAME,
    Level,
    Outcome,
    ansi_output,
    diff_theme,
    notice_text,
    session_row,
    tool_title,
)


def _styled(text: Text) -> list[tuple[str, Style | str]]:
    """The plain pieces of `text` with the style each carries."""
    return [(text.plain[s.start : s.end], s.style) for s in text.spans]


@pytest.mark.parametrize(
    ("outcome", "mark", "style"),
    [("ok", "✓", OK), ("error", "✗", ERROR), ("ended", "⏹", DIM)],
)
def test_a_title_starts_with_a_mark_for_its_outcome(outcome: str, mark: str, style: Style) -> None:
    title = tool_title("edit a.py", "1 replaced", cast("Outcome", outcome))
    assert title.plain.startswith(f"{mark} edit a.py")
    assert _styled(title)[0] == (mark, style)


def test_a_title_has_no_mark_while_the_outcome_is_unknown() -> None:
    title = tool_title("edit a.py", "1 replaced", None)
    assert title.plain == "edit a.py · 1 replaced"


def test_a_title_styles_name_argument_and_summary() -> None:
    title = tool_title("edit src/app.py", "1 replaced", "ok")
    assert title.plain == "✓ edit src/app.py · 1 replaced"
    pieces = _styled(title)
    assert pieces[0] == ("✓", OK)
    assert pieces[1] == ("edit", TOOL_NAME)
    assert pieces[2] == ("src/app.py", ACCENT)
    assert all(style == DIM for _, style in pieces[3:])
    assert "".join(plain for plain, _ in pieces[3:]) == " · 1 replaced"


def test_a_label_without_a_space_is_only_a_name() -> None:
    title = tool_title("ls", "", None)
    assert title.plain == "ls"
    assert _styled(title) == [("ls", TOOL_NAME)]


def test_markup_in_a_label_stays_literal() -> None:
    title = tool_title("read [red]x[/red].txt", "saw [$var]", "ok")
    assert title.plain == "✓ read [red]x[/red].txt · saw [$var]"


@pytest.mark.parametrize("level", ["info", "warn", "error"])
def test_a_notice_has_its_icon_and_style(level: str) -> None:
    text = notice_text(cast("Level", level), "hello")
    assert text.plain == f"{ICONS[level]} hello"
    assert text.style == NOTICE[level]


def test_the_levels_have_distinct_styles_and_icons() -> None:
    assert len({str(s) for s in NOTICE.values()}) == 3
    assert len(set(ICONS.values())) == 3


def test_a_notice_icon_can_be_overridden() -> None:
    assert notice_text("info", "interrupted", "⏹").plain == "⏹ interrupted"


def test_a_notice_keeps_markup_literal() -> None:
    assert notice_text("warn", "[red]x[/red]").plain == "⚠ [red]x[/red]"


def test_a_session_row_pads_the_title_and_right_aligns_the_count() -> None:
    row = session_row("Fix", "2h ago", 12, 10)
    assert row.plain == "Fix         2h ago   12"
    assert any(plain == "2h ago" and style == DIM for plain, style in _styled(row))


def test_a_session_row_cuts_a_long_title_with_an_ellipsis() -> None:
    row = session_row("x" * 50, "now", 1, 10)
    assert row.plain.startswith("x" * 9 + "…  now")
    assert len(row.plain.split("  ")[0]) == 10


def test_a_session_row_measures_the_title_in_cells_not_characters() -> None:
    from rich.cells import cell_len

    row = session_row("日本語のタイトルです", "now", 1, 10)
    title = row.plain.split("  now")[0]
    assert cell_len(title) == 10
    assert title.endswith("…") or title.endswith("… ")
    assert row.plain.endswith("now    1")
    short = session_row("日本", "now", 1, 10)
    assert cell_len(short.plain.split("  now")[0]) == 10


def test_a_session_row_shows_a_multiline_title_on_one_line() -> None:
    row = session_row("first\nsecond\r\nthird", "now", 1, 30)
    assert "\n" not in row.plain
    assert row.plain.startswith("first second third")


def test_a_session_row_keeps_markup_literal() -> None:
    assert session_row("[red]x[/red]", "now", 1, 20).plain.startswith("[red]x[/red]")


def test_ansi_output_turns_colors_into_styles() -> None:
    text = ansi_output("\x1b[32mok\x1b[0m done")
    assert text.plain == "ok done"
    (span,) = text.spans
    assert text.plain[span.start : span.end] == "ok"
    assert isinstance(span.style, Style)
    assert span.style.color is not None
    assert OK.color is not None
    assert span.style.color.number == OK.color.number


def test_ansi_output_keeps_markup_literal() -> None:
    assert ansi_output("[red]x[/red] [$var]").plain == "[red]x[/red] [$var]"


def test_the_diff_theme_follows_the_terminal_brightness() -> None:
    assert diff_theme(True) == "ansi_dark"
    assert diff_theme(False) == "ansi_light"


def test_no_other_tui_module_defines_colors() -> None:
    tui = Path(styles.__file__).parent
    pattern = re.compile(r"\bStyle\(|\bcolor=|\"(?:red|green|yellow|cyan)\"")
    offenders = [
        str(path.relative_to(tui))
        for path in tui.rglob("*.py")
        if path.name != "styles.py" and pattern.search(path.read_text())
    ]
    assert offenders == []


def test_ansi_output_shows_only_the_last_carriage_return_segment_of_a_line() -> None:
    """Rich keeps the last `\\r` segment of a line on screen; the model copy keeps them all."""
    assert ansi_output("10%\r50%\r100%\ndone").plain == "100%\ndone"


def test_ansi_output_keeps_markup_in_colored_text_literal() -> None:
    text = ansi_output("\x1b[31m[red]x[/red] [$var]\x1b[0m")
    assert text.plain == "[red]x[/red] [$var]"


def test_ansi_output_treats_crlf_as_a_newline() -> None:
    assert ansi_output("a\r\nb").plain == "a\nb"
    assert ansi_output("hello\r\nworld\r\n").plain == "hello\nworld\n"


def test_ansi_output_keeps_text_before_a_carriage_return_that_precedes_a_reset_and_newline() -> None:
    text = ansi_output("\x1b[32mx\r\x1b[m\ny")
    assert text.plain == "x\ny"
    assert text.spans[0].style == Style.parse("color(2)")


def test_ansi_output_redraws_in_place_then_ends_the_line_with_crlf() -> None:
    assert ansi_output("10%\r50%\r100%\r\ndone").plain == "100%\ndone"


def test_ansi_output_reads_colon_form_colors() -> None:
    text = ansi_output("\x1b[38:2::255:0:0mred\x1b[0m \x1b[48:5:21mblue\x1b[0m")
    assert text.plain == "red blue"
    assert text.spans[0].style == Style.parse("#ff0000")
    assert text.spans[1].style == Style.parse("on color(21)")


def test_the_terminal_color_and_style_caches_are_bounded() -> None:
    assert styles.terminal_color.cache_info().maxsize == 1024
    assert styles.terminal_style.cache_info().maxsize == 4096
    for n in range(1100):  # more colors than the cache holds: it evicts instead of growing
        styles.terminal_color(f"{n:06x}")
    assert styles.terminal_color.cache_info().currsize <= 1024
