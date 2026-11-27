from __future__ import annotations

import pytest
from src.tui.widgets.chat_input import PLACEHOLDER, PROMPT, ChatInput, InputRow
from src.tui.widgets.terminal import TerminalView
from textual.color import Color
from textual.widget import Widget

from tests.tui.conftest import AppFactory, scripted, settled, until

BOX_CHARS = "│╭╮╰╯┌┐└┘▔▁▎▊"


def _rows(widget: Widget) -> list[str]:
    """The screen rows the widget covers, children included, as the compositor draws them."""
    strips = widget.screen._compositor.render_strips()
    box = widget.region  # includes the border
    return [strips[y].text[box.x : box.right] for y in range(box.y, box.bottom)]


def _fg_at(widget: Widget, row: int, col: int) -> Color:
    """The foreground colour of the cell at (row, col) inside the widget, as rendered."""
    strip = widget.screen._compositor.render_strips()[widget.region.y + row]
    x = widget.region.x + col
    pos = 0
    for seg in strip:
        if pos <= x < pos + seg.cell_length:
            assert seg.style is not None and seg.style.color is not None
            return Color.from_rich_color(seg.style.color)
        pos += seg.cell_length
    raise AssertionError("cell outside the strip")


def _luminance(color: Color) -> float:
    return color.brightness


@pytest.mark.parametrize("width", [40, 80, 120])
async def test_lines_above_and_below_fill_the_width_without_a_box(make_app: AppFactory, width: int) -> None:
    app = make_app()
    async with app.run_test(size=(width, 24)) as pilot:
        await pilot.pause()
        rows = _rows(app.query_one("#input-row", InputRow))

    assert rows[0] == "─" * width
    assert rows[-1] == "─" * width
    assert not any(c in row for row in rows for c in BOX_CHARS)


async def test_lines_are_accent_with_focus_and_panel_without(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        row = app.query_one("#input-row", InputRow)
        css = app.get_css_variables()
        accent, panel = Color.parse(css["accent"]), Color.parse(css["panel"])
        assert row.styles.border_top[1] == accent
        assert row.styles.border_bottom[1] == accent

        app.set_focus(app.query_one("#transcript"))
        await pilot.pause()
        assert row.styles.border_top[1] == panel
        assert row.styles.border_bottom[1] == panel

        app.query_one(ChatInput).focus()
        await pilot.pause()
        assert row.styles.border_top[1] == accent


async def test_lines_follow_focus_between_the_terminal_pane_and_the_input(make_app: AppFactory) -> None:
    app = make_app(terminals=True)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        row = app.query_one("#input-row", InputRow)
        css = app.get_css_variables()
        accent, panel = Color.parse(css["accent"]), Color.parse(css["panel"])

        await pilot.press("ctrl+t")
        await until(pilot, lambda: isinstance(app.focused, TerminalView))
        await pilot.pause()
        assert row.styles.border_top[1] == panel
        assert row.styles.border_bottom[1] == panel

        await pilot.press("ctrl+backslash")
        await pilot.pause()
        assert isinstance(app.focused, ChatInput)
        assert row.styles.border_top[1] == accent
        assert row.styles.border_bottom[1] == accent


async def test_empty_input_shows_marker_and_placeholder_and_typed_text_submits_alone(make_app: AppFactory) -> None:
    from core.engine.provider.base import Done, TextDelta

    app = make_app([[TextDelta("ok"), Done(1, 1)]])
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        row = app.query_one("#input-row", InputRow)
        assert _rows(row)[1].startswith(PROMPT + PLACEHOLDER)
        hint_fg = _fg_at(row, 1, len(PROMPT) + 3)  # past the cursor cell

        await pilot.press("h")
        typed_fg = _fg_at(row, 1, len(PROMPT))  # the typed "h"
        assert hint_fg != typed_fg
        assert _luminance(hint_fg) < _luminance(typed_fg)
        await pilot.press("backspace")

        await pilot.press("h", "i")
        assert app.query_one(ChatInput).text == "hi"
        await pilot.press("enter")
        await settled(app)

    sent = scripted(app).requests[0][0]
    assert sent[-1].content == "hi"


async def test_second_line_starts_under_the_text_without_a_marker(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        chat = app.query_one(ChatInput)
        chat.text = "a\nb"
        await pilot.pause()
        rows = _rows(app.query_one("#input-row", InputRow))
        prompt = app.query_one("#prompt")

        assert rows[1][2] == "a"
        assert rows[2][:2] == "  "
        assert rows[2][2] == "b"
        assert PROMPT.strip() not in rows[2]
        assert prompt.region.right == chat.region.x


async def test_a_single_space_hides_the_placeholder(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one(ChatInput).text = " "
        await pilot.pause()
        rows = _rows(app.query_one("#input-row", InputRow))

    assert "ask anything" not in rows[1]


async def test_the_text_area_grows_to_eight_rows_then_scrolls(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        chat = app.query_one(ChatInput)
        chat.text = "x"
        await pilot.pause()
        assert chat.region.height == 1
        assert app.query_one("#input-row", InputRow).region.height == 3
        chat.text = "\n".join(map(str, range(12)))
        await pilot.pause()
        assert chat.region.height == 8
        assert chat.max_scroll_y > 0
        assert app.query_one("#input-row", InputRow).region.height == 10


async def test_input_is_still_found_by_id_inside_its_row(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert isinstance(app.query_one("#input", ChatInput), ChatInput)
        assert isinstance(app.query_one("#input-row"), InputRow)


async def test_the_text_area_has_no_background_of_its_own(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        chat = app.query_one(ChatInput)
        assert chat.styles.background.a == 0
        assert chat.get_component_styles("text-area--cursor-line").background.a == 0
