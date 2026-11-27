from __future__ import annotations

import pytest
from core.engine.todo import TodoItem
from core.host.queue import Queued
from src.tui.widgets.attachment_panel import AttachmentPanel
from src.tui.widgets.chat_input import InputRow
from src.tui.widgets.command_suggestions import CommandSuggestions
from src.tui.widgets.path_suggestions import PathSuggestions
from src.tui.widgets.queue_panel import QueuePanel
from src.tui.widgets.todo_panel import TodoPanel
from textual.color import Color
from textual.widget import Widget

from tests.tui.conftest import AppFactory

SIZES = [40, 80]
CHIP = "clipboard (PNG, 214 KB)"


def _rows(widget: Widget) -> list[str]:
    strips = widget.screen._compositor.render_strips()
    box = widget.region
    return [strips[y].text[box.x : box.right] for y in range(box.y, box.bottom)]


def _todos(done: int, total: int) -> list[TodoItem]:
    return [TodoItem(f"item {n}", "done" if n < done else "pending") for n in range(total)]


def _q(text: str) -> Queued:
    return Queued(text, text, None)


def _check_top(rows: list[str], label: str, width: int) -> None:
    prefix = f"─ {label} "
    assert rows[0].startswith(prefix)
    assert rows[0] == prefix + "─" * (width - len(prefix))
    assert not any("│" in r for r in rows)


@pytest.mark.parametrize("width", SIZES)
async def test_todo_panel_top_row_has_label_and_fills_the_width(make_app: AppFactory, width: int) -> None:
    app = make_app()
    async with app.run_test(size=(width, 24)) as pilot:
        panel = app.query_one(TodoPanel)
        panel.show(_todos(2, 5))
        await pilot.pause()
        rows = _rows(panel)
    _check_top(rows, "todos 2/5", width)
    assert len(rows) == 6  # top line plus five items, no bottom border


@pytest.mark.parametrize("width", SIZES)
async def test_queue_panel_top_row_has_label_and_five_lines_then_more(make_app: AppFactory, width: int) -> None:
    app = make_app()
    async with app.run_test(size=(width, 24)) as pilot:
        panel = app.query_one(QueuePanel)
        panel.show([_q(f"m{n}") for n in range(7)])
        await pilot.pause()
        rows = _rows(panel)
    _check_top(rows, "queued 7", width)
    assert [r.strip() for r in rows[1:]] == [f"m{n}" for n in range(5)] + ["and 2 more"]


@pytest.mark.parametrize("width", SIZES)
async def test_image_panel_top_row_has_label_and_the_chip(make_app: AppFactory, width: int) -> None:
    app = make_app()
    async with app.run_test(size=(width, 24)) as pilot:
        panel = app.query_one(AttachmentPanel)
        panel.show([CHIP])
        await pilot.pause()
        rows = _rows(panel)
    _check_top(rows, "images 1", width)
    assert rows[1].strip() == f"[image 1: {CHIP}]"


async def test_content_starts_at_column_two_in_all_three_panels(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        todo, queue, images = app.query_one(TodoPanel), app.query_one(QueuePanel), app.query_one(AttachmentPanel)
        todo.show(_todos(0, 2))
        queue.show([_q("hello")])
        images.show([CHIP])
        await pilot.pause()
        for panel in (todo, queue, images):
            first = _rows(panel)[1]
            assert len(first) - len(first.lstrip()) == 2, (type(panel).__name__, first)


async def test_every_panel_top_line_is_the_panel_colour(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        todo, queue, images = app.query_one(TodoPanel), app.query_one(QueuePanel), app.query_one(AttachmentPanel)
        todo.show(_todos(0, 2))
        queue.show([_q("x")])
        images.show([CHIP])
        await pilot.pause()
        panel_color = Color.parse(app.get_css_variables()["panel"])
        for panel in (todo, queue, images):
            assert panel.styles.border_top[1] == panel_color, type(panel).__name__
            assert panel.styles.border_top[0] == "solid"
            assert panel.styles.border_bottom[0] == ""


def _fg_at(widget: Widget, row: int, col: int) -> Color | None:
    strip = widget.screen._compositor.render_strips()[widget.region.y + row]
    x = 0
    target = widget.region.x + col
    for seg in strip:
        if x <= target < x + seg.cell_length:
            return None if seg.style is None or seg.style.color is None else Color.from_rich_color(seg.style.color)
        x += seg.cell_length
    return None


async def test_every_panel_label_is_dim_and_left_aligned(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        todo, queue, images = app.query_one(TodoPanel), app.query_one(QueuePanel), app.query_one(AttachmentPanel)
        todo.show(_todos(0, 2))
        queue.show([_q("x")])
        images.show([CHIP])
        await pilot.pause()
        assert app.get_css_variables()["text-muted"] == "auto 60%"
        for panel in (todo, queue, images):
            name = type(panel).__name__
            colour = panel.styles.border_title_color
            assert colour.a == pytest.approx(0.6), name
            assert panel.styles.border_title_align == "left", name
            assert _rows(panel)[0].startswith("─ "), name
            label_fg = _fg_at(panel, 0, 2)  # first letter of the label
            body_fg = _fg_at(panel, 1, 2)  # first letter of the content
            assert label_fg is not None, name
            assert label_fg != body_fg, name


async def test_todo_label_follows_the_content(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        panel = app.query_one(TodoPanel)
        panel.show(_todos(1, 3))
        await pilot.pause()
        assert (panel.border_title or "").strip() == "todos 1/3"
        _check_top(_rows(panel), "todos 1/3", 80)
        panel.show(_todos(2, 3))
        await pilot.pause()
        assert (panel.border_title or "").strip() == "todos 2/3"
        _check_top(_rows(panel), "todos 2/3", 80)


async def test_queue_label_follows_the_content(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        panel = app.query_one(QueuePanel)
        panel.show([_q("a"), _q("b")])
        await pilot.pause()
        _check_top(_rows(panel), "queued 2", 80)
        panel.show([_q("a"), _q("b"), _q("c")])
        await pilot.pause()
        assert (panel.border_title or "").strip() == "queued 3"
        _check_top(_rows(panel), "queued 3", 80)


async def test_image_label_follows_the_content(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        panel = app.query_one(AttachmentPanel)
        panel.show([CHIP])
        await pilot.pause()
        assert (panel.border_title or "").strip() == "images 1"
        panel.show([CHIP, CHIP])
        await pilot.pause()
        assert (panel.border_title or "").strip() == "images 2"


async def test_hiding_a_panel_clears_its_label(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        todo, queue, images = app.query_one(TodoPanel), app.query_one(QueuePanel), app.query_one(AttachmentPanel)
        todo.show(_todos(0, 2))
        queue.show([_q("a")])
        images.show([CHIP])
        todo.show([])
        queue.show([])
        images.show([])
        await pilot.pause()
        for panel in (todo, queue, images):
            assert panel.border_title == "" and not panel.display


async def test_user_text_never_changes_a_label(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        todo, queue = app.query_one(TodoPanel), app.query_one(QueuePanel)
        todo.show([TodoItem("queued 5", "pending")])
        queue.show([_q("─ todos 9/9")])
        await pilot.pause()
        assert (todo.border_title or "").strip() == "todos 0/1"
        assert (queue.border_title or "").strip() == "queued 1"
        _check_top(_rows(todo), "todos 0/1", 80)
        _check_top(_rows(queue), "queued 1", 80)


async def test_widget_order_above_the_input(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(80, 24)):
        wanted = (TodoPanel, QueuePanel, AttachmentPanel, CommandSuggestions, PathSuggestions, InputRow)
        kids = list(app.screen.children)
        positions = [next(i for i, k in enumerate(kids) if isinstance(k, cls)) for cls in wanted]
        assert positions == sorted(positions)
