# ruff: noqa: RUF001
"""The transcript's look: markers, gutters, spacing, and that replay draws what live entries draw."""

from __future__ import annotations

import pytest
from core.engine.session import TranscriptItem
from core.engine.tools.base import Preview, ToolResult
from rich.style import Style
from rich.text import Text
from src.tui.widgets.transcript import AssistantEntry, GutterMark, NoticeEntry, Transcript, UserEntry
from textual.app import App, ComposeResult
from textual.pilot import Pilot
from textual.widget import Widget
from textual.widgets import Collapsible, Static
from textual.widgets._collapsible import CollapsibleTitle


class Host(App[None]):
    def compose(self) -> ComposeResult:
        yield Transcript(id="t")


def _strips(widget: Widget):
    return widget.screen._compositor.render_strips()


def rows(widget: Widget) -> list[str]:
    """Screen rows the widget covers, right-trimmed."""
    box = widget.region
    strips = _strips(widget)
    return [strips[y].text[box.x : box.right].rstrip() for y in range(box.y, box.bottom)]


def shown(app: App[None]) -> list[str]:
    """The transcript's rows without the empty tail below its content."""
    out = rows(app.query_one(Transcript))
    while out and not out[-1]:
        out.pop()
    return out


def style_at(widget: Widget, row: int, col: int) -> Style:
    strip = _strips(widget)[widget.region.y + row]
    x, pos = widget.region.x + col, 0
    for seg in strip:
        if pos <= x < pos + seg.cell_length:
            return seg.style or Style()
        pos += seg.cell_length
    raise AssertionError("outside the strip")


async def settle(pilot: Pilot) -> None:
    await pilot.pause()
    await pilot.pause()


def at(width: int = 80):
    return Host().run_test(size=(width, 40))


async def test_a_user_message_is_marked_and_wraps_under_its_text() -> None:
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        await t.add_user("fix it")
        await t.add_user("a\nb")
        await settle(pilot)
        first, second = pilot.app.query(UserEntry)
        assert rows(first) == ["› fix it"]
        assert rows(second) == ["› a", "  b"]
        assert first.styles.background.a == 0


async def test_a_custom_command_label_is_a_dim_row_at_column_2() -> None:
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        await t.add_user("do it", "/name (project)")
        await settle(pilot)
        (entry,) = pilot.app.query(UserEntry)
        assert rows(entry) == ["  /name (project)", "› do it"]
        assert style_at(entry, 0, 3).color != style_at(entry, 1, 2).color  # dim label, normal text


async def test_turns_are_separated_by_one_blank_row_and_a_turn_has_none() -> None:
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        await t.add_user("one")
        await t.write_assistant("reply")
        await t.add_tool("c1", "read a.py")
        await t.add_notice("info", "note")
        await t.add_user("two")
        await settle(pilot)
        out = shown(pilot.app)
        assert out[0] == "› one"
        assert out[1].startswith("● reply")
        assert out[2].startswith("  ⋯ read a.py")
        assert out[3].startswith("  ℹ note")
        assert out[4] == ""
        assert out[5] == "› two"
        assert len(out) == 6


async def test_every_row_of_a_streamed_and_a_finished_reply_sits_at_column_2() -> None:
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        await t.write_assistant("first para\n\nsecond para")
        await settle(pilot)
        for _ in range(2):
            (entry,) = pilot.app.query(AssistantEntry)
            out = rows(entry)
            assert out[0].startswith("● first para")
            body = [r for r in out if r.strip()]
            assert len(body) == 2 and all(r.startswith("  ") for r in body[1:])
            await t.end_assistant()
            await settle(pilot)


async def test_the_reply_mark_has_the_accent_colour() -> None:
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        await t.write_assistant("hi")
        await settle(pilot)
        (entry,) = pilot.app.query(AssistantEntry)
        mark = style_at(entry, 0, 0)
        text = style_at(entry, 0, 2)
        assert mark.color is not None and mark.color != text.color


async def test_a_reply_leaves_no_blank_row_before_the_next_entry() -> None:
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        await t.write_assistant("a\n\nb")
        await t.add_notice("info", "n")
        await settle(pilot)
        out = shown(pilot.app)
        assert out[-1].startswith("  ℹ n") and out[-2].strip() == "b"


async def test_a_tool_row_is_one_row_with_a_dim_running_mark_then_the_outcome() -> None:
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        entry = await t.add_tool("c1", "read a.py")
        await settle(pilot)
        assert rows(entry) == ["  ⋯ read a.py"]
        assert style_at(entry, 0, 2).color != style_at(entry, 0, 4).color  # dim mark, normal tool name
        assert style_at(entry, 0, 4).bold
        await entry.finish(ToolResult("x", summary="120 lines"), None, False)
        await settle(pilot)
        assert rows(entry) == ["  ✓ read a.py · 120 lines"]
        assert entry.region.height == 1
        assert not any(c in rows(entry)[0] for c in "▶▼─")


async def test_an_error_shows_a_cross_and_an_interrupt_shows_the_stop_mark() -> None:
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        bad = await t.add_tool("c1", "read a.py")
        stopped = await t.add_tool("c2", "bash ls")
        await bad.finish(ToolResult("x", is_error=True, summary="boom"), None, False)
        await t.close_open_tools("interrupted")
        await settle(pilot)
        assert rows(bad)[0].startswith("  ✗ read")
        assert rows(stopped)[0].startswith("  ⏹ bash")


async def test_an_applied_edit_is_open_with_every_diff_row_under_the_gutter() -> None:
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        entry = await t.add_tool("c1", "edit a.py")
        preview = Preview("diff", "edit a.py", "-old\n+new\n")
        await entry.finish(ToolResult("ok", summary="+1 −1"), preview, True)
        await settle(pilot)
        assert not entry.collapsed
        out = rows(entry)
        assert out[0].startswith("  ✓ edit a.py")
        assert len(out) > 1 and all(r.startswith("    │") for r in out[1:])


async def test_a_bash_result_opens_and_folds_on_click_and_on_enter() -> None:
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        entry = await t.add_tool("c1", "bash ls")
        await entry.finish(ToolResult("out1\nout2", summary="exit 0"), None, False)
        await settle(pilot)
        assert entry.collapsed and len(rows(entry)) == 1
        title = entry.query_one(CollapsibleTitle)
        await pilot.click(title)
        await settle(pilot)
        assert not entry.collapsed
        out = rows(entry)
        assert out[1:] == ["    │ out1", "    │ out2"]
        await pilot.click(title)
        await settle(pilot)
        assert entry.collapsed
        title.focus()
        await pilot.press("enter")
        await settle(pilot)
        assert not entry.collapsed
        await pilot.press("enter")
        await settle(pilot)
        assert entry.collapsed


async def test_a_running_task_is_open_with_its_child_line_under_the_gutter() -> None:
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        entry = await t.add_tool("c1", "task look", task=True)
        await entry.add_child("k1", "grep TODO")
        await settle(pilot)
        assert not entry.collapsed
        assert rows(entry)[1] == "    │ ⋯ grep TODO"
        await entry.finish_child("k1", ToolResult("x", summary="3 matches"))
        await entry.finish(ToolResult("report", summary="1 steps"), None, False)
        await settle(pilot)
        assert entry.collapsed
        assert "grep TODO · 3 matches" in "\n".join(str(s.render()) for s in entry.query_one(Collapsible.Contents).query("Static"))


async def test_a_warning_is_yellow_at_column_2_and_wraps_at_column_4() -> None:
    async with at(40) as pilot:
        t = pilot.app.query_one(Transcript)
        await t.add_notice("warn", "word " * 14)
        await settle(pilot)
        (entry,) = pilot.app.query(NoticeEntry)
        out = rows(pilot.app.query_one(Transcript))
        assert out[0].startswith("  ⚠ word")
        assert len(out) > 1 and all(r.startswith("    word") for r in out[1:] if r)
        assert style_at(entry, 0, 0).color is not None
        assert style_at(entry, 0, 0).color == style_at(entry, 0, 2).color  # icon and text share the yellow
        await t.add_notice("info", "plain")
        await settle(pilot)
        info = pilot.app.query(NoticeEntry).last()
        assert style_at(entry, 0, 0).color != style_at(info, 0, 2).color


async def test_markup_in_every_kind_is_drawn_literally() -> None:
    raw = "[bold]x[/] [$var]"
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        await t.add_user(raw)
        entry = await t.add_tool("c1", f"read {raw}")
        await entry.finish(ToolResult("x", summary=raw), None, False)
        await t.add_notice("info", raw)
        await settle(pilot)
        out = shown(pilot.app)
        assert out[0] == f"› {raw}"
        assert out[1] == f"  ✓ read {raw} · {raw}"
        assert out[2] == f"  ℹ {raw}"


async def test_replayed_rows_match_live_rows_except_the_tool_mark() -> None:
    items = [
        TranscriptItem("user", "go"),
        TranscriptItem("assistant", "para one\n\npara two"),
        TranscriptItem("tool", "read a.py", "12 lines"),
        TranscriptItem("notice", "note"),
        TranscriptItem("user", "again"),
    ]
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        await t.replay(items)
        await settle(pilot)
        replayed = shown(pilot.app)
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        await t.add_user("go")
        await t.write_assistant("para one\n\npara two")
        entry = await t.add_tool("c1", "read a.py")
        await entry.finish(ToolResult("x", summary="12 lines"), None, False)
        await t.add_notice("info", "note")
        await t.add_user("again")
        await settle(pilot)
        live = shown(pilot.app)
    assert "  · read a.py · 12 lines" in replayed  # the outcome is not saved: a dim dot, not a check
    live = [r.replace("✓", "·", 1) for r in live]
    assert replayed == live


async def test_undo_removes_the_turns_rows_and_leaves_no_double_blank_row() -> None:
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        for text in ("one", "two"):
            token = await t.add_user(text)
            await t.write_assistant("r " + text)
            await t.end_assistant()
            t.end_turn(token)
        await settle(pilot)
        assert await t.remove_last_turn()
        await settle(pilot)
        assert shown(pilot.app) == ["› one", "● r one"]
        assert isinstance(next(iter(t.children)), UserEntry)


class Marked(GutterMark, Static):
    DEFAULT_CSS = "Marked { padding: 0 0 0 2; height: auto; }"
    gutter_mark = Text("● ")


class MarkHost(App[None]):
    def compose(self) -> ComposeResult:
        yield Marked("hello world " * 10, id="m")
        yield Static("\n".join(str(i) for i in range(60)))


async def test_the_gutter_mark_is_skipped_when_the_widget_is_too_narrow() -> None:
    async with MarkHost().run_test(size=(10, 30)) as pilot:
        await settle(pilot)
        assert rows(pilot.app.query_one(Marked))[0].startswith("● hello")
    async with MarkHost().run_test(size=(1, 30)) as pilot:
        await settle(pilot)
        assert "●" not in "".join(rows(pilot.app.query_one(Marked)))


async def test_the_gutter_mark_is_drawn_again_after_scrolling_back() -> None:
    async with at(40) as pilot:
        t = pilot.app.query_one(Transcript)
        await t.write_assistant("one\n\ntwo")
        for i in range(60):
            await t.add_user(f"u{i}")
        await settle(pilot)
        (entry,) = pilot.app.query(AssistantEntry)
        t.scroll_home(animate=False)
        await settle(pilot)
        assert rows(entry)[0].startswith("● one")
        t.scroll_end(animate=False)
        await settle(pilot)
        t.scroll_to(y=1, animate=False)  # the entry's first row is off screen
        await settle(pilot)
        t.scroll_home(animate=False)
        await settle(pilot)
        assert rows(entry)[0].startswith("● one")


TAILS = {
    "bullet list": ["Steps:\n\n- one\n", "- two\n", "- three"],
    "ordered list": ["Steps:\n\n1. one\n", "2. two\n", "3. three"],
    "table": ["T:\n\n| a | b |\n|---|---|\n", "| 1 | 2 |\n", "| 3 | 4 |"],
    "blockquote": ["Q:\n\n> one\n", "> two\n", "> three"],
}


@pytest.mark.parametrize("tail", list(TAILS))
async def test_a_streamed_reply_ending_in_a_block_leaves_no_blank_row(tail: str) -> None:
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        for delta in TAILS[tail]:
            await t.write_assistant(delta)
            await pilot.pause()
        await t.add_notice("info", "after")  # ends the reply
        await settle(pilot)
        out = shown(pilot.app)
        assert out[-1].startswith("  ℹ after")
        assert out[-2].strip() != ""
        (entry,) = pilot.app.query(AssistantEntry)
        assert [c.has_class("newest") for c in entry.children].count(True) == 1
        assert entry.children[-1].has_class("newest")


@pytest.mark.parametrize("tail", list(TAILS))
async def test_only_the_last_block_of_a_streaming_reply_is_newest(tail: str) -> None:
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        for delta in TAILS[tail]:
            await t.write_assistant(delta)
            await pilot.pause()
            (entry,) = pilot.app.query(AssistantEntry)
            flags = [c.has_class("newest") for c in entry.children]
            assert flags == [False] * (len(flags) - 1) + [True]


async def test_a_wide_notice_icon_does_not_overdraw_the_text() -> None:
    async with at() as pilot:
        t = pilot.app.query_one(Transcript)
        await t.add_notice("info", "hello", icon="🚀")
        await settle(pilot)
        row = shown(pilot.app)[0]
        assert row.endswith("hello") and "hello" in row
        (entry,) = pilot.app.query(NoticeEntry)
        assert rows(entry)[0].strip().endswith("hello")
