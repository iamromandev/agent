"""The terminal pane and view: a real terminal on a pty, drawn by pyte, typed into through Textual."""

from __future__ import annotations

import pytest
from src.engine.terminal.manager import TerminalManager
from src.tui.widgets.terminal import FEED_SLICE, MAX_PENDING, TerminalPane, TerminalView, _Tab, _title
from textual import events
from textual.app import App, ComposeResult
from textual.pilot import Pilot

from tests.tui.conftest import until


class Host(App[None]):
    def __init__(self, manager: TerminalManager) -> None:
        super().__init__()
        self.manager = manager

    def compose(self) -> ComposeResult:
        yield TerminalPane(self.manager)


async def open_pane(app: Host, pilot: Pilot, name: str) -> TerminalView:
    """Show the pane and wait until the tab for `name` has its view."""
    pane = app.query_one(TerminalPane)
    pane.display = True

    def mounted() -> bool:
        view = pane.view(name)
        return view is not None and view.is_attached

    await until(pilot, mounted)
    view = pane.view(name)
    assert view is not None
    return view


def screen_text(view: TerminalView) -> str:
    return "\n".join(line.rstrip() for line in view.emulator.display)


async def test_a_terminal_killed_by_a_signal_shows_the_signal(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal = terminal_manager.start("p", "sleep 30")
        pane = app.query_one(TerminalPane)
        await open_pane(app, pilot, "p")

        await terminal.stop(0.3)

        await until(pilot, lambda: pane.title("p") == "p ✗ SIGTERM")


async def test_closing_a_terminal_removes_its_tab_and_its_subscription(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal = terminal_manager.start("p", "sleep 30")
        pane = app.query_one(TerminalPane)
        await open_pane(app, pilot, "p")
        assert terminal._subscribers  # the view listens

        await terminal_manager.close("p")

        await until(pilot, lambda: pane.view("p") is None and not terminal._subscribers)
        assert pane.title("p") is None


async def test_pasted_text_is_sent_with_newlines_as_enter(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal = terminal_manager.start("cat", "cat")
        view = await open_pane(app, pilot, "cat")
        view.focus()
        await pilot.pause()

        view.post_message(events.Paste("one\ntwo"))

        # the tty echoes both lines and `cat` repeats the first: the newline was taken as Enter
        await until(pilot, lambda: "".join(terminal.buffer.tail(5)).count("one") == 2)
        assert "two" in "".join(terminal.buffer.tail(5))


async def test_an_answer_to_a_cursor_position_request_reaches_the_program(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        # asks where the cursor is, then shows what came back
        terminal = terminal_manager.start("p", r"stty raw -echo; printf '\033[6n'; dd bs=1 count=6 2>/dev/null | cat -v; sleep 30")
        await open_pane(app, pilot, "p")

        await until(pilot, lambda: any("^[[1;1R" in line for line in terminal.buffer.tail(5)))


async def test_a_key_without_an_encoding_is_left_to_the_app(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal = terminal_manager.start("p", "cat -v")
        view = await open_pane(app, pilot, "p")
        view.focus()
        await pilot.pause()

        await pilot.press("volume_up", "x", "enter")

        await until(pilot, lambda: "x" in terminal.buffer.tail(3))
        assert not any("volume" in line for line in terminal.buffer.tail(5))


async def test_escape_and_control_characters_are_removed_from_pasted_text(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal = terminal_manager.start("p", r"printf '\033[?2004h'; stty raw -echo; cat -v")
        view = await open_pane(app, pilot, "p")
        view.focus()
        await until(pilot, lambda: view.emulator.bracketed_paste)

        # an Escape would end the bracketed paste early ("\x1b[201~") and run what follows as typed
        view.post_message(events.Paste("a\x1b[201~b\x07c\x7fd\x00e\te\nf\r\ng\x9bh"))

        await until(pilot, lambda: any("^[[201~" in line for line in terminal.buffer.tail(5)))
        shown = "\n".join(terminal.buffer.tail(10))
        assert "^[[200~a[201~bcde\te\nf\ngh^[[201~" in shown  # the markers are the only escapes


def test_the_title_of_a_terminal_killed_by_an_unnamed_signal_says_the_number() -> None:
    class Dead:
        status = ("exited", -200)

    assert _title("p", Dead()) == "p ✗ signal 200"  # ty: ignore[invalid-argument-type]


async def test_a_view_that_cannot_be_added_stops_listening_to_its_terminal(
    terminal_manager: TerminalManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(self: TerminalPane, name: str, tab: object) -> None:
        raise RuntimeError("no room")

    monkeypatch.setattr(TerminalPane, "_add", broken)
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal = terminal_manager.start("p", "sleep 30")
        await app.query_one(TerminalPane).settled()
        await pilot.pause()

        assert terminal._subscribers == []


async def test_views_not_yet_mounted_stop_listening_when_the_pane_goes_away(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    terminal = terminal_manager.start("p", "sleep 30")
    async with app.run_test() as pilot:
        pane = app.query_one(TerminalPane)
        await open_pane(app, pilot, "p")
        pane._tabs["q"] = _Tab(TerminalView(terminal), "terminal-x")  # created, never mounted
        assert len(terminal._subscribers) == 2

        await pane.remove()

    assert terminal._subscribers == []


async def test_one_tick_draws_only_a_slice_of_the_waiting_output(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal_manager.start("p", "sleep 30")
        view = await open_pane(app, pilot, "p")
        await pilot.pause()
        view._feed(b"x" * 100_000)
        waiting = view.pending

        view._drain(budget=0.0)  # an exhausted budget still makes progress: one slice

        assert 0 < waiting - view.pending <= FEED_SLICE


def _drain_all(view: TerminalView) -> None:
    while view.pending:
        view._drain(budget=60)


async def _skipping_view(pilot: Pilot, app: Host, terminal_manager: TerminalManager) -> TerminalView:
    terminal_manager.start("p", "sleep 30")
    view = await open_pane(app, pilot, "p")
    await pilot.pause()
    view._pending.clear()
    view._waiting = 0
    return view


async def test_after_skipping_output_a_chunk_without_a_restart_point_is_skipped_too(
    terminal_manager: TerminalManager,
) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        view = await _skipping_view(pilot, app, terminal_manager)
        view._feed(b"x" * (MAX_PENDING - 10))
        view._feed(b"middle of a title")  # no Escape, no line end: it may all belong to the cut sequence
        view._feed(b"still the title\r\nnew")

        _drain_all(view)

        assert "middle" not in screen_text(view) and "still" not in screen_text(view)
        assert "new" in screen_text(view)


async def test_after_skipping_output_the_colour_and_charset_start_afresh(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        view = await _skipping_view(pilot, app, terminal_manager)
        view._stream.feed(b"\x1b[1;31m\x1b%@\x1b)0\x0e")  # bold red; ISO 8859-1 mode, the line-drawing charset shifted in
        assert view.emulator.cursor.attrs.fg == "red" and view.emulator.charset == 1

        view._feed(b"x" * (MAX_PENDING - 10))
        view._feed(b"\r\nplain" + b"y" * 20)

        _drain_all(view)

        attrs = view.emulator.cursor.attrs
        assert attrs.fg == "default" and not attrs.bold
        assert view.emulator.charset == 0
        assert "plain" in screen_text(view)  # not drawn with line-drawing glyphs
