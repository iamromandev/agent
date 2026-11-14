"""The terminal pane and view: a real terminal on a pty, drawn by pyte, typed into through Textual."""

from __future__ import annotations

import time

import pytest
from src.agent.terminal.manager import TerminalManager
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


async def test_the_pane_is_hidden_until_it_is_shown(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test():
        assert app.query_one(TerminalPane).display is False


async def test_output_of_the_process_is_drawn_on_the_emulator_screen(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal_manager.start("p", r"printf '\033[31mred\033[0m done'; sleep 30")
        view = await open_pane(app, pilot, "p")

        await until(pilot, lambda: "red done" in screen_text(view))
        assert view.emulator.buffer[0][0].fg == "red"


async def test_output_written_before_the_tab_is_mounted_is_not_lost(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal_manager.start("p", "echo early; sleep 30")  # prints before anything is mounted
        view = await open_pane(app, pilot, "p")

        await until(pilot, lambda: "early" in screen_text(view))


async def test_terminals_started_before_the_pane_get_a_tab(terminal_manager: TerminalManager) -> None:
    terminal_manager.start("old", "sleep 30")
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        await open_pane(app, pilot, "old")

        assert app.query_one(TerminalPane).title("old") == "old"


async def test_typed_keys_reach_the_process(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal = terminal_manager.start("sh", None)
        view = await open_pane(app, pilot, "sh")
        view.focus()
        await pilot.pause()

        await pilot.press(*"echo $((20+22))", "enter")

        await until(pilot, lambda: "42" in terminal.buffer.tail(5))


async def test_the_pty_gets_the_size_of_the_view_and_follows_resizes(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test(size=(100, 40)) as pilot:
        terminal = terminal_manager.start("sh", None)
        view = await open_pane(app, pilot, "sh")
        view.focus()
        await pilot.pause()

        async def stty() -> str:
            before = len(terminal.buffer.tail(100))
            await pilot.press(*"stty size", "enter")
            await until(pilot, lambda: any(line.count(" ") == 1 and line.replace(" ", "").isdigit() for line in terminal.buffer.tail(100)[before:]))
            return next(line for line in terminal.buffer.tail(100)[before:] if line.replace(" ", "").isdigit())

        await until(pilot, lambda: view.size.width > 0)
        assert await stty() == f"{view.size.height} {view.size.width}"

        await pilot.resize_terminal(70, 30)
        await pilot.pause()
        await until(pilot, lambda: view.size.width == 70)
        assert await stty() == f"{view.size.height} 70"
        assert (view.emulator.lines, view.emulator.columns) == (view.size.height, 70)


async def test_a_pane_smaller_than_a_terminal_can_be_gets_the_minimum_size(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test(size=(6, 4)) as pilot:
        terminal_manager.start("p", "stty size; sleep 30")
        view = await open_pane(app, pilot, "p")
        await pilot.pause()

        assert view.emulator.columns >= 10 and view.emulator.lines >= 3


async def test_an_exited_terminal_shows_its_status_in_the_tab_title(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal_manager.start("ok", "sleep 30")
        terminal_manager.start("bad", "exit 3")
        pane = app.query_one(TerminalPane)
        await open_pane(app, pilot, "bad")

        await until(pilot, lambda: pane.title("bad") == "bad ✗ exited 3")
        assert pane.title("ok") == "ok"


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


async def test_starting_a_name_again_replaces_the_tab_of_the_exited_terminal(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal_manager.start("p", "exit 1")
        pane = app.query_one(TerminalPane)
        old = await open_pane(app, pilot, "p")
        await until(pilot, lambda: pane.title("p") == "p ✗ exited 1")

        terminal_manager.start("p", "echo again; sleep 30")

        await until(pilot, lambda: pane.view("p") is not None and pane.view("p") is not old)
        new = pane.view("p")
        assert new is not None
        await until(pilot, lambda: new.is_attached and "again" in screen_text(new))
        assert pane.title("p") == "p"


async def test_a_new_tab_is_selected_unless_a_terminal_has_focus(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal_manager.start("a", "sleep 30")
        pane = app.query_one(TerminalPane)
        first = await open_pane(app, pilot, "a")
        first.focus()
        await pilot.pause()

        terminal_manager.start("b", "sleep 30")
        await open_pane(app, pilot, "b")

        assert pane.active_name == "a"  # the user is typing there

        app.set_focus(None)
        terminal_manager.start("c", "sleep 30")
        await open_pane(app, pilot, "c")
        await until(pilot, lambda: pane.active_name == "c")


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


async def test_pasted_text_is_wrapped_when_the_program_asked_for_bracketed_paste(
    terminal_manager: TerminalManager,
) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        # the shell reads the paste raw: `cat -v` shows the escape bytes as ^[
        terminal = terminal_manager.start("p", r"printf '\033[?2004h'; stty raw -echo; cat -v")
        view = await open_pane(app, pilot, "p")
        view.focus()
        await until(pilot, lambda: view.emulator.bracketed_paste)

        view.post_message(events.Paste("hi"))

        await until(pilot, lambda: any("^[[200~hi^[[201~" in line for line in terminal.buffer.tail(5)))


async def test_the_program_gets_the_arrow_keys_in_application_cursor_mode(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal = terminal_manager.start("p", r"printf '\033[?1h'; stty raw -echo; cat -v")
        view = await open_pane(app, pilot, "p")
        view.focus()
        await until(pilot, lambda: view.emulator.application_cursor)

        await pilot.press("up")

        await until(pilot, lambda: any("^[OA" in line for line in terminal.buffer.tail(5)))


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


async def test_a_sequence_the_emulator_cannot_take_does_not_lose_the_rest_of_the_output(
    terminal_manager: TerminalManager,
) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal_manager.start("p", r"printf '\033[?6h\033[6none\033[;5Ctwo\033[5Cthree'; sleep 30")
        view = await open_pane(app, pilot, "p")

        await until(pilot, lambda: "three" in screen_text(view))
        assert screen_text(view).startswith("onetwo")


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


async def test_output_waiting_to_be_drawn_is_bounded_and_the_screen_ends_with_the_newest(
    terminal_manager: TerminalManager,
) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal_manager.start("p", "sleep 30")
        view = await open_pane(app, pilot, "p")
        chunk = b"0123456789" * 6_000 + b"\r\n"  # 60 KB
        for _ in range(MAX_PENDING // len(chunk) * 5):  # five times what may wait, with no tick in between
            view._feed(chunk)
        view._feed(b"THE-END\r\n")

        assert view.pending <= MAX_PENDING
        await until(pilot, lambda: "THE-END" in screen_text(view))
        assert view.pending == 0


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


async def test_receiving_output_does_not_hold_the_event_loop_while_the_emulator_parses_it(
    terminal_manager: TerminalManager,
) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal_manager.start("p", "sleep 30")
        view = await open_pane(app, pilot, "p")
        chunk = b"0123456789abcdefghijklmnopqrstuvwxyz0123456789\r\n" * 1_400  # 64 KB: ~50 ms for pyte

        start = time.perf_counter()
        view._feed(chunk)

        assert time.perf_counter() - start < 0.02  # it is queued, not parsed
        assert view.pending == len(chunk)


async def test_a_tick_leaves_a_big_backlog_for_the_next_ones(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        terminal_manager.start("p", "sleep 30")
        view = await open_pane(app, pilot, "p")
        await pilot.pause()
        view._feed(b"x" * 200_000)  # about 150 ms of emulator work

        view._drain()

        assert 0 < view.pending < 200_000  # some was drawn, most waits (a tick takes FEED_BUDGET, not all)


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


async def test_after_skipping_output_the_tail_of_a_cut_sequence_is_not_drawn_as_text(
    terminal_manager: TerminalManager,
) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        view = await _skipping_view(pilot, app, terminal_manager)
        view._feed(b"x" * (MAX_PENDING - 10))  # this one is skipped ...
        view._feed(b"1;31mLEAKED\r\nkept\r\n")  # ... and the next starts inside its colour sequence
        view._feed(b"y" * 100)

        _drain_all(view)

        text = screen_text(view)
        assert "LEAKED" not in text and "1;31m" not in text
        assert "kept" in text and "y" * 100 in text.replace("\n", "")


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


async def test_output_that_is_not_skipped_is_drawn_whole(terminal_manager: TerminalManager) -> None:
    app = Host(terminal_manager)
    async with app.run_test() as pilot:
        view = await _skipping_view(pilot, app, terminal_manager)
        view._feed(b"1;31mnot a gap\r\n")  # nothing was skipped before it: it is trimmed by nothing
        view._feed(b"end")

        assert view.pending == len(b"1;31mnot a gap\r\n") + 3
