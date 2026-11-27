"""The key-hints line under the input: placement, wording per state, width, height and focus."""

from __future__ import annotations

import asyncio

from core.engine.provider.base import Done, TextDelta
from core.testing.fakes import Gate
from src.tui import StartChoice
from src.tui.app import AgentApp
from src.tui.hints import BUSY as BUSY_ENTRIES
from src.tui.hints import IDLE as IDLE_ENTRIES
from src.tui.hints import SUGGESTIONS as SUGGESTION_ENTRIES
from src.tui.hints import TERMINAL as TERMINAL_ENTRIES
from src.tui.widgets.chat_input import ChatInput, InputRow
from src.tui.widgets.command_suggestions import CommandSuggestions
from src.tui.widgets.key_hints import KeyHints
from src.tui.widgets.status import StatusBar
from src.tui.widgets.terminal import TerminalPane, TerminalView
from textual.binding import Binding
from textual.color import Color
from textual.widget import Widget

from tests.tui.conftest import AppFactory, chat, reached, settled, type_and_send, until

IDLE = "enter send · ctrl+j newline · tab mode · / commands · @ files · ctrl+t terminal"
BUSY = "esc interrupt · enter queue message · tab mode · ctrl+t terminal"
BUSY_NO_TERMINALS = "esc interrupt · enter queue message · tab mode"
IDLE_NO_TERMINALS = "enter send · ctrl+j newline · tab mode · / commands · @ files"


def _strip(widget: Widget, y: int):
    return widget.screen._compositor.render_strips()[y]


def _row_text(app, widget: Widget) -> str:
    return _strip(widget, widget.region.y).text[widget.region.x : widget.region.right]


def _hints(app) -> KeyHints:
    return app.query_one(KeyHints)


async def test_the_line_sits_between_the_input_and_the_status_bar_two_columns_in_and_dim(make_app: AppFactory) -> None:
    app = make_app(terminals=True)
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        row, hints, status = app.query_one("#input-row", InputRow), _hints(app), app.query_one(StatusBar)
        assert hints.region.y == row.region.bottom
        assert status.region.y == hints.region.bottom
        line = _strip(hints, hints.region.y)
        assert line.text[: 2 + len("enter send")] == "  enter send"
        assert _row_text(app, hints).strip() == IDLE

        pos, fg = 0, None
        for seg in line:
            if pos <= 2 < pos + seg.cell_length:
                assert seg.style is not None and seg.style.color is not None
                fg = Color.from_rich_color(seg.style.color)
            pos += seg.cell_length
        assert fg is not None
        assert hints.styles.auto_color and abs(hints.styles.color.a - 0.6) < 0.01  # `$text-muted`: auto 60%
        assert app.get_css_variables()["text-muted"] == "auto 60%"


async def test_without_a_terminal_manager_idle_ends_at_files(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(120, 24)) as pilot:
        await pilot.pause()
        assert _hints(app).plain == IDLE_NO_TERMINALS


async def test_in_a_narrow_window_the_line_is_one_row_cut_with_an_ellipsis(make_app: AppFactory) -> None:
    app = make_app(terminals=True)
    async with app.run_test(size=(40, 24)) as pilot:
        await pilot.pause()
        hints = _hints(app)
        assert hints.region.height == 1
        assert _row_text(app, hints).rstrip().endswith("…")


async def test_the_line_hides_under_20_rows_and_shows_again_on_resize(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(80, 16)) as pilot:
        await pilot.pause()
        assert not _hints(app).display
        await pilot.resize_terminal(80, 30)
        await pilot.pause()
        assert _hints(app).display
        await pilot.resize_terminal(80, 19)
        await pilot.pause()
        assert not _hints(app).display
        await pilot.resize_terminal(80, 20)
        await pilot.pause()
        assert _hints(app).display


async def test_the_line_never_takes_focus_and_a_click_leaves_it_on_the_input(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert KeyHints.can_focus is False
        await pilot.click("#hints")
        await pilot.pause()
        assert isinstance(app.focused, ChatInput)


# --- busy ---------------------------------------------------------------------------------------


async def test_a_turn_shows_busy_and_idle_returns_when_it_ends(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate, TextDelta("ok"), Done(1, 1)]], terminals=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert _hints(app).plain == IDLE
        await type_and_send(pilot, "go")
        await reached(gate)
        await until(pilot, lambda: _hints(app).plain == BUSY)
        gate.release.set()
        await settled(app)
        await until(pilot, lambda: _hints(app).plain == IDLE)


async def test_compact_shows_busy_and_idle_returns_after_it(make_app: AppFactory, save_session) -> None:
    save_session([m for n in range(4) for m in chat(f"u{n}", f"a{n}" + "." * 600)])
    gate = Gate()
    app = make_app(
        [[gate, TextDelta("gist"), Done(5, 5)]], start=StartChoice("continue"), tools=False, terminals=True
    )
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()
        assert _hints(app).plain == IDLE
        await type_and_send(pilot, "/compact")
        await reached(gate)
        await until(pilot, lambda: _hints(app).plain == BUSY)
        gate.release.set()
        await settled(app)
        await until(pilot, lambda: _hints(app).plain == IDLE)


async def test_undo_shows_idle_because_esc_and_enter_do_nothing_then_idle_after(
    make_app: AppFactory, monkeypatch
) -> None:
    app = make_app([[TextDelta("ok"), Done(1, 1)]], terminals=True)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "hello")
        await settled(app)
        real, gate = app.agent.undo, Gate()

        async def held_undo(emit):
            gate.reached.set()
            await gate.release.wait()
            return await real(emit)

        monkeypatch.setattr(app.agent, "undo", held_undo)
        task = asyncio.create_task(app._undo_command())  # the slash handler would block the pilot
        await reached(gate)
        assert app.host.undoing and app.query_one(StatusBar).busy
        await pilot.pause()
        assert _hints(app).plain == IDLE
        gate.release.set()
        await task
        await until(pilot, lambda: not app.host.undoing and _hints(app).plain == IDLE)


async def test_without_a_terminal_manager_busy_ends_at_tab_mode(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate, TextDelta("ok"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_and_send(pilot, "go")
        await reached(gate)
        await until(pilot, lambda: _hints(app).plain == BUSY_NO_TERMINALS)
        gate.release.set()
        await settled(app)


# --- suggestion lists ----------------------------------------------------------------------------

SUGGESTIONS = "↑↓ select · tab complete · esc close"


async def test_the_command_list_shows_suggestions_and_esc_returns_idle(make_app: AppFactory) -> None:
    app = make_app(terminals=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("/", "m", "o")
        await until(pilot, lambda: _hints(app).plain == SUGGESTIONS)
        await pilot.press("escape")
        await until(pilot, lambda: _hints(app).plain == IDLE)


async def test_the_path_list_shows_suggestions(make_app: AppFactory, tmp_path) -> None:
    (tmp_path / "alpha.txt").write_text("x")
    app = make_app(terminals=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("@", "a", "l")
        await until(pilot, lambda: _hints(app).plain == SUGGESTIONS)
        await pilot.press("escape")
        await until(pilot, lambda: _hints(app).plain == IDLE)


async def test_suggestions_win_over_busy(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate, TextDelta("ok"), Done(1, 1)]], terminals=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_and_send(pilot, "go")
        await reached(gate)
        await until(pilot, lambda: _hints(app).plain == BUSY)
        await pilot.press("/", "m", "o")
        await until(pilot, lambda: _hints(app).plain == SUGGESTIONS)
        await pilot.press("escape")  # closes the list only
        await until(pilot, lambda: _hints(app).plain == BUSY)
        gate.release.set()
        await settled(app)


# --- terminal pane -------------------------------------------------------------------------------

TERMINAL = "ctrl+t chat · ctrl+pageup/pagedown tabs"


async def test_ctrl_t_shows_the_terminal_line_and_ctrl_t_again_returns_idle(make_app: AppFactory) -> None:
    app = make_app(terminals=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+t")
        await until(pilot, lambda: _hints(app).plain == TERMINAL)
        await pilot.press("ctrl+t")
        await until(pilot, lambda: _hints(app).plain == IDLE)
        assert isinstance(app.focused, ChatInput)


# --- R6: every key the line names does what it says --------------------------------------------
# Idle (citations checked: each presses the key in that state)
#   enter send, ctrl+j newline: test_idle_ctrl_j_adds_a_line_and_enter_sends_it (below)
#   tab mode:                   test_app_shell.py::test_tab_toggles_the_mode_without_moving_focus
#   / commands:                 test_the_command_list_shows_suggestions_and_esc_returns_idle (above)
#   @ files:                    test_the_path_list_shows_suggestions (above)
#   ctrl+t terminal:            test_ctrl_t_opens_the_pane_when_idle (below)
# Busy
#   esc interrupt:              test_queue.py::test_ac9_interrupting_puts_the_queue_and_the_typed_text_back
#   enter queue message:        test_queue.py::test_ac1_a_message_sent_during_a_turn_is_queued
#   tab mode:                   test_tab_sets_the_mode_during_a_turn (below)
#   ctrl+t terminal:            test_ctrl_t_opens_the_pane_during_a_turn (below)
# Suggestions
#   down, tab, esc:             test_down_tab_and_esc_work_on_the_open_command_list (below)
#   (path list: test_path_suggestions.py::test_down_moves_and_a_path_with_a_space_is_quoted,
#    test_tab_with_the_list_open_does_not_toggle_the_mode, test_escape_closes_the_list_and_keeps_the_text)
# Terminal
#   ctrl+t chat:                test_ctrl_t_shows_the_terminal_line_and_ctrl_t_again_returns_idle (above)
#   ctrl+pagedown tabs:         test_terminal_app.py::test_switching_tabs_from_a_terminal_moves_focus_to_the_view_of_the_new_tab


async def test_ctrl_t_opens_the_pane_when_idle(make_app: AppFactory) -> None:
    app = make_app(terminals=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app.query_one(TerminalPane).display
        await pilot.press("ctrl+t")
        await until(pilot, lambda: app.query_one(TerminalPane).display and isinstance(app.focused, TerminalView))


async def test_ctrl_t_opens_the_pane_during_a_turn(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate, TextDelta("ok"), Done(1, 1)]], terminals=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_and_send(pilot, "go")
        await reached(gate)
        await pilot.press("ctrl+t")
        await until(pilot, lambda: app.query_one(TerminalPane).display and isinstance(app.focused, TerminalView))
        gate.release.set()
        await settled(app)


async def test_tab_sets_the_mode_during_a_turn(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate, TextDelta("ok"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await pilot.pause()
        before = app.agent.mode
        await type_and_send(pilot, "go")
        await reached(gate)
        await pilot.press("tab")
        assert app.agent.mode is not before
        gate.release.set()
        await settled(app)


async def test_down_tab_and_esc_work_on_the_open_command_list(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        popup = app.query_one(CommandSuggestions)
        await pilot.press("/")
        await until(pilot, lambda: popup.is_open and popup.option_count > 1)
        assert popup.highlighted == 0
        await pilot.press("down")
        assert popup.highlighted == 1
        chosen = popup.chosen()
        await pilot.press("tab")
        assert app.query_one(ChatInput).text == f"/{chosen} "
        app.query_one(ChatInput).clear()
        await pilot.press("/")
        await until(pilot, lambda: popup.is_open)
        await pilot.press("escape")
        await until(pilot, lambda: not popup.is_open)


def test_the_app_and_input_bindings_behind_the_named_keys_exist_exactly() -> None:
    bound = {b.key for b in AgentApp.BINDINGS if isinstance(b, Binding)}
    named = {e.split()[0] for e in (*IDLE_ENTRIES, *BUSY_ENTRIES, *SUGGESTION_ENTRIES, *TERMINAL_ENTRIES)}
    assert {"ctrl+t", "tab", "ctrl+pageup/pagedown"} <= named
    assert {"ctrl+t", "tab", "escape", "ctrl+pageup", "ctrl+pagedown"} <= bound
    assert "ctrl+j" in {b.key for b in ChatInput.BINDINGS if isinstance(b, Binding)}  # enter is `_on_key`, pressed below


async def test_idle_ctrl_j_adds_a_line_and_enter_sends_it(make_app: AppFactory) -> None:
    app = make_app([[TextDelta("ok"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await pilot.press("a", "ctrl+j", "b")
        assert app.query_one(ChatInput).text == "a\nb"
        await pilot.press("enter")
        await settled(app)
        assert app.query_one(ChatInput).text == ""
    assert app.agent.history[-2].content == "a\nb"
