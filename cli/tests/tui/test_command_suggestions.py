from __future__ import annotations

import time

from src.engine.commands.loader import Command, Commands
from src.engine.mode import Mode
from src.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from src.tui.app import BUILTINS
from src.tui.screens.approval import ApprovalScreen
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.command_suggestions import CommandSuggestions
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import NoticeEntry, UserEntry
from textual.app import App, ComposeResult

from tests.fakes import Gate
from tests.tui.conftest import AppFactory, reached, scripted, settled, until

ENTRIES = [
    ("review", "Review the diff", "project"),
    ("redo", "Again", "global"),
    ("help", "this help", None),
    ("zed", "bad\x1b[31m", "global"),
]


class Host(App[None]):
    def compose(self) -> ComposeResult:
        yield CommandSuggestions()


def prompts(widget: CommandSuggestions) -> list[str]:
    return [str(widget.get_option_at_index(i).prompt) for i in range(widget.option_count)]


async def test_it_opens_for_a_slash_prefix_and_lists_sorted_matches() -> None:
    async with Host().run_test() as pilot:
        widget = pilot.app.query_one(CommandSuggestions)
        assert not widget.is_open

        widget.update_for("/re", ENTRIES)

        assert widget.is_open
        assert prompts(widget) == ["/redo  Again (global)", "/review  Review the diff (project)"]
        assert widget.chosen() == "redo"


async def test_a_bare_slash_lists_everything_and_builtins_show_no_source() -> None:
    async with Host().run_test() as pilot:
        widget = pilot.app.query_one(CommandSuggestions)
        widget.update_for("/", ENTRIES)

        assert widget.option_count == 4
        assert "/help  this help" in prompts(widget)


async def test_it_closes_when_nothing_matches_after_a_space_or_without_a_slash() -> None:
    async with Host().run_test() as pilot:
        widget = pilot.app.query_one(CommandSuggestions)
        for text in ("/zz", "/review x", "/review\nx", "re", "", " /re", "/Re"):
            widget.update_for("/re", ENTRIES)
            assert widget.is_open
            widget.update_for(text, ENTRIES)
            assert not widget.is_open, text


async def test_move_changes_the_highlight_within_bounds() -> None:
    async with Host().run_test() as pilot:
        widget = pilot.app.query_one(CommandSuggestions)
        widget.update_for("/re", ENTRIES)

        widget.move(1)
        assert widget.chosen() == "review"
        widget.move(1)
        assert widget.chosen() == "review"
        widget.move(-5)
        assert widget.chosen() == "redo"


async def test_descriptions_are_escaped_and_not_parsed_as_markup() -> None:
    async with Host().run_test() as pilot:
        widget = pilot.app.query_one(CommandSuggestions)
        widget.update_for("/z", [("zed", "bad\x1b[31m [b]x[/b]", "global")])

        (line,) = prompts(widget)
        assert "\x1b" not in line and "\\x1b[31m [b]x[/b]" in line


async def test_filtering_200_entries_is_fast() -> None:
    entries = [(f"cmd{i}", "description " * 5, "project") for i in range(200)]
    async with Host().run_test() as pilot:
        widget = pilot.app.query_one(CommandSuggestions)
        widget.update_for("/cmd1", entries)  # warm up
        start = time.perf_counter()
        widget.update_for("/cmd", entries)
        widget.update_for("/cmd1", entries)
        assert (time.perf_counter() - start) / 2 < 0.005


# in the app

REPLY = [TextDelta("ok"), Done(1, 1)]


def _commands() -> Commands:
    items = (Command("review", "Do $ARGUMENTS", "Review it", "project"), Command("refactor", "R", None, "global"))
    return Commands({c.name: c for c in items}, ())


def _popup(app) -> CommandSuggestions:
    return app.query_one(CommandSuggestions)


async def test_typing_a_slash_prefix_opens_the_popup_and_a_space_closes_it(make_app: AppFactory) -> None:
    app = make_app(commands=_commands())
    async with app.run_test() as pilot:
        await pilot.press(*"/re")
        await until(pilot, lambda: _popup(app).is_open)
        assert _popup(app).chosen() == "refactor"
        await pilot.press("v")
        await until(pilot, lambda: _popup(app).chosen() == "review")
        await pilot.press("space")
        await until(pilot, lambda: not _popup(app).is_open)
        await pilot.press("backspace", "backspace", "backspace", "backspace")
        await pilot.press(*"/zz")
        await pilot.pause()
        assert not _popup(app).is_open


async def test_down_then_tab_completes_to_the_second_entry_and_sends_nothing(make_app: AppFactory) -> None:
    app = make_app(commands=_commands())
    async with app.run_test() as pilot:
        await pilot.press(*"/re")
        await until(pilot, lambda: _popup(app).is_open)
        await pilot.press("down", "tab")
        await pilot.pause()

        assert app.query_one(ChatInput).text == "/review "
        assert not _popup(app).is_open
        assert app.agent.mode is Mode.BUILD  # Tab did not toggle the mode
        assert scripted(app).requests == [] and not list(app.query(UserEntry))
        assert app.query_one(ChatInput).cursor_location == (0, 8)


async def test_enter_completes_without_sending(make_app: AppFactory) -> None:
    app = make_app(commands=_commands())
    async with app.run_test() as pilot:
        await pilot.press(*"/re")
        await until(pilot, lambda: _popup(app).is_open)
        await pilot.press("down", "enter")
        await pilot.pause()

        assert app.query_one(ChatInput).text == "/review "
        assert scripted(app).requests == [] and not list(app.query(UserEntry))


async def test_completed_command_runs_on_the_next_enter(make_app: AppFactory) -> None:
    app = make_app([REPLY], commands=_commands())
    async with app.run_test() as pilot:
        await pilot.press(*"/rev")
        await until(pilot, lambda: _popup(app).is_open)
        await pilot.press("tab")
        await pilot.press(*"x")
        await pilot.press("enter")
        await settled(app)

        assert scripted(app).requests[0][0][-1].content == "Do x"


async def test_escape_closes_the_popup_and_does_not_interrupt_a_running_turn(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate, TextDelta("late"), Done(1, 1)]], commands=_commands())
    async with app.run_test() as pilot:
        await pilot.press(*"hi")
        await pilot.press("enter")
        await reached(gate)
        await pilot.press(*"/re")
        await until(pilot, lambda: _popup(app).is_open)

        await pilot.press("escape")
        await pilot.pause()

        assert not _popup(app).is_open
        assert app.query_one(StatusBar).busy  # the turn was not interrupted
        assert app.query_one(ChatInput).text == "/re"
        gate.release.set()
        await settled(app)
        assert any(e.text == "hi" for e in app.query(UserEntry))
        assert not any("interrupt" in n.text for n in app.query(NoticeEntry))


async def test_with_the_popup_closed_tab_toggles_the_mode_and_escape_interrupts(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate]], commands=_commands())
    async with app.run_test() as pilot:
        await pilot.press("tab")
        assert app.agent.mode is Mode.PLAN
        await pilot.press("tab")
        await pilot.press(*"hi")
        await pilot.press("enter")
        await reached(gate)

        await pilot.press("escape")
        await settled(app)

        assert not app.query_one(StatusBar).busy


async def test_the_builtin_names_are_suggested(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.press(*"/h")
        await until(pilot, lambda: _popup(app).is_open)
        assert _popup(app).chosen() == "help"
        assert BUILTINS["help"] in prompts(_popup(app))[0]


async def test_enter_on_a_fully_typed_command_name_sends_it(make_app: AppFactory) -> None:
    app = make_app([REPLY], commands=_commands())
    async with app.run_test() as pilot:
        await pilot.press(*"/review")
        await until(pilot, lambda: _popup(app).is_open)
        await pilot.press("enter")
        await settled(app)

        assert scripted(app).requests[0][0][-1].content == "Do"
        assert app.query_one(ChatInput).text == ""


async def test_the_popup_hides_when_the_input_loses_focus_and_returns_with_it(make_app: AppFactory) -> None:
    app = make_app(commands=_commands(), terminals=True)
    async with app.run_test() as pilot:
        await pilot.press(*"/re")
        await until(pilot, lambda: _popup(app).is_open)

        await pilot.press("ctrl+t")  # focus goes to the terminal
        await pilot.pause()
        assert not _popup(app).is_open

        await pilot.press("ctrl+t")  # and back to the input, which still holds "/re"
        await until(pilot, lambda: _popup(app).is_open)


async def test_tab_and_escape_belong_to_the_dialog_not_the_popup_or_mode(make_app: AppFactory) -> None:
    call = ToolCallEvent(ToolCall("w", "write", {"path": "a.txt", "content": "x"}))
    app = make_app([[call, Done(1, 1)]], commands=_commands())
    async with app.run_test() as pilot:
        await pilot.press(*"hi")
        await pilot.press("enter")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await pilot.press("tab", "escape")
        await pilot.pause()

        assert app.agent.mode is Mode.BUILD
        assert not _popup(app).is_open


async def test_tab_and_escape_with_the_terminal_focused_go_to_the_terminal(make_app: AppFactory) -> None:
    app = make_app(commands=_commands(), terminals=True)
    async with app.run_test() as pilot:
        await pilot.press(*"/re")
        await until(pilot, lambda: _popup(app).is_open)
        await pilot.press("ctrl+t")
        await pilot.pause()
        await pilot.press("tab", "escape")
        await pilot.pause()

        assert app.agent.mode is Mode.BUILD
        assert not _popup(app).is_open
        assert app.query_one(ChatInput).text == "/re"


async def test_up_and_down_move_the_cursor_in_multiline_input_without_a_popup(make_app: AppFactory) -> None:
    app = make_app(commands=_commands())
    async with app.run_test() as pilot:
        await pilot.press("a", "b")
        await pilot.press("ctrl+j")
        await pilot.press("c")
        chat = app.query_one(ChatInput)
        assert chat.cursor_location == (1, 1)

        await pilot.press("up")
        assert chat.cursor_location[0] == 0
        await pilot.press("down")
        assert chat.cursor_location[0] == 1
        assert chat.text == "ab\nc"


async def test_clicking_a_row_completes_it_like_tab(make_app: AppFactory) -> None:
    app = make_app(commands=_commands())
    async with app.run_test() as pilot:
        await pilot.press(*"/re")
        await until(pilot, lambda: _popup(app).is_open)
        await pilot.click(CommandSuggestions, offset=(2, 2))  # the second row
        await pilot.pause()

        assert app.query_one(ChatInput).text == "/review "
        assert not _popup(app).is_open
        assert isinstance(app.focused, ChatInput)


async def test_a_long_description_stays_on_one_row() -> None:
    async with Host().run_test() as pilot:
        widget = pilot.app.query_one(CommandSuggestions)
        widget.update_for("/z", [("zed", "x" * 500, "global")])

        (line,) = prompts(widget)
        assert "\n" not in line and len(line) < 120


async def test_tab_with_the_popup_open_during_a_turn_completes_and_changes_no_mode(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate, TextDelta("late"), Done(1, 1)]], commands=_commands())
    async with app.run_test() as pilot:
        await pilot.press(*"hi")
        await pilot.press("enter")
        await reached(gate)
        await pilot.press(*"/rev")
        await until(pilot, lambda: _popup(app).is_open)
        await pilot.press("tab")
        await pilot.pause()

        assert not app.query_one(StatusBar).pending_mode
        assert app.query_one(ChatInput).text == "/review "
        assert app.query_one(StatusBar).busy
        gate.release.set()
        await settled(app)
