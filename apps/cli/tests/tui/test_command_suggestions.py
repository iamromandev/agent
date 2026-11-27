from __future__ import annotations

from core.engine.commands.loader import Command, Commands
from core.engine.mode import Mode
from core.engine.provider.base import Done, TextDelta
from core.testing.fakes import Gate
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.command_suggestions import CommandSuggestions
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import NoticeEntry, UserEntry
from textual.app import App, ComposeResult

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


# in the app

REPLY = [TextDelta("ok"), Done(1, 1)]


def _commands() -> Commands:
    items = (Command("review", "Do $ARGUMENTS", "Review it", "project"), Command("refactor", "R", None, "global"))
    return Commands({c.name: c for c in items}, ())


def _popup(app) -> CommandSuggestions:
    return app.query_one(CommandSuggestions)


async def test_enter_completes_without_sending(make_app: AppFactory) -> None:
    app = make_app(commands=_commands())
    async with app.run_test() as pilot:
        await pilot.press(*"/re")
        await until(pilot, lambda: _popup(app).is_open)
        await pilot.press("down", "enter")
        await pilot.pause()

        assert app.query_one(ChatInput).text == "/review "
        assert scripted(app).requests == [] and not list(app.query(UserEntry))


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
        await pilot.press("tab", "tab")
        assert app.agent.mode is Mode.PLAN
        await pilot.press("tab")
        await pilot.press(*"hi")
        await pilot.press("enter")
        await reached(gate)

        await pilot.press("escape")
        await settled(app)

        assert not app.query_one(StatusBar).busy


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
