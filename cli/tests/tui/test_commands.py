from __future__ import annotations

from pathlib import Path

import pytest
from src.engine.commands.loader import Command, Commands
from src.engine.mode import Mode
from src.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.queue_panel import QueuePanel
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import NoticeEntry, UserEntry

from tests.fakes import Gate
from tests.tui.conftest import AppFactory, reached, scripted, settled, type_and_send, until


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


ROOT = Path(__file__).resolve().parents[3]


async def test_an_unknown_command_gets_a_warning_and_no_request(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/nope")
        await pilot.pause()

        assert any(n.startswith("⚠") and "Unknown command" in n and "/nope" in n for n in _notices(app))
        assert scripted(app).requests == []


async def test_a_mode_toggle_mid_turn_shows_next_request_until_it_applies(
    make_app: AppFactory, tmp_path: Path
) -> None:
    (tmp_path / "a.py").write_text("x\n")
    gate = Gate()
    read = ToolCallEvent(ToolCall("r1", "read", {"path": "a.py"}))
    app = make_app([[gate, read, Done(1, 1)], [TextDelta("second"), Done(1, 1)]])
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        await type_and_send(pilot, "go")
        await reached(gate)
        await pilot.press("tab", "tab")
        assert status.line.startswith("PLAN (next request)")
        assert app.agent.mode is Mode.PLAN

        gate.release.set()
        await settled(app)
        await pilot.pause()

        assert status.line.startswith("PLAN ·")
        first_tools, second_tools = (tools for _, tools in scripted(app).requests)
        assert "edit" in first_tools
        assert "edit" not in second_tools


@pytest.mark.parametrize("command", ["/new"])
async def test_new_and_clear_after_the_edits_mode_return_to_build(make_app: AppFactory, command: str) -> None:
    app = make_app([])
    async with app.run_test() as pilot:
        await pilot.press("tab")
        assert app.query_one(StatusBar).line.startswith("EDITS")
        await type_and_send(pilot, command)
        await pilot.pause()

        assert app.agent.mode is Mode.BUILD
        assert app.query_one(StatusBar).line.startswith("BUILD")


# custom commands

REPLY = [TextDelta("ok"), Done(1, 1)]


def _commands(*commands: Command, warnings: tuple[str, ...] = ()) -> Commands:
    return Commands({c.name: c for c in commands}, warnings)


async def test_a_custom_command_is_queued_while_a_turn_runs_and_runs_after_it(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate], REPLY], commands=_commands(Command("x", "Do X: $ARGUMENTS", None, "global")))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "first")
        await reached(gate)
        await type_and_send(pilot, "/x now")
        await pilot.pause()

        assert not any("busy" in n for n in _notices(app))
        assert app.query_one(QueuePanel).plain == "/x now"
        assert app.query_one(ChatInput).text == ""
        gate.release.set()
        await settled(app)
        await pilot.pause()
        assert len(scripted(app).requests) == 2
        assert scripted(app).requests[1][0][-1].content == "Do X: now"


async def test_an_empty_custom_command_warns_and_starts_no_turn(make_app: AppFactory) -> None:
    app = make_app(commands=_commands(Command("blank", "  \n", None, "project")))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/blank")
        await pilot.pause()

        assert any("/blank is empty" in n for n in _notices(app))
        assert scripted(app).requests == [] and not list(app.query(UserEntry))


async def test_help_lists_custom_commands_sorted_with_descriptions_and_sources(make_app: AppFactory) -> None:
    app = make_app(
        commands=_commands(
            Command("review", "r", "Review the staged diff", "project"),
            Command("deploy", "d", None, "global"),
            Command("evil", "e", "bad\x1b[31m text", "global"),
        )
    )
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/help")
        await pilot.pause()
        (text,) = [n for n in _notices(app) if "Enter" in n]

    assert "Custom commands:" in text
    lines = text.split("Custom commands:\n")[1].splitlines()
    assert lines == [
        "/deploy (global)",
        "/evil  bad\\x1b[31m text (global)",
        "/review  Review the staged diff (project)",
    ]
    assert "\x1b" not in text


async def test_launch_warnings_about_commands_are_shown(make_app: AppFactory) -> None:
    app = make_app(commands=_commands(warnings=("a/b.md: not a valid command name",)))
    async with app.run_test() as pilot:
        await until(pilot, lambda: any("not a valid command name" in n for n in _notices(app)))
        assert any(n.startswith("⚠") for n in _notices(app))


async def test_the_transcript_shows_a_custom_prompt_escaped_but_the_model_gets_it_raw(make_app: AppFactory) -> None:
    body = "a\x1b[2Jb\u202eevil\n\tline2"
    app = make_app([REPLY], commands=_commands(Command("x", body, None, "project")))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/x")
        await settled(app)

        (entry,) = app.query(UserEntry)
        shown = entry.body.plain
        assert "\x1b" not in shown and "\u202e" not in shown
        assert "\\x1b[2J" in shown and "\\u202e" in shown
        assert shown.endswith("\na\\x1b[2Jb\\u202eevil\n\tline2")
        strips = entry.screen._compositor.render_strips()
        assert any("›" in strips[y].text for y in range(entry.region.y, entry.region.bottom))  # noqa: RUF001
        assert scripted(app).requests[0][0][-1].content == body
