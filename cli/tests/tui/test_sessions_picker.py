"""The session picker: /sessions, --resume without an id, keys only."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from src.agent.mode import Mode
from src.agent.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from src.tui import StartChoice
from src.tui.screens.approval import ApprovalScreen
from src.tui.screens.sessions import SessionPicker, relative
from src.tui.styles import ICONS
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import AssistantEntry, NoticeEntry, UserEntry
from textual.widgets import OptionList

from tests.fakes import Gate
from tests.tui.conftest import AppFactory, chat, choose, reached, scripted, settled, type_and_send, until

SaveSession = Callable[..., Path]
INFO = ICONS["info"]


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _options(app) -> list[str]:
    return [str(app.screen.query_one(OptionList).get_option_at_index(i).prompt) for i in range(app.screen.query_one(OptionList).option_count)]


async def _open_picker(app, pilot) -> None:
    await type_and_send(pilot, "/sessions")
    await until(pilot, lambda: isinstance(app.screen, SessionPicker))


@pytest.fixture
def three_sessions(save_session: SaveSession) -> list[Path]:
    """Oldest to newest: first, second, third. The picker lists them newest first."""
    return [
        save_session(chat("first question", "first answer"), age=3000),
        save_session(chat("second question", "second answer"), age=2000),
        save_session(chat("third question", "third answer"), age=1000),
    ]


async def test_sessions_lists_them_newest_first_with_title_time_and_count(
    make_app: AppFactory, three_sessions: list[Path]
) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await _open_picker(app, pilot)

        options = _options(app)
        assert [o[:15].strip() for o in options] == ["third question", "second question", "first question"]
        assert all(o.rstrip().endswith("2") for o in options)  # one user and one assistant message
        assert "min ago" in options[0] or "today" in options[0]


async def test_down_then_enter_switches_to_the_second_session_and_redraws_it(
    make_app: AppFactory, three_sessions: list[Path]
) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await _open_picker(app, pilot)
        await pilot.press("down")
        await pilot.press("enter")
        await settled(app)
        await pilot.pause()

        assert not isinstance(app.screen, SessionPicker)
        assert [e.text for e in app.query(UserEntry)] == ["second question"]
        assert [e.text for e in app.query(AssistantEntry)] == ["second answer"]
        assert [m.content for m in app.agent.history] == ["second question", "second answer"]
        assert app.query_one(StatusBar).estimated


async def test_enter_alone_resumes_the_newest(make_app: AppFactory, three_sessions: list[Path]) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await _open_picker(app, pilot)
        await pilot.press("enter")
        await settled(app)
        await pilot.pause()

        assert [e.text for e in app.query(UserEntry)] == ["third question"]


async def test_escape_leaves_the_current_session_unchanged(make_app: AppFactory, three_sessions: list[Path]) -> None:
    app = make_app([[TextDelta("current answer"), Done(10, 5)]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "current question")
        await settled(app)
        await pilot.pause()
        history = list(app.agent.history)
        entries = len(list(app.query("#transcript > *")))

        await _open_picker(app, pilot)
        await pilot.press("escape")
        await pilot.pause()

        assert not isinstance(app.screen, SessionPicker)
        assert app.agent.history == history
        assert len(list(app.query("#transcript > *"))) == entries
        assert [e.text for e in app.query(UserEntry)] == ["current question"]
        assert not app.query_one(StatusBar).busy


async def test_tab_in_the_picker_does_not_change_the_mode(make_app: AppFactory, three_sessions: list[Path]) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await _open_picker(app, pilot)
        await pilot.press("tab")
        await pilot.press("escape")
        await pilot.pause()

        assert app.agent.mode is Mode.BUILD


async def test_sessions_with_none_saved_shows_a_notice(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/sessions")
        await settled(app)
        await pilot.pause()

        assert not isinstance(app.screen, SessionPicker)
        assert f"{INFO} No saved sessions for this folder" in _notices(app)


async def test_sessions_while_a_turn_runs_is_refused(make_app: AppFactory, three_sessions: list[Path]) -> None:
    gate = Gate()
    app = make_app([[TextDelta("a"), gate, TextDelta("b"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await reached(gate)
        await type_and_send(pilot, "/sessions")
        await pilot.pause()

        assert not isinstance(app.screen, SessionPicker)
        assert any("busy" in n and "Esc" in n for n in _notices(app))
        assert app.agent.history  # untouched
        gate.release.set()
        await settled(app)


async def test_resume_without_an_id_opens_the_picker_at_launch(make_app: AppFactory, three_sessions: list[Path]) -> None:
    app = make_app(start=StartChoice("pick"))
    async with app.run_test() as pilot:
        await until(pilot, lambda: isinstance(app.screen, SessionPicker))
        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("enter")
        await settled(app)
        await pilot.pause()

        assert [e.text for e in app.query(UserEntry)] == ["first question"]


async def test_resume_without_an_id_and_no_sessions_shows_a_notice(make_app: AppFactory) -> None:
    app = make_app(start=StartChoice("pick"))
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()

        assert f"{INFO} No saved sessions for this folder" in _notices(app)
        assert not isinstance(app.screen, SessionPicker)


async def test_escape_in_the_launch_picker_starts_a_new_session(make_app: AppFactory, three_sessions: list[Path]) -> None:
    app = make_app([[TextDelta("hi"), Done(1, 1)]], start=StartChoice("pick"))
    async with app.run_test() as pilot:
        await until(pilot, lambda: isinstance(app.screen, SessionPicker))
        await pilot.press("escape")
        await settled(app)

        assert app.agent.history == []
        await type_and_send(pilot, "hello")
        await settled(app)
        assert [e.text for e in app.query(UserEntry)] == ["hello"]


@pytest.mark.parametrize("title", ["[bold]x[/]", "[$var] \\[x]\\", "[red", "echo [/]"])
async def test_a_title_with_markup_characters_shows_literally(
    make_app: AppFactory, save_session: SaveSession, title: str
) -> None:
    save_session(chat(title, "reply"))
    app = make_app()
    async with app.run_test() as pilot:
        await _open_picker(app, pilot)
        (option,) = _options(app)
        assert option.startswith(title)

        await pilot.press("enter")
        await settled(app)
        await pilot.pause()

        assert [e.text for e in app.query(UserEntry)] == [title]
        assert f'{INFO} Resumed "{title}" (2 messages)' in _notices(app)


async def test_a_long_title_is_cut_in_the_list(make_app: AppFactory, save_session: SaveSession) -> None:
    save_session(chat("a" * 60, "reply"))
    app = make_app()
    async with app.run_test() as pilot:
        await _open_picker(app, pilot)
        (option,) = _options(app)

        assert option.startswith("a" * 39 + "…  ")
        assert "a" * 40 not in option


async def test_switching_resets_always_grants_and_the_next_edit_asks_again(
    make_app: AppFactory, save_session: SaveSession, tmp_path: Path
) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    save_session(chat("old question", "old answer"), age=5000)
    edit = ToolCallEvent(ToolCall("c1", "edit", {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}))
    edit2 = ToolCallEvent(ToolCall("c2", "edit", {"path": "a.py", "old_string": "x = 2", "new_string": "x = 3"}))
    app = make_app([[edit, Done(1, 1)], [TextDelta("edited"), Done(1, 1)], [edit2, Done(1, 1)]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "change x")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await choose(pilot, "always")
        await settled(app)
        await pilot.pause()
        assert (tmp_path / "a.py").read_text() == "x = 2\n"
        assert app.agent.grants.covers("edit", {"path": "a.py", "old_string": "x = 2", "new_string": "x = 3"}, Mode.BUILD)

        await _open_picker(app, pilot)
        await pilot.press("down")  # the live session is newest; the old one is second
        await pilot.press("enter")
        await settled(app)
        await pilot.pause()
        assert [e.text for e in app.query(UserEntry)] == ["old question"]

        await type_and_send(pilot, "again")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await pilot.press("escape")
        await settled(app)
        assert (tmp_path / "a.py").read_text() == "x = 2\n"
        assert len(scripted(app).requests) == 3


# relative(): a fixed "now" so the wording is exact

NOW = datetime(2026, 11, 12, 15, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (NOW - timedelta(seconds=5), "just now"),
        (NOW + timedelta(minutes=3), "just now"),  # a clock a little ahead is not "in the future"
        (NOW - timedelta(seconds=59), "just now"),
        (NOW - timedelta(minutes=1), "1 min ago"),
        (NOW - timedelta(minutes=45), "45 min ago"),
        (NOW - timedelta(hours=2), "today 13:30"),
        (datetime(2026, 11, 12, 0, 5, tzinfo=UTC), "today 00:05"),
        (datetime(2026, 11, 11, 23, 59, tzinfo=UTC), "yesterday"),
        (datetime(2026, 11, 11, 1, 0, tzinfo=UTC), "yesterday"),
        (datetime(2026, 11, 10, 20, 0, tzinfo=UTC), "2 days ago"),
        (NOW - timedelta(days=30), "30 days ago"),
        (NOW - timedelta(days=31), "2026-10-12"),
        (datetime(2025, 1, 2, 3, 4, tzinfo=UTC), "2025-01-02"),
    ],
)
def test_relative_wording(moment: datetime, expected: str) -> None:
    assert relative(moment, NOW) == expected


def test_relative_reads_the_time_in_the_clock_of_now() -> None:
    plus_two = datetime(2026, 11, 12, 12, 0, tzinfo=timezone_of(2))
    moment = datetime(2026, 11, 11, 23, 0, tzinfo=UTC)  # 01:00 the next day at +02:00 = today

    assert relative(moment, plus_two) == "today 01:00"


def timezone_of(hours: int):
    from datetime import timezone

    return timezone(timedelta(hours=hours))


def test_rows_align_title_time_and_count_in_fixed_columns() -> None:
    from datetime import UTC, datetime

    from src.agent.session import SessionInfo
    from src.tui.screens.sessions import TITLE_WIDTH, describe

    now = datetime(2026, 11, 13, 12, 0, tzinfo=UTC)

    def info(title: str, count: int) -> SessionInfo:
        return SessionInfo("x", Path("x"), title, now, now - timedelta(hours=2), count)

    short, long, many = describe(info("Fix", 5), now), describe(info("x" * 80, 5), now), describe(info("Fix", 123), now)
    assert len(short.plain) == len(long.plain) == len(many.plain)
    assert short.plain.endswith("  " + "  5")
    assert many.plain.endswith("123")
    assert long.plain[: TITLE_WIDTH] == "x" * (TITLE_WIDTH - 1) + "…"
