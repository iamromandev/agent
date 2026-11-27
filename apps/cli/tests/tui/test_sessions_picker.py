"""The session picker: /sessions, --resume without an id, keys only."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from src.tui import StartChoice
from src.tui.screens.sessions import SessionPicker, relative
from src.tui.styles import ICONS
from src.tui.widgets.transcript import NoticeEntry, UserEntry
from textual.widgets import OptionList

from tests.tui.conftest import AppFactory, chat, settled, type_and_send, until

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


# relative(): a fixed "now" so the wording is exact

NOW = datetime(2026, 11, 12, 15, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (NOW - timedelta(hours=2), "today 13:30"),
        (datetime(2026, 11, 11, 1, 0, tzinfo=UTC), "yesterday"),
        (datetime(2026, 11, 10, 20, 0, tzinfo=UTC), "2 days ago"),
        (datetime(2025, 1, 2, 3, 4, tzinfo=UTC), "2025-01-02"),
    ],
)
def test_relative_wording(moment: datetime, expected: str) -> None:
    assert relative(moment, NOW) == expected


def timezone_of(hours: int):
    from datetime import timezone

    return timezone(timedelta(hours=hours))


def test_a_session_open_in_another_process_is_marked() -> None:
    from core.engine.session import SessionInfo
    from src.tui.screens.sessions import describe

    info = SessionInfo("01X", Path("/s/01X.jsonl"), "hello", NOW, NOW, 2, open_elsewhere=True)
    assert "(open elsewhere)" in str(describe(info, NOW))
    assert "(open elsewhere)" not in str(describe(SessionInfo("01X", Path("/s/01X.jsonl"), "hello", NOW, NOW, 2), NOW))
