"""`@path` completion in the app (AC7)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from src.engine.provider.base import Done, TextDelta
from src.tui.app import AgentApp
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.path_suggestions import PathSuggestions
from src.tui.widgets.status import StatusBar
from textual.pilot import Pilot

from tests.tui.conftest import AppFactory, scripted, settled, until

REPLY = [TextDelta("ok"), Done(1, 1)]


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x")
    (tmp_path / "src" / "b.py").write_text("x")
    (tmp_path / "my dir").mkdir()
    (tmp_path / "my dir" / "n.txt").write_text("x")
    (tmp_path / "my file.txt").write_text("x")
    (tmp_path / ".env").write_text("SECRET=1")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "x").write_text("x")
    return tmp_path


def popup(app: AgentApp) -> PathSuggestions:
    return app.query_one(PathSuggestions)


def chosen(app: AgentApp) -> str:
    choice = popup(app).chosen()
    assert choice is not None
    return choice.path


def at(app: AgentApp, index: int) -> str:
    choice = popup(app).at(index)
    assert choice is not None
    return choice.path


def box(app: AgentApp) -> ChatInput:
    return app.query_one(ChatInput)


async def indexed(app: AgentApp, pilot: Pilot) -> None:
    await until(pilot, lambda: app.file_index.ready)


async def test_enter_completes_a_file_without_sending(make_app: AppFactory, ws: Path) -> None:
    app = make_app([REPLY], workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press(*"@src/a")
        await until(pilot, lambda: popup(app).is_open)
        await pilot.press("enter")
        await pilot.pause()

        assert box(app).text == "@src/a.py "
        assert not popup(app).is_open
        assert scripted(app).requests == []


async def test_enter_on_an_exactly_typed_file_sends(make_app: AppFactory, ws: Path) -> None:
    app = make_app([REPLY], workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press(*"@src/a.py")
        await until(pilot, lambda: popup(app).is_open)
        await pilot.press("enter")
        await settled(app)

        assert len(scripted(app).requests) == 1
        assert box(app).text == ""
        assert not popup(app).is_open


async def test_escape_closes_the_list_and_keeps_the_text(make_app: AppFactory, ws: Path) -> None:
    app = make_app(workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press(*"@sr")
        await until(pilot, lambda: popup(app).is_open)
        await pilot.press("escape")
        await pilot.pause()

        assert not popup(app).is_open
        assert box(app).text == "@sr"
        assert not app.query_one(StatusBar).busy


async def test_down_moves_and_a_path_with_a_space_is_quoted(make_app: AppFactory, ws: Path) -> None:
    app = make_app(workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press(*"@my")
        await until(pilot, lambda: popup(app).is_open)
        await pilot.press("down")
        assert chosen(app) == "my file.txt"
        await pilot.press("tab")
        await pilot.pause()

        assert box(app).text == '@"my file.txt" '
        assert not popup(app).is_open


async def test_tab_with_the_list_open_does_not_toggle_the_mode(make_app: AppFactory, ws: Path) -> None:
    app = make_app(workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        before = app.agent.mode
        await pilot.press(*"@sr")
        await until(pilot, lambda: popup(app).is_open)
        await pilot.press("tab")
        assert app.agent.mode is before


async def test_clicking_a_row_completes_it(make_app: AppFactory, ws: Path) -> None:
    app = make_app(workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press(*"@src/")
        await until(pilot, lambda: popup(app).is_open)
        await pilot.click(PathSuggestions, offset=(2, 2))  # the second row
        await pilot.pause()

        assert box(app).text == "@src/b.py "
        assert isinstance(app.focused, ChatInput)


async def test_a_rebuild_asked_for_during_a_build_runs_after_it(make_app: AppFactory, ws: Path) -> None:
    app = make_app(workspace=ws)
    real = app.file_index.build
    first_done = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def build() -> None:
        real()
        calls.append(1)
        if len(calls) == 1:
            first_done.set()
            release.wait(5)

    app.file_index.build = build  # ty: ignore[invalid-assignment]
    async with app.run_test() as pilot:
        await until(pilot, first_done.is_set)
        (ws / "late.txt").write_text("x")
        app._reindex()  # the turn ended while the build was still running
        release.set()
        await until(pilot, lambda: bool(app.file_index.suggest("late")))
        assert len(calls) == 2


async def test_the_post_build_update_after_the_app_is_gone_does_not_raise(make_app: AppFactory, ws: Path) -> None:
    app = make_app(workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
    await app._build_index()
