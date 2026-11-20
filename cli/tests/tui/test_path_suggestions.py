"""`@path` completion in the app (AC7)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from src.engine.mentions_index import current_token
from src.engine.mode import Mode
from src.engine.provider.base import Done, TextDelta
from src.tui.app import AgentApp
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.command_suggestions import CommandSuggestions
from src.tui.widgets.path_suggestions import PathSuggestions
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import NoticeEntry
from textual.pilot import Pilot
from textual.widgets import TextArea

from tests.fakes import Gate
from tests.tui.conftest import AppFactory, reached, scripted, settled, until

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


@pytest.mark.parametrize(
    ("text", "token"),
    [("@sr", "sr"), ("see @src/a", "src/a"), ('x @"my f', '"my f'), ("me@x", None), ("@a b", None), ("", None), ("@", "")],
)
def test_current_token(text: str, token: str | None) -> None:
    assert current_token(text) == token


async def test_typing_at_lists_matches_and_hides_secrets_and_ignored_files(make_app: AppFactory, ws: Path) -> None:
    app = make_app(workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press("@")
        await until(pilot, lambda: popup(app).is_open)
        listed = [at(app, i) for i in range(popup(app).option_count)]
        assert "src/" in listed
        assert ".env" not in listed


async def test_tab_completes_a_folder_and_keeps_the_list_open(make_app: AppFactory, ws: Path) -> None:
    app = make_app(workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press(*"@sr")
        await until(pilot, lambda: popup(app).is_open)
        assert chosen(app) == "src/"
        await pilot.press("tab")
        await pilot.pause()

        assert box(app).text == "@src/"
        assert popup(app).is_open
        assert chosen(app) == "src/a.py"


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


async def test_tab_on_an_exactly_typed_file_completes_and_does_not_send(make_app: AppFactory, ws: Path) -> None:
    app = make_app([REPLY], workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press(*"@src/a.py")
        await until(pilot, lambda: popup(app).is_open)
        await pilot.press("tab")
        await pilot.pause()

        assert box(app).text == "@src/a.py "
        assert scripted(app).requests == []


async def test_enter_on_an_exactly_typed_quoted_file_sends(make_app: AppFactory, ws: Path) -> None:
    app = make_app([REPLY], workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press(*'@"my file.txt')
        await until(pilot, lambda: popup(app).is_open)
        await pilot.press("enter")
        await settled(app)

        assert len(scripted(app).requests) == 1


async def test_one_enter_completes_a_prefix_and_a_second_sends(make_app: AppFactory, ws: Path) -> None:
    app = make_app([REPLY], workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press(*"@src/")
        await pilot.press("a")
        await until(pilot, lambda: popup(app).is_open and chosen(app) == "src/a.py")
        await pilot.press("enter")
        await pilot.pause()
        assert box(app).text == "@src/a.py "
        assert scripted(app).requests == []

        await pilot.press("enter")
        await settled(app)
        assert len(scripted(app).requests) == 1


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


async def test_a_folder_with_a_space_opens_a_quote_and_stays_open(make_app: AppFactory, ws: Path) -> None:
    app = make_app(workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press(*"@my")
        await until(pilot, lambda: popup(app).is_open)
        await pilot.press("tab")
        await pilot.pause()

        assert box(app).text == '@"my dir/'
        assert chosen(app) == "my dir/n.txt"


async def test_completion_replaces_only_the_token_at_the_cursor(make_app: AppFactory, ws: Path) -> None:
    app = make_app(workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press(*"look at @src/b")
        await until(pilot, lambda: popup(app).is_open)
        await pilot.press("tab")
        await pilot.pause()

        assert box(app).text == "look at @src/b.py "


async def test_a_command_prefix_opens_the_command_list_not_the_path_list(make_app: AppFactory, ws: Path) -> None:
    app = make_app(workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press(*"/he")
        await until(pilot, lambda: app.query_one(CommandSuggestions).is_open)
        assert not popup(app).is_open


async def test_tab_with_the_list_closed_still_toggles_the_mode(make_app: AppFactory, ws: Path) -> None:
    app = make_app(workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press(*"hello")
        await pilot.press("tab", "tab")
        assert app.agent.mode is Mode.PLAN


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


async def test_the_index_is_rebuilt_after_a_turn(make_app: AppFactory, ws: Path) -> None:
    app = make_app([REPLY], workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        (ws / "created.txt").write_text("x")
        await pilot.press(*"hi")
        await pilot.press("enter")
        await settled(app)
        await until(pilot, lambda: bool(app.file_index.suggest("creat")))


async def test_gitignored_files_are_not_offered(make_app: AppFactory, ws: Path) -> None:
    (ws / ".gitignore").write_text("*.log\nbuild/\n")
    (ws / "debug.log").write_text("x")
    app = make_app(workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press("@")
        await until(pilot, lambda: popup(app).is_open)
        listed = [at(app, i) for i in range(popup(app).option_count)]
        assert "src/" in listed and "debug.log" not in listed and "build/" not in listed


async def test_escape_with_the_list_closed_still_interrupts_a_running_turn(make_app: AppFactory, ws: Path) -> None:
    gate = Gate()
    app = make_app([[gate]], workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press(*"hi")
        await pilot.press("enter")
        await reached(gate)
        assert not popup(app).is_open

        await pilot.press("escape")
        await settled(app)
        assert any("interrupt" in n.text for n in app.query(NoticeEntry))


async def test_down_then_a_selection_event_keeps_the_highlight(make_app: AppFactory, ws: Path) -> None:
    app = make_app(workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press(*"@src/")
        await until(pilot, lambda: popup(app).is_open)
        await pilot.press("down")
        assert chosen(app) == "src/b.py"

        box(app).post_message(TextArea.SelectionChanged(box(app).selection, box(app)))
        await pilot.pause()
        await pilot.pause()
        assert chosen(app) == "src/b.py"


async def test_typing_a_character_rebuilds_the_list_once(
    make_app: AppFactory, ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app(workspace=ws)
    async with app.run_test() as pilot:
        await indexed(app, pilot)
        await pilot.press("@")
        await until(pilot, lambda: popup(app).is_open)
        await pilot.pause()
        rebuilds: list[int] = []
        original = PathSuggestions.clear_options

        def counting(self: PathSuggestions) -> PathSuggestions:
            rebuilds.append(1)
            return original(self)

        monkeypatch.setattr(PathSuggestions, "clear_options", counting)
        await pilot.press("s")
        await pilot.pause()
        await pilot.pause()
        assert rebuilds == [1]


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


async def test_quitting_while_a_slow_build_runs_returns_promptly(
    make_app: AppFactory, ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.engine.mentions_index as module

    started = threading.Event()
    ended = threading.Event()

    def slow_walk(top: object, *a: object, **k: object):
        try:
            for i in range(10_000):
                started.set()
                time.sleep(0.01)
                yield str(ws), [], [f"f{i}.txt"]
        finally:
            ended.set()

    monkeypatch.setattr(module.os, "walk", slow_walk)
    monkeypatch.setenv("PATH", "")
    app = make_app(workspace=ws)
    start = time.monotonic()
    async with app.run_test() as pilot:
        await until(pilot, started.is_set)
    assert ended.wait(1), "the build thread kept walking after the app quit"
    assert time.monotonic() - start < 3
