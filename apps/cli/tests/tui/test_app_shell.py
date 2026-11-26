from __future__ import annotations

from pathlib import Path

import pytest
from src.engine.mode import Mode
from src.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from src.tui.screens.approval import ApprovalScreen
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.status import StatusBar, display_path

from tests.tui.conftest import AppFactory, scripted, settled, type_and_send, until


def test_display_path_abbreviates_the_home_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert display_path(tmp_path / "proj") == "~/proj"
    assert display_path(tmp_path) == "~"
    assert display_path(Path("/opt/elsewhere")) == "/opt/elsewhere"


async def test_status_bar_shows_mode_model_workspace_and_window(
    make_app: AppFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    app = make_app(model="foo", context_window=4096, workspace=tmp_path / "w")
    (tmp_path / "w").mkdir()
    async with app.run_test() as pilot:
        await pilot.pause()
        line = app.query_one(StatusBar).line

    assert line.startswith("BUILD")
    assert "foo" in line
    assert "/ 4096" in line
    assert "~/w" in line
    assert "working" not in line


async def test_tab_toggles_the_mode_without_moving_focus(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        assert app.focused is app.query_one(ChatInput)
        for mode, word in ((Mode.EDITS, "EDITS"), (Mode.PLAN, "PLAN"), (Mode.BUILD, "BUILD")):
            await pilot.press("tab")
            assert app.agent.mode is mode
            assert app.query_one(StatusBar).line.startswith(word)
            assert app.focused is app.query_one(ChatInput)


async def test_tab_with_the_approval_dialog_open_leaves_the_edits_mode(make_app: AppFactory) -> None:
    call = ToolCall("b1", "bash", {"command": "ls"})
    app = make_app([[ToolCallEvent(call), Done(1, 1)], [TextDelta("done"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await pilot.press("tab")
        assert app.agent.mode is Mode.EDITS
        await type_and_send(pilot, "go")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await pilot.press("tab")

        assert app.agent.mode is Mode.EDITS


async def test_enter_submits_and_ctrl_j_adds_a_line(make_app: AppFactory) -> None:
    app = make_app([[TextDelta("ok"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await pilot.press("a", "ctrl+j", "b")
        assert app.query_one(ChatInput).text == "a\nb"
        assert scripted(app).requests == []
        await pilot.press("enter")
        await settled(app)

    sent = scripted(app).requests[0][0]
    assert (sent[-1].role, sent[-1].content) == ("user", "a\nb")


async def test_enter_on_blank_input_submits_nothing(make_app: AppFactory) -> None:
    app = make_app([[TextDelta("ok"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await pilot.press("space", "enter")
        await settled(app)

    assert scripted(app).requests == []
