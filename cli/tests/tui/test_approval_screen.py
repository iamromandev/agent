from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from src.tui.screens.approval import ApprovalScreen
from src.tui.styles import diff_theme
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import AssistantEntry, NoticeEntry, ToolEntry
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static

from tests.tui.conftest import AppFactory, scripted, settled, type_and_send, until


def _edit_script() -> list[list]:
    call = ToolCall("e1", "edit", {"path": "a.py", "old_string": "old", "new_string": "new"})
    return [[ToolCallEvent(call), Done(1, 1)], [TextDelta("done"), Done(1, 1)]]


def _option_ids(screen: ApprovalScreen) -> list[str | None]:
    options = screen.query_one(OptionList)
    return [options.get_option_at_index(i).id for i in range(options.option_count)]


async def _open_dialog(app, pilot) -> ApprovalScreen:
    await type_and_send(pilot, "go")
    await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
    assert isinstance(app.screen, ApprovalScreen)
    return app.screen


async def test_allow_once_applies_the_edit_and_the_transcript_shows_the_diff(
    make_app: AppFactory, tmp_path: Path
) -> None:
    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script())
    async with app.run_test() as pilot:
        screen = await _open_dialog(app, pilot)
        assert "-old" in screen.preview_text
        assert "+new" in screen.preview_text
        assert _option_ids(screen) == ["once", "always", "deny"]

        await pilot.press("up", "up", "enter")  # the highlight starts on Deny
        await settled(app)
        await pilot.pause()

        assert (tmp_path / "a.py").read_text() == "new\n"
        (tool,) = app.query(ToolEntry)
        assert "-old" in tool.detail
        assert "+new" in tool.detail
        assert not isinstance(app.screen, ApprovalScreen)
        assert [e.text for e in app.query(AssistantEntry)] == ["done"]


async def test_always_is_chosen_with_up_and_enter_and_remembered(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script())
    async with app.run_test() as pilot:
        await _open_dialog(app, pilot)
        await pilot.press("up", "enter")
        await settled(app)

        assert (tmp_path / "a.py").read_text() == "new\n"
        assert "edit" in app.agent.grants.tools


async def test_deny_with_a_note_tells_the_model_and_leaves_the_file(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script())
    async with app.run_test() as pilot:
        await _open_dialog(app, pilot)
        await pilot.press("enter")
        await pilot.press(*["space" if c == " " else c for c in "use pathlib"])
        await pilot.press("enter")
        await settled(app)

        assert (tmp_path / "a.py").read_text() == "old\n"
        tool_message = scripted(app).requests[1][0][-1]
        assert tool_message.role == "tool"
        assert "denied" in tool_message.content
        assert "use pathlib" in tool_message.content


async def test_escape_on_the_dialog_denies_and_interrupts_the_turn(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("old\n")
    app = make_app([*_edit_script(), [TextDelta("again"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await _open_dialog(app, pilot)
        await pilot.press("escape")
        await settled(app)
        await pilot.pause()

        assert (tmp_path / "a.py").read_text() == "old\n"
        assert not isinstance(app.screen, ApprovalScreen)
        assert any("interrupted" in e.text for e in app.query(NoticeEntry))
        (tool,) = app.query(ToolEntry)
        assert tool.finished
        assert not app.query_one(StatusBar).busy
        assert not app.agent.busy

        await type_and_send(pilot, "next")
        await settled(app)
        await pilot.pause()
        assert [e.text for e in app.query(AssistantEntry)][-1] == "done"


async def test_ctrl_c_with_the_dialog_open_interrupts_and_closes_it(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script())
    async with app.run_test() as pilot:
        await _open_dialog(app, pilot)
        await pilot.press("ctrl+c")
        await settled(app)
        await until(pilot, lambda: not isinstance(app.screen, ApprovalScreen))

        assert (tmp_path / "a.py").read_text() == "old\n"
        assert any("interrupted" in e.text for e in app.query(NoticeEntry))


def _highlighted_id(screen: ApprovalScreen) -> str | None:
    options = screen.query_one(OptionList)
    assert options.highlighted is not None
    return options.get_option_at_index(options.highlighted).id


class _Overlay(ModalScreen[None]):
    """Stands in for any screen that ends up above the dialog."""


@pytest.mark.parametrize("cancel", ["interrupt"])
async def test_cancelling_the_turn_removes_the_dialog_even_when_it_is_not_on_top(
    make_app: AppFactory, tmp_path: Path, cancel: str
) -> None:
    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script())
    async with app.run_test() as pilot:
        await _open_dialog(app, pilot)
        await app.push_screen(_Overlay())
        assert not isinstance(app.screen, ApprovalScreen)

        if cancel == "interrupt":
            app.interrupt_turn()
        else:
            await pilot.press("ctrl+c")
        await until(pilot, lambda: not app.query_one(StatusBar).busy)
        await pilot.pause()

        assert not any(isinstance(s, ApprovalScreen) for s in app.screen_stack)
        assert (tmp_path / "a.py").read_text() == "old\n"
        assert any("interrupted" in e.text for e in app.query(NoticeEntry))


def _record_diff_themes(monkeypatch: pytest.MonkeyPatch, module: str) -> list[bool]:
    """Wrap the module's `diff_theme` so a test can see which `dark` flag a diff was drawn with."""
    asked: list[bool] = []
    real = diff_theme

    def spy(dark: bool) -> str:
        asked.append(dark)
        return real(dark)

    monkeypatch.setattr(f"{module}.diff_theme", spy)
    return asked


# "Always here"


def _rules(tmp_path: Path):
    from src.engine.mcp.trust import TrustStore
    from src.engine.permissions.setup import prepare

    workspace = tmp_path.resolve()
    return prepare(workspace, tmp_path / "global.json", TrustStore(tmp_path.parent / f"{tmp_path.name}-store.json")).rules


async def test_choosing_always_here_applies_the_edit_grants_it_and_saves_the_rule(
    make_app: AppFactory, tmp_path: Path
) -> None:
    from tests.tui.conftest import choose

    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script(), permissions=_rules(tmp_path))
    async with app.run_test() as pilot:
        await _open_dialog(app, pilot)
        await choose(pilot, "project")
        await settled(app)

        assert (tmp_path / "a.py").read_text() == "new\n"
        assert "edit" in app.agent.grants.tools
    saved = json.loads((tmp_path / ".agent" / "settings.json").read_text())
    assert saved == {"permissions": {"allow": ["edit"]}}


def _label(screen: ApprovalScreen, option_id: str) -> str:
    options = screen.query_one(OptionList)
    return str(options.get_option(option_id).prompt)


def _bash_script(command: str) -> list[list]:
    return [[ToolCallEvent(ToolCall("b1", "bash", {"command": command})), Done(1, 1)], [TextDelta("done"), Done(1, 1)]]


async def test_a_protected_read_dialog_names_the_path_and_offers_only_allow_once_and_deny(
    make_app: AppFactory, tmp_path: Path
) -> None:
    from src.engine.guardrails.setup import prepare as prepare_guard

    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "config").write_text("Host x\n")
    guard = prepare_guard(tmp_path, None, None, {}, "Linux", home=home.resolve())
    call = ToolCall("r1", "read", {"path": str(home / ".ssh" / "config")})
    app = make_app([[ToolCallEvent(call), Done(1, 1)], [TextDelta("done"), Done(1, 1)]], guard=guard)
    async with app.run_test() as pilot:
        screen = await _open_dialog(app, pilot)

        assert _option_ids(screen) == ["once", "deny"]
        line = str(screen.query_one("#protected", Static).visual)
        assert line == f"Protected path: {(home / '.ssh' / 'config').resolve()}"
