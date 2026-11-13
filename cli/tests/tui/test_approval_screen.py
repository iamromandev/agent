from __future__ import annotations

from pathlib import Path

import pytest
from src.agent.mode import Mode
from src.agent.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from src.tui.screens.approval import ApprovalScreen
from src.tui.styles import diff_theme
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import AssistantEntry, NoticeEntry, ToolEntry
from textual.screen import ModalScreen
from textual.visual import RichVisual
from textual.widgets import Collapsible, Input, OptionList, Static

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


async def test_plan_mode_bash_offers_only_allow_once_and_deny(make_app: AppFactory) -> None:
    call = ToolCall("b1", "bash", {"command": "ls"})
    app = make_app([[ToolCallEvent(call), Done(1, 1)], [TextDelta("done"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await pilot.press("tab")
        screen = await _open_dialog(app, pilot)

        assert _option_ids(screen) == ["once", "deny"]
        assert "ls" in screen.preview_text


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


async def test_escape_while_typing_a_deny_note_still_denies_and_interrupts(
    make_app: AppFactory, tmp_path: Path
) -> None:
    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script())
    async with app.run_test() as pilot:
        await _open_dialog(app, pilot)
        await pilot.press("enter", "x", "escape")
        await settled(app)

        assert (tmp_path / "a.py").read_text() == "old\n"
        assert any("interrupted" in e.text for e in app.query(NoticeEntry))


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


async def test_tab_in_the_dialog_does_not_toggle_the_mode(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script())
    async with app.run_test() as pilot:
        await _open_dialog(app, pilot)
        assert app.agent.mode is Mode.BUILD

        await pilot.press("tab")
        await pilot.pause()

        assert app.agent.mode is Mode.BUILD
        assert app.query_one(StatusBar).line.startswith("BUILD")
        assert isinstance(app.screen, ApprovalScreen)
        assert app.screen.focused is not None  # focus stays inside the dialog


def _highlighted_id(screen: ApprovalScreen) -> str | None:
    options = screen.query_one(OptionList)
    assert options.highlighted is not None
    return options.get_option_at_index(options.highlighted).id


async def test_the_dialog_starts_with_deny_highlighted(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script())
    async with app.run_test() as pilot:
        screen = await _open_dialog(app, pilot)

        assert _highlighted_id(screen) == "deny"


async def test_plan_mode_bash_dialog_also_starts_on_deny(make_app: AppFactory) -> None:
    call = ToolCall("b1", "bash", {"command": "ls"})
    app = make_app([[ToolCallEvent(call), Done(1, 1)]])
    async with app.run_test() as pilot:
        await pilot.press("tab")
        screen = await _open_dialog(app, pilot)

        assert _option_ids(screen) == ["once", "deny"]
        assert _highlighted_id(screen) == "deny"


async def test_enter_right_after_the_dialog_opens_denies(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script())
    async with app.run_test() as pilot:
        await _open_dialog(app, pilot)
        await pilot.press("enter", "enter")  # a stray (double) Enter: open the note, submit it empty
        await settled(app)

        assert (tmp_path / "a.py").read_text() == "old\n"
        assert "edit" not in app.agent.grants.tools
        tool_message = scripted(app).requests[1][0][-1]
        assert tool_message.role == "tool"
        assert "denied" in tool_message.content


async def test_a_single_enter_only_opens_the_deny_note(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script())
    async with app.run_test() as pilot:
        await _open_dialog(app, pilot)
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, ApprovalScreen)
        assert app.screen.query_one(Input).has_class("-visible")
        assert (tmp_path / "a.py").read_text() == "old\n"


async def test_moving_up_twice_then_enter_allows_once_without_a_grant(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script())
    async with app.run_test() as pilot:
        await _open_dialog(app, pilot)
        await pilot.press("up", "up", "enter")
        await settled(app)

        assert (tmp_path / "a.py").read_text() == "new\n"
        assert "edit" not in app.agent.grants.tools


async def test_plan_mode_bash_up_then_enter_allows_once(make_app: AppFactory) -> None:
    call = ToolCall("b1", "bash", {"command": "echo hello-from-bash"})
    app = make_app([[ToolCallEvent(call), Done(1, 1)], [TextDelta("done"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await pilot.press("tab")
        await _open_dialog(app, pilot)
        await pilot.press("up", "enter")
        await settled(app)
        await pilot.pause()

        (tool,) = app.query(ToolEntry)
        assert "hello-from-bash" in tool.detail


async def test_ctrl_p_does_nothing_with_or_without_the_dialog(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script())
    async with app.run_test() as pilot:
        depth = len(app.screen_stack)
        await pilot.press("ctrl+p")
        await pilot.pause()
        assert len(app.screen_stack) == depth

        screen = await _open_dialog(app, pilot)
        depth = len(app.screen_stack)
        await pilot.press("ctrl+p")
        await pilot.pause()

        assert len(app.screen_stack) == depth
        assert app.screen is screen


class _Overlay(ModalScreen[None]):
    """Stands in for any screen that ends up above the dialog."""


@pytest.mark.parametrize("cancel", ["interrupt", "ctrl+c"])
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


async def test_cancelling_a_turn_with_the_dialog_on_top_leaves_no_dialog(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script())
    async with app.run_test() as pilot:
        await _open_dialog(app, pilot)
        app.interrupt_turn()
        await until(pilot, lambda: not app.query_one(StatusBar).busy)
        await pilot.pause()

        assert not any(isinstance(s, ApprovalScreen) for s in app.screen_stack)


async def test_a_denied_edit_shows_no_diff_in_its_tool_entry(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script())
    async with app.run_test() as pilot:
        await _open_dialog(app, pilot)
        await pilot.press("enter", "enter")  # Deny, then submit an empty note
        await settled(app)
        await pilot.pause()

        (tool,) = app.query(ToolEntry)
        assert tool.finished
        assert "denied" in tool.detail
        assert "-old" not in tool.detail
        assert "+new" not in tool.detail
        assert tool.collapsed


async def test_an_edit_cancelled_with_escape_shows_no_diff_in_its_tool_entry(
    make_app: AppFactory, tmp_path: Path
) -> None:
    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script())
    async with app.run_test() as pilot:
        await _open_dialog(app, pilot)
        await pilot.press("escape")
        await until(pilot, lambda: not app.query_one(StatusBar).busy)
        await pilot.pause()

        (tool,) = app.query(ToolEntry)
        assert tool.finished
        assert "-old" not in tool.detail
        assert "+new" not in tool.detail
        assert tool.collapsed
        assert (tmp_path / "a.py").read_text() == "old\n"


def _record_diff_themes(monkeypatch: pytest.MonkeyPatch, module: str) -> list[bool]:
    """Wrap the module's `diff_theme` so a test can see which `dark` flag a diff was drawn with."""
    asked: list[bool] = []
    real = diff_theme

    def spy(dark: bool) -> str:
        asked.append(dark)
        return real(dark)

    monkeypatch.setattr(f"{module}.diff_theme", spy)
    return asked


@pytest.mark.parametrize(("theme", "expected"), [("textual-dark", True), ("textual-light", False)])
async def test_the_approval_diff_uses_the_theme_matching_the_app_theme(
    make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, theme: str, expected: bool
) -> None:
    asked = _record_diff_themes(monkeypatch, "src.tui.screens.approval")
    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script())
    async with app.run_test() as pilot:
        app.theme = theme
        screen = await _open_dialog(app, pilot)
        assert isinstance(screen.query_one("#preview", Static).visual, RichVisual)  # the highlighted diff
        assert asked == [expected]


@pytest.mark.parametrize(("theme", "expected"), [("textual-dark", True), ("textual-light", False)])
async def test_the_transcript_diff_uses_the_theme_matching_the_app_theme(
    make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, theme: str, expected: bool
) -> None:
    asked = _record_diff_themes(monkeypatch, "src.tui.widgets.transcript")
    (tmp_path / "a.py").write_text("old\n")
    app = make_app(_edit_script())
    async with app.run_test() as pilot:
        app.theme = theme
        await _open_dialog(app, pilot)
        await pilot.press("up", "up", "enter")
        await settled(app)
        await pilot.pause()
        (tool,) = app.query(ToolEntry)
        (diff, *_) = tool.query(Collapsible.Contents).first().query(Static)
        assert isinstance(diff.visual, RichVisual)
        assert asked == [expected]
