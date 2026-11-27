"""The terminal pane inside the app: its key, key routing to a focused terminal, and cleanup on quit."""

from __future__ import annotations

import asyncio
import fcntl
import struct
import termios
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from core.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from core.engine.terminal.manager import Terminal, TerminalManager
from core.testing.fakes import Gate
from core.testing.helpers import assert_all_gone, wait_for_pids
from src.tui.app import AgentApp
from src.tui.screens.approval import ApprovalScreen
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.terminal import TerminalPane, TerminalView
from src.tui.widgets.transcript import NoticeEntry
from textual.pilot import Pilot

from tests.tui.conftest import AppFactory, settled, type_and_send, until


def pane_of(app: AgentApp) -> TerminalPane:
    return app.query_one(TerminalPane)


async def test_the_focus_key_returns_to_the_input_and_leaves_the_pane_open(make_app: AppFactory) -> None:
    app = make_app(terminals=True)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+t")
        await until(pilot, lambda: isinstance(app.focused, TerminalView))

        await pilot.press("ctrl+backslash")

        assert isinstance(app.focused, ChatInput)
        assert pane_of(app).display is True


@pytest.fixture
def actions(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Record the app actions that must not fire while a terminal has focus."""
    fired: list[str] = []
    for name in ("action_toggle_mode", "action_interrupt_turn", "action_interrupt_or_quit"):
        monkeypatch.setattr(AgentApp, name, lambda self, name=name: fired.append(name))
    yield fired


def _chain(workspace: Path, name: str) -> str:
    """A command that records its PID and its child's, and waits on it."""
    return f"sh -c 'sleep 1000 & echo $! > {workspace}/{name}.child; echo $$ > {workspace}/{name}.pid; wait'"


async def _start_chains(manager: TerminalManager, workspace: Path) -> list[int]:
    manager.start("one", _chain(workspace, "one"))
    manager.start("two", _chain(workspace, "two"))
    files = [workspace / f"{n}.{kind}" for n in ("one", "two") for kind in ("pid", "child")]
    return await wait_for_pids(*files)


async def test_quitting_with_slash_quit_leaves_no_terminal_process(
    make_app: AppFactory, terminal_manager: TerminalManager, tmp_path: Path
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        pids = await _start_chains(terminal_manager, tmp_path)

        await type_and_send(pilot, "/quit")

    await assert_all_gone(*pids)
    assert terminal_manager.names() == []


def winsize(terminal: Terminal) -> tuple[int, int]:
    """The (rows, columns) the pty is set to."""
    rows, cols, *_ = struct.unpack("HHHH", fcntl.ioctl(terminal.process.fd, termios.TIOCGWINSZ, b"\0" * 8))
    return rows, cols


async def test_keys_typed_in_a_terminal_do_not_answer_an_approval_dialog_that_opens_meanwhile(
    make_app: AppFactory, tmp_path: Path
) -> None:
    (tmp_path / "a.py").write_text("old\n")
    gate = Gate()
    call = ToolCall("e1", "edit", {"path": "a.py", "old_string": "old", "new_string": "new"})
    app = make_app(
        [[gate, ToolCallEvent(call), Done(1, 1)], [TextDelta("done"), Done(1, 1)]],
        terminals=True,
    )
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await asyncio.wait_for(gate.reached.wait(), 5)
        await pilot.press("ctrl+t")  # the user goes on with the shell while the model works
        await until(pilot, lambda: isinstance(app.focused, TerminalView))

        gate.release.set()
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await pilot.press("up", "up", "enter")  # what the user was typing for the shell: Allow once

        assert isinstance(app.screen, ApprovalScreen)
        assert (tmp_path / "a.py").read_text() == "old\n"

        await until(pilot, lambda: time.monotonic() >= app._keys_ignored_until)  # the dialog takes keys again
        await pilot.press("up", "up", "enter")
        await until(pilot, lambda: (tmp_path / "a.py").read_text() == "new\n")  # slow CI runners need more than settled()
        await settled(app)


async def _open(app: AgentApp, pilot: Pilot, manager: TerminalManager, *commands: str) -> list[Terminal]:
    """Start a terminal per command (named t1, t2, ...), show the pane and wait for every tab."""
    terminals = [manager.start(f"t{i}", command) for i, command in enumerate(commands, 1)]
    await pilot.press("ctrl+t")
    await until(pilot, lambda: isinstance(app.focused, TerminalView))
    await until(pilot, lambda: all(pane_of(app).title(t.name) is not None for t in terminals))
    return terminals


async def test_ctrl_pagedown_and_ctrl_pageup_switch_the_tab_and_wrap_around(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        await _open(app, pilot, terminal_manager, "sleep 30", "sleep 30", "sleep 30")
        await pilot.press("ctrl+backslash")  # to the input: the keys work from there
        pane = pane_of(app)
        await until(pilot, lambda: pane.active_name == "t3")  # a new tab is selected when nothing is typed in

        await pilot.press("ctrl+pagedown")
        await until(pilot, lambda: pane.active_name == "t1")
        await pilot.press("ctrl+pagedown")
        await until(pilot, lambda: pane.active_name == "t2")
        await pilot.press("ctrl+pageup", "ctrl+pageup")
        await until(pilot, lambda: pane.active_name == "t3")
        assert isinstance(app.focused, ChatInput)


async def test_switching_tabs_from_a_terminal_moves_focus_to_the_view_of_the_new_tab(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        await _open(app, pilot, terminal_manager, "cat -v", "cat -v")
        pane = pane_of(app)
        await until(pilot, lambda: pane.active_name == "t2" and pane.view("t2") is app.focused)

        await pilot.press("ctrl+pagedown")  # wraps to the first

        await until(pilot, lambda: pane.active_name == "t1" and app.focused is pane.view("t1"))
        await pilot.press(*"x")  # typed into the first terminal
        t1, t2 = terminal_manager.get("t1"), terminal_manager.get("t2")
        assert t1 is not None and t2 is not None
        await until(pilot, lambda: "x" in t1.buffer.tail(3))
        assert "x" not in t2.buffer.tail(3)


async def test_the_tab_keys_do_nothing_while_the_pane_is_hidden(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        await _open(app, pilot, terminal_manager, "sleep 30", "sleep 30")
        pane = pane_of(app)
        await pilot.press("ctrl+backslash", "ctrl+t")  # hidden, the input has focus
        assert pane.display is False
        before = pane.active_name

        await pilot.press("ctrl+pagedown", "ctrl+w")
        await pilot.pause(0.1)

        assert pane.active_name == before
        assert terminal_manager.names() == ["t1", "t2"]


async def test_ctrl_w_closes_the_tab_of_an_exited_terminal_with_focus_and_stays_in_the_pane(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        await _open(app, pilot, terminal_manager, "sleep 30", "exit 3")
        pane = pane_of(app)
        await until(pilot, lambda: pane.title("t2") == "t2 ✗ exited 3")
        await until(pilot, lambda: pane.active_name == "t2" and app.focused is pane.view("t2"))

        await pilot.press("ctrl+w")

        await until(pilot, lambda: pane.view("t2") is None)
        assert terminal_manager.names() == ["t1"]
        await until(pilot, lambda: pane.active_name == "t1" and app.focused is pane.view("t1"))
        assert pane.display is True


async def test_a_burst_of_ctrl_w_closes_only_the_exited_tab_not_the_running_one_behind_it(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        t1, _ = await _open(app, pilot, terminal_manager, "sleep 30", "exit 3")
        pane = pane_of(app)
        await until(pilot, lambda: pane.title("t2") == "t2 ✗ exited 3")
        await until(pilot, lambda: pane.active_name == "t2" and app.focused is pane.view("t2"))

        for _ in range(6):  # keys a few ms apart, as a held key repeats
            app.simulate_key("ctrl+w")
            await asyncio.sleep(0.004)
        await until(pilot, lambda: pane.view("t2") is None)
        await pilot.pause(0.3)

        assert terminal_manager.names() == ["t1"]
        assert t1.status == ("running", None)
        assert pane.view("t1") is not None


async def test_a_slow_close_does_not_take_focus_from_the_input_the_user_went_to(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        # t1 ignores SIGTERM: closing it takes the grace period
        await _open(app, pilot, terminal_manager, "trap '' TERM; while :; do sleep 1; done", "sleep 30")
        pane = pane_of(app)
        await pilot.click("#--content-tab-terminal-1")  # the tab bar, on t1
        await until(pilot, lambda: pane.active_name == "t1" and not isinstance(app.focused, TerminalView | ChatInput))
        await pilot.pause(0.3)  # the shell has set its trap by now

        await pilot.press("ctrl+w")  # the close waits out the grace period
        await pilot.press("ctrl+backslash")  # the user goes to the chat input meanwhile
        await until(pilot, lambda: isinstance(app.focused, ChatInput))

        await until(pilot, lambda: pane.view("t1") is None, timeout=10)
        await pilot.pause(0.3)

        assert isinstance(app.focused, ChatInput)
        assert pane.active_name == "t2"


async def test_closing_the_last_tab_hides_the_pane_and_focuses_the_input(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        await _open(app, pilot, terminal_manager, "exit 0")
        pane = pane_of(app)
        await until(pilot, lambda: pane.title("t1") == "t1 ✗ exited 0")

        await pilot.press("ctrl+w")

        await until(pilot, lambda: pane.view("t1") is None)
        await until(pilot, lambda: pane.display is False)
        assert isinstance(app.focused, ChatInput)
        assert terminal_manager.names() == []

        await pilot.press("ctrl+t")  # starts a fresh shell
        await until(pilot, lambda: isinstance(app.focused, TerminalView))
        assert terminal_manager.names() == ["shell"]


def _notices(app: AgentApp) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


async def test_no_announcement_when_the_pane_is_shown_or_the_user_opens_it_with_ctrl_t(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+t")  # starts the shell and shows the pane
        await until(pilot, lambda: isinstance(app.focused, TerminalView))
        terminal_manager.start("api", "sleep 30")  # the pane is shown now
        await until(pilot, lambda: pane_of(app).title("api") is not None)
        await pilot.pause(0.1)

        assert not any("started" in n for n in _notices(app))
