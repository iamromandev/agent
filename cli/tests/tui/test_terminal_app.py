"""The terminal pane inside the app: its key, key routing to a focused terminal, and cleanup on quit."""

from __future__ import annotations

import asyncio
import fcntl
import os
import signal
import struct
import termios
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from src.agent.mode import Mode
from src.agent.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from src.agent.terminal.manager import Terminal, TerminalManager
from src.tui.app import AgentApp
from src.tui.screens.approval import ApprovalScreen
from src.tui.screens.sessions import SessionPicker
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.terminal import TerminalPane, TerminalView
from src.tui.widgets.transcript import AssistantEntry, NoticeEntry, Transcript
from textual.pilot import Pilot

from tests.fakes import Gate
from tests.helpers import assert_all_gone, wait_for_pids
from tests.tui.conftest import AppFactory, chat, choose, settled, type_and_send, until


def pane_of(app: AgentApp) -> TerminalPane:
    return app.query_one(TerminalPane)


async def test_ctrl_t_opens_the_pane_with_a_focused_shell_in_the_workspace(
    make_app: AppFactory, tmp_path: Path
) -> None:
    app = make_app(terminals=True)
    async with app.run_test() as pilot:
        assert pane_of(app).display is False

        await pilot.press("ctrl+t")
        pane = pane_of(app)
        await until(pilot, lambda: isinstance(app.focused, TerminalView))

        assert pane.display is True
        assert app.terminals is not None and app.terminals.names() == ["shell"]
        shell = app.terminals.get("shell")
        assert shell is not None and shell.command is None  # the user's shell
        await pilot.press(*"pwd", "enter")
        await until(pilot, lambda: str(tmp_path.resolve()) in shell.buffer.tail(5))


async def test_ctrl_t_moves_focus_to_the_input_and_then_hides_the_pane_keeping_the_shell(
    make_app: AppFactory,
) -> None:
    app = make_app(terminals=True)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+t")
        await until(pilot, lambda: isinstance(app.focused, TerminalView))
        assert app.terminals is not None
        shell = app.terminals.get("shell")
        assert shell is not None

        await pilot.press("ctrl+t")
        assert isinstance(app.focused, ChatInput)
        assert pane_of(app).display is True

        await pilot.press("ctrl+t")
        assert pane_of(app).display is False
        assert isinstance(app.focused, ChatInput)
        assert shell.status == ("running", None)
        assert app.terminals.names() == ["shell"]


async def test_showing_the_pane_again_focuses_the_same_shell(make_app: AppFactory) -> None:
    app = make_app(terminals=True)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+t")
        await until(pilot, lambda: isinstance(app.focused, TerminalView))
        view = app.focused
        await pilot.press("ctrl+t", "ctrl+t", "ctrl+t")  # to the input, hide, show

        await until(pilot, lambda: app.focused is view)
        assert app.terminals is not None and app.terminals.names() == ["shell"]


async def test_the_focus_key_returns_to_the_input_and_leaves_the_pane_open(make_app: AppFactory) -> None:
    app = make_app(terminals=True)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+t")
        await until(pilot, lambda: isinstance(app.focused, TerminalView))

        await pilot.press("ctrl+backslash")

        assert isinstance(app.focused, ChatInput)
        assert pane_of(app).display is True


async def test_ctrl_t_opens_the_pane_on_an_existing_terminal_without_starting_a_shell(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        terminal_manager.start("api", "sleep 30")

        await pilot.press("ctrl+t")
        await until(pilot, lambda: isinstance(app.focused, TerminalView))

        assert terminal_manager.names() == ["api"]
        assert pane_of(app).active_name == "api"


async def test_without_terminals_there_is_no_pane_and_ctrl_t_does_nothing(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.press("ctrl+t")

        assert not app.query(TerminalPane)
        assert isinstance(app.focused, ChatInput)


@pytest.fixture
def actions(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Record the app actions that must not fire while a terminal has focus."""
    fired: list[str] = []
    for name in ("action_toggle_mode", "action_interrupt_turn", "action_interrupt_or_quit"):
        monkeypatch.setattr(AgentApp, name, lambda self, name=name: fired.append(name))
    yield fired


async def test_tab_escape_and_ctrl_c_reach_a_focused_terminal_instead_of_the_app(
    make_app: AppFactory, terminal_manager: TerminalManager, actions: list[str]
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        cat = terminal_manager.start("cat", "cat -v")
        await pilot.press("ctrl+t")
        await until(pilot, lambda: isinstance(app.focused, TerminalView))

        await pilot.press(*"hi", "tab", "escape", "up", "enter")

        await until(pilot, lambda: "hi\t^[^[[A" in cat.buffer.tail(5))
        await pilot.press("ctrl+c")  # the tty turns it into SIGINT: it ends `cat`
        await until(pilot, lambda: cat.status == ("exited", -signal.SIGINT))
        assert actions == []
        assert app.agent.mode is Mode.BUILD
        assert app.is_running


async def test_the_app_keys_work_again_when_the_input_has_focus(
    make_app: AppFactory, terminal_manager: TerminalManager, actions: list[str]
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        terminal_manager.start("cat", "cat")
        await pilot.press("ctrl+t")
        await until(pilot, lambda: isinstance(app.focused, TerminalView))
        await pilot.press("ctrl+backslash")

        await pilot.press("tab", "escape", "ctrl+c")

        assert actions == ["action_toggle_mode", "action_interrupt_turn", "action_interrupt_or_quit"]


async def test_new_and_clear_keep_the_terminals_and_their_tabs(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        api = terminal_manager.start("api", "sleep 30")
        await pilot.press("ctrl+t")
        await until(pilot, lambda: isinstance(app.focused, TerminalView))
        view = pane_of(app).view("api")
        await pilot.press("ctrl+backslash")

        for command in ("/new", "/clear"):
            await type_and_send(pilot, command)
            await settled(app)

            assert terminal_manager.names() == ["api"]
            assert api.status == ("running", None)
            assert pane_of(app).view("api") is view


async def test_switching_sessions_keeps_the_terminals_and_their_tabs(
    make_app: AppFactory, terminal_manager: TerminalManager, save_session: Callable[..., Path]
) -> None:
    save_session(chat("old question", "old answer"), age=100)
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        api = terminal_manager.start("api", "sleep 30")
        await pilot.press("ctrl+t")
        await until(pilot, lambda: isinstance(app.focused, TerminalView))
        view = pane_of(app).view("api")
        await pilot.press("ctrl+backslash")

        await type_and_send(pilot, "/sessions")
        await until(pilot, lambda: isinstance(app.screen, SessionPicker))
        await pilot.press("enter")  # the one saved session
        await until(pilot, lambda: any(e.text == "old answer" for e in app.query(AssistantEntry)))
        await settled(app)

        assert terminal_manager.names() == ["api"]
        assert api.status == ("running", None)
        assert pane_of(app).view("api") is view
        assert pane_of(app).title("api") == "api"


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


async def test_quitting_with_ctrl_c_when_idle_leaves_no_terminal_process(
    make_app: AppFactory, terminal_manager: TerminalManager, tmp_path: Path
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        pids = await _start_chains(terminal_manager, tmp_path)

        await pilot.press("ctrl+c")

    await assert_all_gone(*pids)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
async def test_a_termination_signal_exits_the_app_and_stops_the_terminals(
    make_app: AppFactory, terminal_manager: TerminalManager, tmp_path: Path, sig: signal.Signals
) -> None:
    previous = signal.signal(sig, lambda *_: None)  # were the app to ignore the signal, the test run survives
    try:
        app = make_app(terminals=terminal_manager)
        async with app.run_test() as pilot:
            pids = await _start_chains(terminal_manager, tmp_path)

            os.kill(os.getpid(), sig)
            await until(pilot, lambda: not app.is_running)

        await assert_all_gone(*pids)
    finally:
        signal.signal(sig, previous)


async def test_an_unhandled_error_in_the_app_still_stops_the_terminals(
    make_app: AppFactory, terminal_manager: TerminalManager, tmp_path: Path
) -> None:
    app = make_app(terminals=terminal_manager)

    def boom() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        async with app.run_test() as pilot:
            pids = await _start_chains(terminal_manager, tmp_path)

            app.call_later(boom)
            await until(pilot, lambda: not app.is_running)

    await assert_all_gone(*pids)


def winsize(terminal: Terminal) -> tuple[int, int]:
    """The (rows, columns) the pty is set to."""
    rows, cols, *_ = struct.unpack("HHHH", fcntl.ioctl(terminal.process.fd, termios.TIOCGWINSZ, b"\0" * 8))
    return rows, cols


async def test_a_terminal_started_while_the_pane_is_hidden_gets_the_pane_size_when_shown(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test(size=(100, 40)) as pilot:
        api = terminal_manager.start("api", "sleep 30")  # no tab is visible: it keeps the default size
        await pilot.pause(0.2)
        assert pane_of(app).display is False

        await pilot.press("ctrl+t")
        await until(pilot, lambda: isinstance(app.focused, TerminalView))

        view = app.focused
        assert isinstance(view, TerminalView)
        await until(pilot, lambda: view.size.width > 0 and winsize(api) == (view.size.height, view.size.width))
        assert (view.emulator.lines, view.emulator.columns) == (view.size.height, view.size.width)


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


async def test_an_approval_dialog_opened_from_the_input_takes_keys_at_once(
    make_app: AppFactory, tmp_path: Path
) -> None:
    (tmp_path / "a.py").write_text("old\n")
    call = ToolCall("e1", "edit", {"path": "a.py", "old_string": "old", "new_string": "new"})
    app = make_app([[ToolCallEvent(call), Done(1, 1)], [TextDelta("done"), Done(1, 1)]], terminals=True)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))

        await pilot.press("up", "up", "enter")
        await settled(app)

        assert (tmp_path / "a.py").read_text() == "new\n"


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
async def test_a_termination_signal_ends_the_app_with_a_failing_exit_code(
    make_app: AppFactory, terminal_manager: TerminalManager, sig: signal.Signals
) -> None:
    previous = signal.signal(sig, lambda *_: None)
    try:
        app = make_app(terminals=terminal_manager)
        async with app.run_test() as pilot:
            os.kill(os.getpid(), sig)
            await until(pilot, lambda: not app.is_running)

        assert app.return_code == 128 + sig
    finally:
        signal.signal(sig, previous)


@pytest.mark.parametrize("disposition", [signal.SIG_IGN, "handler"])
async def test_the_signal_dispositions_from_before_the_app_are_restored_when_it_ends(
    make_app: AppFactory, terminal_manager: TerminalManager, disposition: object
) -> None:
    """`nohup agent` ignores SIGHUP; it must go on ignoring it after the app has closed."""
    seen: list[int] = []
    wanted = (lambda signum, frame: seen.append(signum)) if disposition == "handler" else disposition
    before = {sig: signal.signal(sig, wanted) for sig in (signal.SIGTERM, signal.SIGHUP)}  # ty: ignore[invalid-argument-type]
    try:
        app = make_app(terminals=terminal_manager)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert signal.getsignal(signal.SIGHUP) is not wanted  # the app's own while it runs

        for sig in (signal.SIGTERM, signal.SIGHUP):
            assert signal.getsignal(sig) is wanted
    finally:
        for sig, handler in before.items():
            signal.signal(sig, handler)


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


async def test_clicking_a_tab_selects_it(make_app: AppFactory, terminal_manager: TerminalManager) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        await _open(app, pilot, terminal_manager, "sleep 30", "sleep 30")
        pane = pane_of(app)
        await until(pilot, lambda: pane.active_name == "t2")

        await pilot.click("#--content-tab-terminal-1")  # the tab of t1 (ids come from the pane)

        await until(pilot, lambda: pane.active_name == "t1")


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


async def test_ctrl_w_reaches_a_running_terminal_instead_of_closing_it(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        (cat,) = await _open(app, pilot, terminal_manager, "stty raw -echo; cat -v")  # readline needs ctrl+w to delete a word

        await pilot.press("ctrl+w", "enter")

        await until(pilot, lambda: any("^W" in line for line in cat.buffer.tail(5)))
        assert cat.status == ("running", None)
        assert pane_of(app).view("t1") is not None


async def test_ctrl_w_in_the_input_edits_the_text_and_closes_nothing(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        await _open(app, pilot, terminal_manager, "exit 0")
        await until(pilot, lambda: pane_of(app).title("t1") == "t1 ✗ exited 0")
        await pilot.press("ctrl+backslash")
        await pilot.press(*"one two", "ctrl+w")

        assert app.query_one(ChatInput).text == "one "
        assert terminal_manager.names() == ["t1"]


async def test_ctrl_w_with_the_tab_bar_focused_stops_a_running_terminal_and_closes_its_tab(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        t1, t2 = await _open(app, pilot, terminal_manager, "sleep 30", "sleep 30")
        pane = pane_of(app)
        await until(pilot, lambda: pane.active_name == "t2")
        await pilot.click("#--content-tab-terminal-1")  # clicking a tab focuses the tab bar
        await until(pilot, lambda: pane.active_name == "t1" and not isinstance(app.focused, TerminalView | ChatInput))

        await pilot.press("ctrl+w")

        await until(pilot, lambda: pane.view("t1") is None)
        assert t1.status == ("exited", -signal.SIGTERM)  # stopped first
        assert terminal_manager.names() == ["t2"]
        assert t2.status == ("running", None)
        await until(pilot, lambda: pane.active_name == "t2")


async def test_ctrl_w_with_the_transcript_focused_closes_no_running_terminal(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        (t1,) = await _open(app, pilot, terminal_manager, "sleep 30")
        app.set_focus(app.query_one(Transcript))
        await until(pilot, lambda: isinstance(app.focused, Transcript))

        await pilot.press("ctrl+w")
        await pilot.pause(0.2)

        assert t1.status == ("running", None)
        assert terminal_manager.names() == ["t1"]
        assert pane_of(app).view("t1") is not None


async def test_ctrl_w_with_nothing_focused_closes_no_running_terminal(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        (t1,) = await _open(app, pilot, terminal_manager, "sleep 30")
        app.set_focus(None)
        await until(pilot, lambda: app.focused is None)

        await pilot.press("ctrl+w")
        await pilot.pause(0.2)

        assert t1.status == ("running", None)
        assert terminal_manager.names() == ["t1"]


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


async def test_a_terminal_started_while_the_pane_is_hidden_is_announced_in_the_transcript(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        terminal_manager.start("api", "sleep 30")

        await until(pilot, lambda: any("terminal api started" in n for n in _notices(app)))
        (notice,) = [n for n in _notices(app) if "terminal api" in n]
        assert "ctrl+t to show" in notice
        assert pane_of(app).display is False


async def test_the_model_starting_a_terminal_while_the_pane_is_hidden_is_announced(
    make_app: AppFactory,
) -> None:
    call = ToolCall("t1", "term_start", {"name": "web", "command": "sleep 30"})
    app = make_app([[ToolCallEvent(call), Done(1, 1)], [TextDelta("started"), Done(1, 1)]], terminals=True)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "start it")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await choose(pilot, "once")
        await settled(app)

        await until(pilot, lambda: any("terminal web started" in n for n in _notices(app)))


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


async def test_when_a_close_ends_the_focus_has_already_left_the_tab_bar(
    make_app: AppFactory, terminal_manager: TerminalManager
) -> None:
    # Removing the focused view moves focus to the tab bar; a deferred `focus()` back to the running tab would
    # leave a window after the close where ctrl+w (allowed from the tab bar) closes the running one too.
    app = make_app(terminals=terminal_manager)
    async with app.run_test() as pilot:
        await _open(app, pilot, terminal_manager, "sleep 30", "exit 3")
        pane = pane_of(app)
        await until(pilot, lambda: pane.title("t2") == "t2 ✗ exited 3")
        await until(pilot, lambda: pane.active_name == "t2" and app.focused is pane.view("t2"))

        await app._close_terminal_tab("t2")

        assert app.focused is pane.view("t1")
        assert not app.check_action("close_terminal_tab", ())
