from __future__ import annotations

import asyncio
import os
import signal
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from pathlib import Path

import pytest
from src.agent import Agent
from src.agent.approval import Approver
from src.agent.provider.base import Message, ProviderEvent
from src.agent.session import (
    FileRecorder,
    Recorder,
    SessionStore,
    ToolMeta,
    sessions_root,
    workspace_dir,
)
from src.agent.terminal.manager import TerminalManager
from src.agent.tools import default_tools
from src.agent.tools.base import ToolContext
from src.tui import NEW, StartChoice
from src.tui.app import AgentApp
from textual.pilot import Pilot

from tests.fakes import Gate, ScriptedProvider
from tests.helpers import assert_all_gone

WAIT = 5.0  # no test waits longer than this for the app; a regression fails instead of hanging
TEST_LIMIT = 30

AppFactory = Callable[..., AgentApp]


class Terminals:
    """The terminal managers a test made, and every PID they started."""

    def __init__(self) -> None:
        self.managers: list[TerminalManager] = []
        self.pids: list[int] = []

    def make(self, workspace: Path) -> TerminalManager:
        manager = TerminalManager(workspace, grace=0.3)  # a short grace: stopping is quick in tests

        def started(name: str, event: str) -> None:
            if event == "started" and (terminal := manager.get(name)) is not None:
                self.pids.append(terminal.process.pid)

        manager.listen(started)
        self.managers.append(manager)
        return manager


@pytest.fixture
async def terminal_pool(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Terminals]:
    """Managers for the test; each is closed afterwards and every process it started must be gone."""
    monkeypatch.setenv("SHELL", "/bin/sh")  # terminals run $SHELL: not the developer's
    made = Terminals()
    yield made
    for manager in made.managers:
        await manager.close_all()
    await assert_all_gone(*made.pids)


@pytest.fixture
def terminal_manager(terminal_pool: Terminals, tmp_path: Path) -> TerminalManager:
    return terminal_pool.make(tmp_path.resolve())


@pytest.fixture
def make_app(tmp_path: Path, terminal_pool: Terminals) -> AppFactory:
    """Build an AgentApp over a ScriptedProvider and a tmp_path workspace."""

    def build(
        script: list[list[ProviderEvent | Gate]] | None = None,
        *,
        delay: float = 0.0,
        model: str = "fake",
        context_window: int = 1000,
        check_result: str | None = None,
        max_steps: int = 5,
        workspace: Path | None = None,
        on_close: Callable[[], Awaitable[None]] | None = None,
        start: StartChoice = NEW,
        compact_at: float = 0.8,
        compact_keep_turns: int = 2,
        tools: bool = True,
        terminals: bool | TerminalManager = False,
    ) -> AgentApp:
        """Wired like `src.tui.run`: sessions are saved under the test's own sessions folder."""
        root = (workspace or tmp_path).resolve()
        provider = ScriptedProvider(
            script or [], delay=delay, model=model, context_window=context_window, check_result=check_result
        )
        sessions = sessions_root(None)  # AGENT_SESSIONS_DIR: a folder of this test only
        manager: TerminalManager | None = None
        if isinstance(terminals, TerminalManager):
            manager = terminals
        elif terminals:  # a manager of this test: its processes are stopped (and checked) when the test ends
            manager = terminal_pool.make(root)

        def new_recorder() -> Recorder:
            return FileRecorder(workspace_dir(sessions, root), root)

        def factory(approver: Approver) -> Agent:
            return Agent(
                provider,
                default_tools(manager) if tools else [],
                approver,
                ToolContext(root, 5.0),
                max_steps,
                recorder=new_recorder(),
                compact_at=compact_at,
                compact_keep_turns=compact_keep_turns,
            )

        return AgentApp(
            factory,
            provider,
            root,
            store=SessionStore(sessions, root),
            start=start,
            recorder_factory=new_recorder,
            on_close=on_close,
            terminals=manager,
        )

    return build


SavedMessages = Sequence[tuple[Message, ToolMeta | None]]


@pytest.fixture
def save_session(tmp_path: Path) -> Callable[..., Path]:
    """Write a session file for the test workspace through a real FileRecorder; returns its path.

    `age` is how many seconds ago it was last updated (its mtime), so tests can order sessions.
    """

    def save(messages: SavedMessages, *, workspace: Path | None = None, age: float = 0.0) -> Path:
        root = (workspace or tmp_path).resolve()
        recorder = FileRecorder(workspace_dir(sessions_root(None), root), root)
        for message, meta in messages:
            recorder.message(message, meta)
        recorder.close()
        assert recorder.path is not None
        when = time.time() - age
        os.utime(recorder.path, (when, when))
        return recorder.path

    return save


def chat(user: str, assistant: str) -> list[tuple[Message, ToolMeta | None]]:
    """One saved exchange."""
    return [(Message("user", user), None), (Message("assistant", assistant), None)]


async def until(pilot: Pilot, condition: Callable[[], bool], timeout: float = 5.0) -> None:
    """Let the app process messages until `condition()` holds."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await pilot.pause(0.01)
    await asyncio.sleep(0)


async def type_and_send(pilot: Pilot, text: str) -> None:
    await pilot.press(*text)
    await pilot.press("enter")


def scripted(app: AgentApp) -> ScriptedProvider:
    assert isinstance(app.provider, ScriptedProvider)
    return app.provider


async def choose(pilot: Pilot, option_id: str) -> None:
    """Pick an approval option without depending on where the highlight starts, then press Enter."""
    from src.tui.screens.approval import ApprovalScreen
    from textual.widgets import OptionList

    screen = pilot.app.screen
    assert isinstance(screen, ApprovalScreen)
    options = screen.query_one(OptionList)
    options.highlighted = options.get_option_index(option_id)
    await pilot.press("enter")


@pytest.fixture(autouse=True)
def _test_time_limit():
    """Net for anything not covered by a bounded wait (a shutdown that never ends, a stuck key press).

    The event loop runs in the main thread, so an alarm that raises there fails the test.
    """

    def fail(signum: int, frame: object) -> None:
        raise TimeoutError(f"test ran longer than {TEST_LIMIT}s")

    previous = signal.signal(signal.SIGALRM, fail)
    signal.alarm(TEST_LIMIT)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


async def settled(app: AgentApp) -> None:
    """Wait for the app's workers to finish, at most WAIT seconds."""
    await asyncio.wait_for(app.workers.wait_for_complete(), WAIT)


async def reached(gate: Gate) -> None:
    """Wait for a provider (or other fake) to arrive at its gate, at most WAIT seconds."""
    await asyncio.wait_for(gate.reached.wait(), WAIT)
