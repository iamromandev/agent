from __future__ import annotations

import asyncio
import os
import signal
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from pathlib import Path

import pytest
from core.engine import Agent
from core.engine.agents.loader import Agents
from core.engine.approval import Approver
from core.engine.checkpoints.store import CheckpointStore
from core.engine.commands.loader import Commands
from core.engine.configguard import ConfigGuard, HeldGlobal
from core.engine.configwatch import Watch
from core.engine.format.runner import FormatHub
from core.engine.guardrails.setup import Guardrails
from core.engine.hooks.setup import Hooks
from core.engine.image_pull import ImagePuller
from core.engine.imagegen import MfluxGenerator
from core.engine.lsp.hub import LspHub
from core.engine.mcp.hub import McpHub
from core.engine.mentions_index import FileIndex
from core.engine.permissions.setup import Rules
from core.engine.provider.base import Message, ProviderEvent
from core.engine.session import (
    FileRecorder,
    Recorder,
    SessionStore,
    ToolMeta,
    sessions_root,
    workspace_dir,
)
from core.engine.skills.loader import Skills
from core.engine.terminal.manager import TerminalManager
from core.engine.tools import default_tools, read_only_tools
from core.engine.tools.base import Tool, ToolContext
from core.engine.web import Fetcher
from core.engine.workflows.loader import Workflows
from core.engine.workflows.runs import RunIndex
from core.engine.workflows.timings import TimingStore
from core.host.port import LocalPort
from core.host.session import SessionHost
from core.testing.fakes import FakeCatalog, Gate, ScriptedProvider
from core.testing.helpers import assert_all_gone
from src.tui import NEW, StartChoice
from src.tui.app import AgentApp
from textual.pilot import Pilot

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
        mcp: McpHub | None = None,
        mcp_trust: Callable[[], None] | None = None,
        permissions: Rules | None = None,
        permission_warnings: Sequence[str] = (),
        task: bool = False,
        task_fetcher: Fetcher | None = None,
        commands: Commands | None = None,
        skills: Skills | None = None,
        agents: Agents | None = None,
        guard: Guardrails | None = None,
        hooks: Hooks | None = None,
        catalog: FakeCatalog | None = None,
        window_cap: int = 8192,
        launch_notices: Sequence[str] = (),
        checkpoints: CheckpointStore | None = None,
        imager: MfluxGenerator | None = None,
        editor: MfluxGenerator | None = None,
        images: ImagePuller | None = None,
        workflows: Workflows | None = None,
        run_index: RunIndex | None = None,
        timings: TimingStore | None = None,
        extra_tools: list[Tool] | None = None,
        config_guard: ConfigGuard | None = None,
        watch: Watch | None = None,
        held: HeldGlobal | None = None,
        lsp: LspHub | None = None,
        format: FormatHub | None = None,
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
                default_tools(manager, skills=skills, imager=imager, editor=editor, pull=images) if tools else [],
                approver,
                ToolContext(root, 5.0, guard=guard),
                max_steps,
                recorder=new_recorder(),
                compact_at=compact_at,
                compact_keep_turns=compact_keep_turns,
                extra_tools=(lambda: [*(mcp.tools() if mcp is not None else []), *(extra_tools or [])])
                if mcp is not None or extra_tools
                else None,
                task_tools=read_only_tools(task_fetcher) if task else None,
                agents=agents,
                rules=permissions,
                hooks=hooks,
                checkpoints=checkpoints,
                watch=watch,
            )

        host = SessionHost(
            factory,
            provider,
            root,
            window_cap=window_cap,
            store=SessionStore(sessions, root),
            recorder_factory=new_recorder,
            catalog=catalog if catalog is not None else FakeCatalog([model]),
            guard=guard,
            permissions=permissions,
            hooks=hooks,
        )
        return AgentApp(
            LocalPort(host, FileIndex(root, host.guard)),
            start=start,
            on_close=on_close,
            terminals=manager,
            mcp=mcp,
            mcp_trust=mcp_trust,
            permission_warnings=permission_warnings,
            commands=commands,
            skills=skills,
            agents=agents,
            launch_notices=launch_notices,
            images=images,
            workflows=workflows,
            run_index=run_index if run_index is not None else RunIndex(tmp_path.parent / f"{tmp_path.name}-runs.json"),
            timings=timings if timings is not None else TimingStore(None),  # never the real data folder
            config_guard=config_guard,
            held=held,
            lsp=lsp,
            format=format,
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


def dialog_ready(app: AgentApp) -> bool:
    """The approval dialog is on top and ready for keys: mounted, its options mounted and focused. `push_screen` makes
    it `app.screen` before it is composed, so a key pressed earlier may land elsewhere."""
    from src.tui.screens.approval import ApprovalScreen
    from textual.widgets import OptionList

    screen = app.screen
    if not isinstance(screen, ApprovalScreen) or not screen.is_mounted:
        return False
    options = screen.query(OptionList)
    return bool(options) and options.first().is_mounted and screen.focused is options.first()


async def dialog(pilot: Pilot) -> None:
    """Wait until the approval dialog is ready for keys (see `dialog_ready`)."""
    app = pilot.app
    assert isinstance(app, AgentApp)
    await until(pilot, lambda: dialog_ready(app))


async def choose(pilot: Pilot, option_id: str) -> None:
    """Pick an approval option without depending on where the highlight starts, then press Enter."""
    from src.tui.screens.approval import ApprovalScreen
    from textual.widgets import OptionList

    await dialog(pilot)
    screen = pilot.app.screen
    assert isinstance(screen, ApprovalScreen)
    options = screen.query_one(OptionList)
    options.highlighted = options.get_option_index(option_id)
    await pilot.press("enter")


@pytest.fixture(autouse=True)
def _colour(monkeypatch: pytest.MonkeyPatch) -> None:
    """Textual turns every colour grey when `NO_COLOR` is set at app start, and `uv run` sets `NO_COLOR=1` for the
    processes it starts, so the look tests would depend on how pytest was launched."""
    monkeypatch.delenv("NO_COLOR", raising=False)


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
    """Wait for the app's workers and the session host's job to finish and their events to be drawn, at most WAIT
    seconds."""

    async def quiet() -> None:
        while True:
            await app.workers.wait_for_complete()
            await app._local().idle()
            if all(worker.is_finished for worker in app.workers) and not app.host.busy_tasks():
                return

    await asyncio.wait_for(quiet(), WAIT)


async def reached(gate: Gate) -> None:
    """Wait for a provider (or other fake) to arrive at its gate, at most WAIT seconds."""
    await asyncio.wait_for(gate.reached.wait(), WAIT)
