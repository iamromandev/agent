"""Hooks in the app: the shared trust prompt, project hooks off until trusted, `/hooks`, the transcript lines."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from pathlib import Path

import pytest
from src.engine.hooks.runner import HookRun
from src.engine.hooks.setup import Hooks
from src.engine.hooks.setup import prepare as prepare_hooks
from src.engine.mcp.hub import McpHub
from src.engine.mcp.setup import prepare as prepare_mcp
from src.engine.mcp.trust import TrustStore, trust_store_path
from src.engine.permissions.config import project_settings_path
from src.engine.permissions.setup import PermissionSetup, prepare
from src.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.tui import StartChoice
from src.tui.app import HELP, AgentApp
from src.tui.screens.mcp_trust import McpTrustScreen
from src.tui.screens.permission_trust import PermissionTrustScreen
from src.tui.screens.sessions import SessionPicker
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import NoticeEntry, UserEntry
from textual.widgets import Label, OptionList

from tests.fakes import FakeConnection, FakeConnector, FakeRunner, Gate, exit_run, ok_run
from tests.tui.conftest import AppFactory, chat, scripted, settled, type_and_send, until

Script = list[list[ProviderEvent | Gate]]


def _read(call_id: str = "c1") -> list[ProviderEvent | Gate]:
    return [ToolCallEvent(ToolCall(call_id, "read", {"path": "a.txt"})), Done(1, 1)]


def _done() -> list[ProviderEvent | Gate]:
    return [TextDelta("done"), Done(1, 1)]


class Launch:
    """What `src.tui.run` wires for hooks: both settings files, the trust store, a FakeRunner and the app."""

    def __init__(self, make_app: AppFactory, tmp_path: Path) -> None:
        self.make_app = make_app
        self.workspace = tmp_path.resolve()
        (self.workspace / "a.txt").write_text("hello\n")
        self.store = TrustStore(trust_store_path(name="permission-trust.json"))
        self.global_path = tmp_path / "config" / "settings.json"
        self.project_path = project_settings_path(self.workspace)
        self.runner = FakeRunner()
        self.setup: PermissionSetup
        self.hooks: Hooks

    def write(self, path: Path, **keys: object) -> bytes:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(keys).encode()
        path.write_bytes(raw)
        return raw

    def project(self, **keys: object) -> bytes:
        return self.write(self.project_path, **keys)

    def app(self, script: Script | None = None, **kwargs: object) -> AgentApp:
        self.setup = prepare(self.workspace, self.global_path, self.store)
        self.hooks = prepare_hooks(self.setup, self.workspace, self.runner)
        warnings = (*self.setup.warnings, *self.hooks.warnings)
        return self.make_app(script, permissions=self.setup.rules, permission_warnings=warnings, hooks=self.hooks, **kwargs)


@pytest.fixture
def launch(make_app: AppFactory, tmp_path: Path) -> Launch:
    return Launch(make_app, tmp_path)


PRE = {"PreToolUse": [{"matcher": "read", "command": "check-read"}]}


def _is_prompt(app: AgentApp) -> bool:
    screen = app.screen
    if not isinstance(screen, PermissionTrustScreen):
        return False
    options = screen.query(OptionList)
    return bool(options) and options.first().highlighted == 0 and screen.focused is options.first()


async def _answer(pilot, option_id: str) -> None:
    await until(pilot, lambda: time.monotonic() >= pilot.app._keys_ignored_until)
    options = pilot.app.screen.query_one(OptionList)
    options.highlighted = options.get_option_index(option_id)
    await pilot.press("enter")


def _notices(app: AgentApp) -> list[str]:
    """The text of every notice without its icon."""
    return [e.text.split(" ", 1)[1] for e in app.query(NoticeEntry)]


def _labels(app: AgentApp) -> list[str]:
    return [str(label.render()) for label in app.screen.query(Label)]


def _tool_results(app: AgentApp) -> list[str]:
    return [m.content for m in app.agent.history if m.role == "tool"]


async def test_ac3_the_prompt_lists_each_hook_and_nothing_is_spawned_until_trust(launch: Launch) -> None:
    launch.project(hooks={**PRE, "Stop": [{"command": "say-bye\x1b[31m"}]})
    app = launch.app([_read(), _done()])
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))

        text = "\n".join(_labels(app))
        assert "Trust this project's settings?" in text and "2 hooks" in text
        assert "hook PreToolUse [read] (timeout 60s) check-read" in text
        assert "hook Stop [*] (timeout 60s) say-bye\\x1b[31m" in text and "\x1b" not in text
        assert launch.runner.calls == []

        await _answer(pilot, "decline")
        await until(pilot, lambda: not isinstance(app.screen, PermissionTrustScreen))
        await type_and_send(pilot, "go")
        await settled(app)

        assert launch.runner.calls == []  # declined: never spawned
        assert "hello" in _tool_results(app)[0]
        assert any("hooks not trusted" in n and "off for this run" in n for n in _notices(app))


async def test_ac4_the_mcp_prompt_and_the_settings_prompt_come_one_after_the_other(
    launch: Launch, tmp_path: Path
) -> None:
    launch.project(permissions={"allow": ["bash(ls)"]}, hooks=PRE)
    (launch.workspace / ".agent" / "mcp.json").write_text(json.dumps({"servers": {"db": {"command": "psql"}}}))
    mcp_setup = prepare_mcp(launch.workspace, tmp_path / "mcp-global.json", TrustStore(trust_store_path()))
    hub = McpHub(FakeConnector({"db": FakeConnection()}), launch.workspace, mcp_setup.servers, 5.0, untrusted=mcp_setup.untrusted)
    app = launch.app(mcp=hub, mcp_trust=mcp_setup.remember)
    seen: list[str] = []

    def record() -> None:
        stack = [type(s).__name__ for s in app.screen_stack if isinstance(s, McpTrustScreen | PermissionTrustScreen)]
        assert len(stack) <= 1, f"overlapping prompts: {stack}"
        if stack and (not seen or seen[-1] != stack[0]):
            seen.append(stack[0])

    async with app.run_test() as pilot:
        await until(pilot, lambda: (record(), isinstance(app.screen, McpTrustScreen))[1])
        await until(pilot, lambda: time.monotonic() >= app._keys_ignored_until)
        await pilot.press("escape")
        await until(pilot, lambda: (record(), isinstance(app.screen, PermissionTrustScreen))[1])
        await _answer(pilot, "trust")
        await until(pilot, lambda: (record(), not isinstance(app.screen, PermissionTrustScreen))[1])

    assert seen == ["McpTrustScreen", "PermissionTrustScreen"]


async def test_ac12_without_hooks_the_command_says_so_and_help_lists_it(launch: Launch) -> None:
    app = launch.app()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/hooks")
        await until(pilot, lambda: "No hooks." in _notices(app))
    assert "/hooks" in HELP


async def test_ac12_hook_warnings_show_at_launch(launch: Launch) -> None:
    launch.write(launch.global_path, hooks={"Stop": [{"command": ""}]})
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: any(n.startswith("hooks (global)") for n in _notices(app)))


# UserPromptSubmit, Stop, SessionStart


def _system_notes(app: AgentApp, request: int = 0) -> list[str]:
    return [m.content for m in scripted(app).requests[request][0][1:] if m.role == "system"]


async def test_ac9_a_blocking_prompt_hook_leaves_the_history_alone_and_puts_the_prompt_back(launch: Launch) -> None:
    launch.write(launch.global_path, hooks={"UserPromptSubmit": [{"command": "gate"}]})
    launch.runner.script = [exit_run(2, "not today\x1b[2J")]
    app = launch.app([_done()])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "hello there")
        await settled(app)
        await pilot.pause()

        assert app.agent.history == [] and scripted(app).requests == []
        assert list(app.query(UserEntry)) == []
        assert app.query_one(ChatInput).text == "hello there"
        assert "hook UserPromptSubmit (global) blocked: not today\\x1b[2J" in _notices(app)
        assert not app.query_one(StatusBar).busy
        assert launch.runner.calls[0][1]["prompt"] == "hello there"


async def test_ac9_prompt_context_is_a_system_note_in_the_request_and_a_dim_line(launch: Launch) -> None:
    launch.write(launch.global_path, hooks={"UserPromptSubmit": [{"command": "ctx"}]})
    launch.runner.script = [ok_run(json.dumps({"context": "on branch dev"}))]
    app = launch.app([_done()])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "hello")
        await settled(app)
        await pilot.pause()

        assert _system_notes(app) == ["[hook context]\non branch dev"]
        assert [m.role for m in app.agent.history] == ["user", "assistant"]
        assert "hook UserPromptSubmit (global) context: on branch dev" in _notices(app)
        assert [e.text for e in app.query(UserEntry)] == ["hello"]


async def test_esc_during_a_prompt_hook_kills_it_and_restores_the_prompt(launch: Launch) -> None:
    class Hanging(FakeRunner):
        started = asyncio.Event()
        cancelled = False

        async def run(
            self, command: str, stdin: bytes, timeout: float, env_extra: Mapping[str, str], *, event: str = ""
        ) -> HookRun:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            raise AssertionError("unreachable")

    runner = Hanging()
    launch.write(launch.global_path, hooks={"UserPromptSubmit": [{"command": "slow"}]})
    launch.runner = runner
    app = launch.app([_done()])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "keep me")
        await until(pilot, lambda: runner.started.is_set())
        await pilot.press("escape")
        await settled(app)
        await pilot.pause()

        assert runner.cancelled and app.agent.history == []
        assert app.query_one(ChatInput).text == "keep me" and not app.query_one(StatusBar).busy


def _start(launch: Launch, script: Script | None = None, **kwargs: object) -> AgentApp:
    launch.write(launch.global_path, hooks={"SessionStart": [{"command": "hello"}]})
    return launch.app(script, **kwargs)


def _sources(launch: Launch) -> list[str]:
    return [c[1]["source"] for c in launch.runner.calls]


async def test_continue_with_nothing_to_resume_is_startup(launch: Launch) -> None:
    app = _start(launch, start=StartChoice("continue"))
    async with app.run_test() as pilot:
        await settled(app)
        await until(pilot, lambda: _sources(launch) == ["startup"])


async def test_ac11_switching_with_sessions_is_switch(launch: Launch, save_session) -> None:
    save_session(chat("old q", "old a"))
    app = _start(launch, [_done()])
    async with app.run_test() as pilot:
        await settled(app)
        await until(pilot, lambda: _sources(launch) == ["startup"])
        await type_and_send(pilot, "/sessions")
        await until(pilot, lambda: isinstance(app.screen, SessionPicker))
        await pilot.press("enter")
        await settled(app)
        await until(pilot, lambda: _sources(launch) == ["startup", "switch"])


async def test_a_cancelled_picker_does_not_run_session_start(launch: Launch, save_session) -> None:
    save_session(chat("old q", "old a"))
    app = _start(launch)
    async with app.run_test() as pilot:
        await settled(app)
        await until(pilot, lambda: _sources(launch) == ["startup"])
        await type_and_send(pilot, "/sessions")
        await until(pilot, lambda: isinstance(app.screen, SessionPicker))
        await pilot.press("escape")
        await settled(app)
        await pilot.pause()

        assert _sources(launch) == ["startup"]


class SlowStart(FakeRunner):
    """The first `hang` runs wait until cancelled (a slow SessionStart hook); later ones answer at once."""

    def __init__(self, hang: int = 1) -> None:
        super().__init__()
        self.hang = hang
        self.started = 0
        self.cancelled = 0

    async def run(
        self, command: str, stdin: bytes, timeout: float, env_extra: Mapping[str, str], *, event: str = ""
    ) -> HookRun:
        self.started += 1
        if self.started <= self.hang:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        return await super().run(command, stdin, timeout, env_extra)


async def test_a_message_typed_while_a_session_start_hook_runs_is_bounced_and_kept_in_the_box(launch: Launch) -> None:
    runner = SlowStart()
    launch.runner = runner
    app = _start(launch, [_done()])
    async with app.run_test() as pilot:
        await until(pilot, lambda: runner.started == 1)
        await type_and_send(pilot, "hi")
        await pilot.pause()

        assert app.query_one(ChatInput).text == "hi" and app.agent.history == []
        assert "busy \u2014 a session is loading" in _notices(app)

        await pilot.press("escape")
        await settled(app)
        await pilot.press("enter")  # the text is still in the box: now it is accepted
        await settled(app)

        assert [m.role for m in app.agent.history] == ["user", "assistant"]


async def test_mcp_servers_start_before_a_slow_session_start_hook_ends(launch: Launch, tmp_path: Path) -> None:
    runner = SlowStart()
    launch.runner = runner
    launch.write(launch.global_path, hooks={"SessionStart": [{"command": "hello"}]})
    mcp_path = tmp_path / "mcp-global.json"
    mcp_path.write_text(json.dumps({"servers": {"db": {"command": "psql"}}}))
    mcp_setup = prepare_mcp(launch.workspace, mcp_path, TrustStore(trust_store_path()))
    connector = FakeConnector({"db": FakeConnection()})
    hub = McpHub(connector, launch.workspace, mcp_setup.servers, 5.0, untrusted=mcp_setup.untrusted)
    app = launch.app([_done()], mcp=hub)
    async with app.run_test() as pilot:
        await until(pilot, lambda: runner.started == 1)
        await until(pilot, lambda: len(connector.connects) == 1)
        await pilot.press("escape")
        await settled(app)


async def test_a_failing_session_start_step_on_sessions_is_one_warning_and_releases_the_switch(
    launch: Launch, save_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_session(chat("old q", "old a"))
    app = _start(launch, [_done()])
    async with app.run_test() as pilot:
        await settled(app)
        await until(pilot, lambda: _sources(launch) == ["startup"])

        async def boom(source: str, emit: object) -> None:
            raise RuntimeError("bad\x1b[31m")

        monkeypatch.setattr(app.agent, "session_started", boom)
        await type_and_send(pilot, "/sessions")
        await until(pilot, lambda: isinstance(app.screen, SessionPicker))
        await pilot.press("enter")
        await settled(app)

        assert "hook SessionStart failed: bad\\x1b[31m" in _notices(app) and not app._switching
        await type_and_send(pilot, "hi")
        await settled(app)
        assert app.agent.history[-1].role == "assistant"


async def test_without_session_start_hooks_no_worker_is_started_and_ctrl_c_right_at_launch_quits(
    launch: Launch, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = launch.app([_done()])
    groups: list[str | None] = []
    original = app.run_worker

    def spy(work, *args, **kwargs):
        groups.append(kwargs.get("group"))
        return original(work, *args, **kwargs)

    monkeypatch.setattr(app, "run_worker", spy)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")
        await until(pilot, lambda: not app.is_running)

    assert "session-start" not in groups


async def test_a_blocking_prompt_hook_still_shows_why_an_image_was_not_attached(launch: Launch) -> None:
    (launch.workspace / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 64)
    launch.write(launch.global_path, hooks={"UserPromptSubmit": [{"command": "gate"}]})
    launch.runner.script = [exit_run(2, "no")]
    app = launch.app([_done()])  # the fake catalog reports no vision
    async with app.run_test() as pilot:
        await type_and_send(pilot, "see @a.png")
        await settled(app)
        await pilot.pause()

        notices = _notices(app)
        assert scripted(app).requests == []
        assert "hook UserPromptSubmit (global) blocked: no" in notices
        assert "not attached a.png (model 'fake' has no vision)" in notices
        assert app.query_one(ChatInput).text == "see @a.png"
