"""Hooks in the app: the shared trust prompt, project hooks off until trusted, `/hooks`, the transcript lines."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from src.engine.hooks.setup import Hooks
from src.engine.hooks.setup import prepare as prepare_hooks
from src.engine.mcp.hub import McpHub
from src.engine.mcp.setup import prepare as prepare_mcp
from src.engine.mcp.trust import TrustStore, trust_store_path
from src.engine.permissions.config import project_settings_path
from src.engine.permissions.setup import PermissionSetup, prepare
from src.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.tui.app import HELP, AgentApp
from src.tui.screens.mcp_trust import McpTrustScreen
from src.tui.screens.permission_trust import PermissionTrustScreen
from src.tui.widgets.transcript import NoticeEntry
from textual.widgets import Label, OptionList

from tests.fakes import FakeConnection, FakeConnector, FakeRunner, Gate, exit_run
from tests.tui.conftest import AppFactory, settled, type_and_send, until

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


async def test_ac3_after_trust_a_hook_runs_and_the_next_launch_does_not_ask(launch: Launch) -> None:
    raw = launch.project(hooks=PRE)
    launch.runner.script = [exit_run(2, "no reads today")]
    app = launch.app([_read(), _done()])
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        assert launch.runner.calls == []
        await _answer(pilot, "trust")
        await until(pilot, lambda: not isinstance(app.screen, PermissionTrustScreen))
        await type_and_send(pilot, "go")
        await settled(app)

        assert [c[0] for c in launch.runner.calls] == ["check-read"]
        assert _tool_results(app) == ["blocked by hook: no reads today"]
        assert any(n == "hook PreToolUse (project) blocked: no reads today" for n in _notices(app))
    assert launch.store.is_trusted(launch.workspace, raw)

    second = launch.app()
    async with second.run_test() as pilot:
        await pilot.pause(0.3)
        assert not isinstance(second.screen, PermissionTrustScreen)


async def test_ac3_editing_the_file_asks_again(launch: Launch) -> None:
    launch.project(hooks=PRE)
    first = launch.app()
    async with first.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(first))
        await _answer(pilot, "trust")
        await until(pilot, lambda: not isinstance(first.screen, PermissionTrustScreen))

    launch.project(hooks={"PreToolUse": [{"command": "changed"}]})
    second = launch.app()
    async with second.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(second))


async def test_global_hooks_need_no_trust(launch: Launch) -> None:
    launch.write(launch.global_path, hooks=PRE)
    launch.runner.script = [exit_run(0)]
    app = launch.app([_read(), _done()])
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        assert not isinstance(app.screen, PermissionTrustScreen)
        await type_and_send(pilot, "go")
        await settled(app)
    assert [c[0] for c in launch.runner.calls] == ["check-read"]


async def test_ac4_one_screen_covers_allow_rules_and_hooks(launch: Launch) -> None:
    launch.project(permissions={"allow": ["bash(ls)"]}, hooks=PRE)
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        text = "\n".join(_labels(app))
        assert "allow bash(ls)" in text and "hook PreToolUse [read] (timeout 60s) check-read" in text
        assert "1 allow rule, 1 hook" in text
        await _answer(pilot, "trust")
        await until(pilot, lambda: not isinstance(app.screen, PermissionTrustScreen))
        await pilot.pause(0.3)
        assert not isinstance(app.screen, PermissionTrustScreen)  # no second screen
    assert launch.setup.rules.untrusted_allow == () and launch.hooks.untrusted() == ()


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


async def test_ac12_the_hooks_command_lists_entries_sources_inactive_ones_and_warnings(launch: Launch) -> None:
    launch.write(launch.global_path, hooks={"PostToolUse": [{"matcher": "edit", "command": "lint"}], "Bogus": []})
    launch.project(hooks=PRE)
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await _answer(pilot, "decline")
        await until(pilot, lambda: not isinstance(app.screen, PermissionTrustScreen))
        await type_and_send(pilot, "/hooks")
        await until(pilot, lambda: any(n.startswith("PostToolUse") for n in _notices(app)))

        listing = next(n for n in _notices(app) if n.startswith("PostToolUse"))
        assert listing.splitlines() == [
            "PostToolUse [edit] (global, 60s) lint",
            "PreToolUse [read] (project, 60s, inactive: untrusted) check-read",
            'hooks (global): unknown event "Bogus"; its entries are skipped',
        ]


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


async def test_ac12_a_warning_and_a_timeout_are_lines_in_the_transcript(launch: Launch) -> None:
    from src.engine.hooks.runner import HookRun

    launch.write(launch.global_path, hooks={"PreToolUse": [{"command": "a"}, {"command": "b"}]})
    launch.runner.script = [exit_run(1, "oops\x1b[2J"), HookRun(None, "", "", True, None)]
    app = launch.app([_read(), _done()])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await settled(app)
        notices = _notices(app)

    assert "hook PreToolUse (global) warning: exit 1: oops\\x1b[2J" in notices
    assert any(n.startswith("hook PreToolUse (global) timeout") for n in notices)
    assert "hello" in _tool_results(app)[0]  # neither blocks
