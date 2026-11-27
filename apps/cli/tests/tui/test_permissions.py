"""Saved permission rules in the app: the trust prompt for a project's allow rules, `/permissions`, "Always here"."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from core.engine.mcp.trust import TrustStore, trust_store_path
from core.engine.permissions.config import RuleFile, project_settings_path
from core.engine.permissions.rules import parse_rule
from core.engine.permissions.setup import PermissionSetup, Rules, prepare
from core.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from core.testing.fakes import Gate
from src.tui.app import AgentApp
from src.tui.screens.permission_trust import PermissionTrustScreen
from src.tui.widgets.transcript import NoticeEntry
from textual.widgets import Label, OptionList

from tests.tui.conftest import AppFactory, type_and_send, until

Script = list[list[ProviderEvent | Gate]]


def _bash(call_id: str, command: str) -> list[ProviderEvent | Gate]:
    return [ToolCallEvent(ToolCall(call_id, "bash", {"command": command})), Done(1, 1)]


def _done() -> list[ProviderEvent | Gate]:
    return [TextDelta("done"), Done(1, 1)]


class Launch:
    """What `src.tui.run` wires for a folder: both settings files, the trust store and the app."""

    def __init__(self, make_app: AppFactory, tmp_path: Path) -> None:
        self.make_app = make_app
        self.workspace = tmp_path.resolve()
        self.store = TrustStore(trust_store_path(name="permission-trust.json"))  # under the test's XDG_DATA_HOME
        self.global_path = tmp_path / "config" / "settings.json"
        self.project_path = project_settings_path(self.workspace)
        self.setup: PermissionSetup

    def write(self, path: Path, allow: list[str] | None = None, deny: list[str] | None = None) -> bytes:
        path.parent.mkdir(parents=True, exist_ok=True)
        permissions = {k: v for k, v in (("allow", allow), ("deny", deny)) if v is not None}
        raw = json.dumps({"permissions": permissions}).encode()
        path.write_bytes(raw)
        return raw

    def project(self, allow: list[str] | None = None, deny: list[str] | None = None) -> bytes:
        return self.write(self.project_path, allow, deny)

    def app(self, script: Script | None = None, **kwargs: object) -> AgentApp:
        self.setup = prepare(self.workspace, self.global_path, self.store)
        return self.make_app(
            script, permissions=self.setup.rules, permission_warnings=self.setup.warnings, **kwargs
        )

    def rules(self) -> Rules:
        return prepare(self.workspace, self.global_path, self.store).rules


@pytest.fixture
def launch(make_app: AppFactory, tmp_path: Path) -> Launch:
    return Launch(make_app, tmp_path)


NO_RULES = RuleFile((), (), (), None, None)


def prepare_file(raw: bytes) -> RuleFile:
    return RuleFile((parse_rule("bash"),), (), (), raw, None)


def _notices(app: AgentApp) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _is_prompt(app: AgentApp) -> bool:
    """The permission trust prompt is fully mounted: its options have focus and "Don't trust" is highlighted."""
    screen = app.screen
    if not isinstance(screen, PermissionTrustScreen):
        return False
    options = screen.query(OptionList)
    return bool(options) and options.first().highlighted == 0 and screen.focused is options.first()


async def _wait_for_guard(pilot) -> None:
    await until(pilot, lambda: time.monotonic() >= pilot.app._keys_ignored_until)


async def _answer(pilot, option_id: str) -> None:
    await _wait_for_guard(pilot)
    options = pilot.app.screen.query_one(OptionList)
    options.highlighted = options.get_option_index(option_id)
    await pilot.press("enter")


def _tool_results(app: AgentApp) -> list[str]:
    from core.engine.provider.base import Message

    return [m.content for m in app.agent.history if isinstance(m, Message) and m.role == "tool"]


# the trust prompt


def _rule_labels(app: AgentApp) -> list[str]:
    return [str(label.render()) for label in app.screen.query_one("#rules").query(Label)]


async def test_the_title_counts_a_single_rule_in_the_singular(launch: Launch) -> None:
    launch.project(allow=["bash"])
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        assert "(1 allow rule)" in str(app.screen.query_one("#title").render())


async def test_a_trust_error_notice_escapes_what_the_error_says(make_app: AppFactory, tmp_path: Path) -> None:
    raw = b'{"permissions": {"allow": ["bash"]}}'
    file = prepare_file(raw)

    def refuse(content: bytes) -> None:
        raise OSError("no space \x1b[2J\nleft")

    app = make_app(permissions=Rules(NO_RULES, file, False, None, refuse))
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await _answer(pilot, "trust")
        await until(pilot, lambda: any("could not remember" in n for n in _notices(app)))
        (notice,) = [n for n in _notices(app) if "could not remember" in n]
        assert "\\x1b[2J\\x0aleft" in notice
        assert "\x1b" not in notice


# both prompts


# warnings and /permissions


async def test_permissions_without_rules_says_so(launch: Launch) -> None:
    app = launch.app()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/permissions")
        await until(pilot, lambda: any("no permission rules" in n for n in _notices(app)))


# Always here, end to end
