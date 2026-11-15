"""Saved permission rules in the app: the trust prompt for a project's allow rules, `/permissions`, "Always here"."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from src.agent.mcp.hub import McpHub
from src.agent.mcp.setup import prepare as prepare_mcp
from src.agent.mcp.trust import TrustStore, trust_store_path
from src.agent.permissions.config import RuleFile, project_settings_path
from src.agent.permissions.rules import parse_rule
from src.agent.permissions.setup import PermissionSetup, Rules, prepare
from src.agent.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.config import get_settings
from src.tui.app import HELP, AgentApp
from src.tui.screens.approval import ApprovalScreen
from src.tui.screens.mcp_trust import McpTrustScreen
from src.tui.screens.permission_trust import PermissionTrustScreen
from src.tui.widgets.transcript import NoticeEntry
from textual.widgets import Label, OptionList

from tests.fakes import FakeConnection, FakeConnector, Gate
from tests.tui.conftest import AppFactory, choose, settled, type_and_send, until

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
    from src.agent.provider.base import Message

    return [m.content for m in app.agent.history if isinstance(m, Message) and m.role == "tool"]


# the trust prompt


async def test_an_untrusted_project_file_with_allow_rules_asks_listing_every_allow_rule(launch: Launch) -> None:
    launch.project(allow=["bash(make test)", "bash(git status*)"], deny=["bash(rm*)"])
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        text = " ".join(str(label.render()) for label in app.screen.query(Label))
        assert "allow bash(make test)" in text
        assert "allow bash(git status*)" in text
        assert "bash(rm*)" not in text  # deny rules need no trust


def _rule_labels(app: AgentApp) -> list[str]:
    return [str(label.render()) for label in app.screen.query_one("#rules").query(Label)]


async def test_the_prompt_lists_the_broad_rules_first_and_counts_the_rules_in_its_title(launch: Launch) -> None:
    launch.project(allow=["bash(make test)", "bash(git*)", "web_fetch(https://a.org/x)", "write", "mcp__gh__issues"])
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        assert _rule_labels(app) == [
            "allow bash(git*)",
            "allow write",
            "allow mcp__gh__issues",
            "allow bash(make test)",
            "allow web_fetch(https://a.org/x)",
        ]
        title = str(app.screen.query_one("#title").render())
        assert "(5 allow rules)" in title


async def test_the_title_counts_a_single_rule_in_the_singular(launch: Launch) -> None:
    launch.project(allow=["bash"])
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        assert "(1 allow rule)" in str(app.screen.query_one("#title").render())


async def test_the_prompt_starts_on_dont_trust_and_ignores_keys_for_a_moment(launch: Launch) -> None:
    launch.project(allow=["bash"])
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        options = app.screen.query_one(OptionList)
        assert options.get_option_at_index(0).id == "decline"
        assert time.monotonic() < app._keys_ignored_until  # keys typed for the chat must not answer it
        await pilot.press("down", "enter")  # inside the guard window: ignored
        assert _is_prompt(app)


@pytest.mark.parametrize("key", ["escape", "enter"])
async def test_escape_declines_and_a_stray_enter_declines(launch: Launch, key: str) -> None:
    launch.project(allow=["bash"])
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await _wait_for_guard(pilot)
        await pilot.press(key)
        await until(pilot, lambda: not isinstance(app.screen, PermissionTrustScreen))
        await until(pilot, lambda: any("not trusted" in n for n in _notices(app)))
    assert launch.rules().untrusted_allow != ()


async def test_declining_keeps_the_allow_rules_off_so_a_call_still_asks_but_deny_rules_still_refuse(
    launch: Launch, tmp_path: Path
) -> None:
    launch.project(allow=["bash"], deny=["bash(touch refused*)"])
    app = launch.app([_bash("c1", "touch refused-file"), _bash("c2", "echo asked"), _done()])
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await _answer(pilot, "decline")
        await until(pilot, lambda: not isinstance(app.screen, PermissionTrustScreen))
        await type_and_send(pilot, "go")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))  # `echo asked` is not auto-approved
        assert not (tmp_path / "refused-file").exists()  # the deny rule refused the first call with no dialog
        await choose(pilot, "deny")
        await pilot.press("enter")
        await settled(app)
        assert _tool_results(app)[0] == "denied by rule bash(touch refused*) (project)"
        assert any("not trusted" in n and "off for this run" in n for n in _notices(app))


async def test_trusting_activates_the_rules_remembers_the_file_and_the_next_launch_does_not_ask(launch: Launch) -> None:
    raw = launch.project(allow=["bash(echo one)"])
    app = launch.app([_bash("c1", "echo one"), _done()])
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await _answer(pilot, "trust")
        await until(pilot, lambda: not isinstance(app.screen, PermissionTrustScreen))
        await type_and_send(pilot, "go")
        await settled(app)
        assert not isinstance(app.screen, ApprovalScreen)
        assert "one" in _tool_results(app)[0]
    assert launch.store.is_trusted(launch.workspace, raw)

    second = launch.app()
    async with second.run_test() as pilot:
        await pilot.pause(0.3)
        assert not isinstance(second.screen, PermissionTrustScreen)


async def test_editing_the_project_file_asks_again(launch: Launch) -> None:
    launch.project(allow=["bash(echo one)"])
    first = launch.app()
    async with first.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(first))
        await _answer(pilot, "trust")
        await until(pilot, lambda: not isinstance(first.screen, PermissionTrustScreen))

    launch.project(allow=["bash(echo one)", "bash(echo two)"])
    second = launch.app()
    async with second.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(second))


async def test_a_project_file_with_only_deny_rules_does_not_ask(launch: Launch) -> None:
    launch.project(deny=["bash(rm*)"])
    app = launch.app()
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        assert not isinstance(app.screen, PermissionTrustScreen)


async def test_a_trust_that_cannot_be_saved_still_activates_the_rules_and_warns(launch: Launch) -> None:
    launch.project(allow=["bash"])
    inside = TrustStore(launch.workspace / "trust.json")
    launch.store = inside  # a store in the workspace: refused for reading and writing
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await _answer(pilot, "trust")
        await until(pilot, lambda: any("inside the workspace" in n and "remember" in n for n in _notices(app)))
        assert launch.setup.rules.untrusted_allow == ()


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


async def test_a_workspace_dot_env_cannot_pre_trust_the_project_file(
    launch: Launch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store is where the process environment says, never where a file in the repository says."""
    import hashlib

    raw = launch.project(allow=["bash"])
    planted = tmp_path.parent / f"{tmp_path.name}-planted"  # outside the workspace
    (planted / "agent").mkdir(parents=True)
    (planted / "agent" / "permission-trust.json").write_text(
        json.dumps({str(launch.workspace): hashlib.sha256(raw).hexdigest()})
    )
    (tmp_path / ".env").write_text(f"XDG_DATA_HOME={planted}\nAGENT_SESSIONS_DIR={planted}\n")
    monkeypatch.delenv("AGENT_SESSIONS_DIR", raising=False)  # the test setup pins it; the `.env` must be what is read
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    try:
        launch.store = TrustStore(trust_store_path(name="permission-trust.json"))  # what `run` builds
        app = launch.app()
    finally:
        get_settings.cache_clear()
    assert not launch.store.path.is_relative_to(planted)
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))


async def test_markup_and_escape_codes_in_a_rule_are_shown_as_written(launch: Launch) -> None:
    launch.project(allow=["bash([bold red]x[/\x1b[2J*)"])
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        text = " ".join(str(label.render()) for label in app.screen.query(Label))
        assert "[bold red]x[/" in text
        assert "\x1b" not in text
        assert "\\x1b" in text


async def test_quitting_with_the_prompt_open_trusts_nothing(launch: Launch) -> None:
    launch.project(allow=["bash"])
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
    assert launch.rules().untrusted_allow != ()


# both prompts


async def test_the_mcp_prompt_and_the_permission_prompt_appear_one_after_the_other(
    make_app: AppFactory, tmp_path: Path
) -> None:
    workspace = tmp_path.resolve()
    launch = Launch(make_app, tmp_path)
    launch.project(allow=["bash"])
    (workspace / ".agent").mkdir(exist_ok=True)
    (workspace / ".agent" / "mcp.json").write_text(json.dumps({"servers": {"db": {"command": "psql"}}}))
    mcp_setup = prepare_mcp(workspace, tmp_path / "mcp-global.json", TrustStore(trust_store_path()))
    hub = McpHub(FakeConnector({"db": FakeConnection()}), workspace, mcp_setup.servers, 5.0, untrusted=mcp_setup.untrusted)
    setup = prepare(workspace, launch.global_path, launch.store)
    app = launch.make_app(mcp=hub, mcp_trust=mcp_setup.remember, permissions=setup.rules)
    seen: list[str] = []

    def record() -> None:
        stack = [type(s).__name__ for s in app.screen_stack if isinstance(s, McpTrustScreen | PermissionTrustScreen)]
        assert len(stack) <= 1, f"overlapping prompts: {stack}"
        if stack and (not seen or seen[-1] != stack[0]):
            seen.append(stack[0])

    async with app.run_test() as pilot:
        await until(pilot, lambda: (record(), isinstance(app.screen, McpTrustScreen))[1])
        await _wait_for_guard(pilot)
        await pilot.press("escape")  # decline the MCP servers
        await until(pilot, lambda: (record(), isinstance(app.screen, PermissionTrustScreen))[1])
        await _answer(pilot, "trust")
        await until(pilot, lambda: (record(), not isinstance(app.screen, PermissionTrustScreen))[1])

    assert seen == ["McpTrustScreen", "PermissionTrustScreen"]
    assert setup.rules.untrusted_allow == ()


# warnings and /permissions


async def test_the_startup_warnings_are_shown_as_notices(launch: Launch) -> None:
    launch.project_path.parent.mkdir(parents=True)
    launch.project_path.write_text("{oops")
    launch.write(launch.global_path, allow=["bash(a*b)"])
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: len([n for n in _notices(app) if "settings.json" in n]) == 2)
        notices = _notices(app)
        assert any("bash(a*b)" in n for n in notices)
        assert any("deny rules are not applied" in n for n in notices)
        await pilot.pause()


async def test_permissions_lists_every_rule_with_its_source_and_the_trust_state(launch: Launch) -> None:
    launch.write(launch.global_path, allow=["bash(make test)", "bash(a*b)"])
    launch.project(allow=["bash"], deny=["bash(rm*)"])
    app = launch.app()
    async with app.run_test() as pilot:
        await until(pilot, lambda: _is_prompt(app))
        await _answer(pilot, "decline")
        await until(pilot, lambda: not isinstance(app.screen, PermissionTrustScreen))
        await type_and_send(pilot, "/permissions")
        await until(pilot, lambda: any("allow bash(make test) (global)" in n for n in _notices(app)))
        (notice,) = [n for n in _notices(app) if "allow bash(make test) (global)" in n]
        assert notice.split(" ", 1)[1].splitlines() == [  # after the level icon
            "allow bash(make test) (global)",
            "invalid bash(a*b) (global): * only at the end",
            "allow bash (project) inactive: untrusted",
            "deny bash(rm*) (project)",
            "project file: untrusted",
        ]


async def test_permissions_without_rules_says_so(launch: Launch) -> None:
    app = launch.app()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/permissions")
        await until(pilot, lambda: any("no permission rules" in n for n in _notices(app)))


async def test_permissions_works_without_any_rules_object(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/permissions")
        await until(pilot, lambda: any("no permission rules" in n for n in _notices(app)))


async def test_help_mentions_permissions() -> None:
    assert "/permissions" in HELP


# Always here, end to end


async def test_always_here_saves_the_rule_and_the_next_identical_call_and_launch_do_not_ask(launch: Launch) -> None:
    script: Script = [_bash("c1", "echo one"), _bash("c2", "echo one"), _done()]
    app = launch.app(script)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await choose(pilot, "project")
        await settled(app)  # the second call ran with no dialog, or the worker would still wait
        assert [m for m in _tool_results(app) if "one" in m] != []
    assert json.loads(launch.project_path.read_text()) == {"permissions": {"allow": ["bash(echo one)"]}}

    second = launch.app([_bash("c3", "echo one"), _done()])
    async with second.run_test() as pilot:
        await pilot.pause(0.2)
        assert not isinstance(second.screen, PermissionTrustScreen)  # the saved file is trusted
        await type_and_send(pilot, "go")
        await settled(second)
        assert not isinstance(second.screen, ApprovalScreen)


async def test_always_here_with_an_unparseable_file_warns_leaves_the_file_and_runs_the_call(launch: Launch) -> None:
    launch.project_path.parent.mkdir(parents=True)
    launch.project_path.write_text("{oops")
    app = launch.app([_bash("c1", "echo one"), _done()])
    async with app.run_test() as pilot:
        await until(pilot, lambda: any("deny rules are not applied" in n for n in _notices(app)))
        await type_and_send(pilot, "go")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await choose(pilot, "project")
        await settled(app)
        await pilot.pause()
        assert any("could not save rule" in n and "this run only" in n for n in _notices(app))
        assert "one" in _tool_results(app)[0]
    assert launch.project_path.read_text() == "{oops"


async def test_global_rules_need_no_prompt_and_approve_calls(launch: Launch) -> None:
    launch.write(launch.global_path, allow=["bash(echo one)"])
    app = launch.app([_bash("c1", "echo one"), _done()])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await settled(app)
        assert not isinstance(app.screen, ApprovalScreen | PermissionTrustScreen)
        assert "one" in _tool_results(app)[0]


async def test_a_launch_without_any_rule_file_behaves_as_before(launch: Launch) -> None:
    app = launch.app([_bash("c1", "echo one"), _done()])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        assert not any(isinstance(s, PermissionTrustScreen) for s in app.screen_stack)
        await choose(pilot, "once")
        await settled(app)
    assert not launch.project_path.exists()
