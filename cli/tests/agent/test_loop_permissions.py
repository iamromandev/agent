"""Permission rules inside the loop: deny first, then allow, then this launch's "Always", then the dialog."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from src.agent.approval import Choice, Decision
from src.agent.events import AgentEvent, Notice, ToolFinished
from src.agent.loop import Agent
from src.agent.mcp.trust import TrustStore
from src.agent.mode import Mode
from src.agent.permissions.config import project_settings_path
from src.agent.permissions.setup import Rules, prepare
from src.agent.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.agent.terminal.manager import TerminalManager
from src.agent.tools import default_tools, read_only_tools
from src.agent.tools.base import ToolContext
from src.agent.web import FetchResponse

from tests.fakes import FakeFetcher, Gate, MemoryRecorder, RecordingApprover, ScriptedProvider

Script = list[list[ProviderEvent | Gate]]
URL = "https://example.com/docs"


def _call(call_id: str, name: str, **args: object) -> list[ProviderEvent | Gate]:
    return [ToolCallEvent(ToolCall(call_id, name, dict(args))), Done(10, 5)]


def _term_start(command: str) -> list[ProviderEvent | Gate]:
    return [ToolCallEvent(ToolCall("c1", "term_start", {"name": "t", "command": command})), Done(10, 5)]


def _done() -> list[ProviderEvent | Gate]:
    return [TextDelta("done"), Done(1, 1)]


class Setup:
    """A workspace with a global and a project settings file, prepared like a launch."""

    def __init__(self, ctx: ToolContext, tmp_path: Path) -> None:
        self.ctx = ctx
        self.global_path = tmp_path / "config" / "settings.json"
        self.store = TrustStore(tmp_path / "data" / "permission-trust.json")
        self.project_path = project_settings_path(ctx.workspace)

    def write(self, path: Path, allow: list[str] | None = None, deny: list[str] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        permissions = {k: v for k, v in (("allow", allow), ("deny", deny)) if v is not None}
        path.write_text(json.dumps({"permissions": permissions}))

    def rules(self, trust: bool = False) -> Rules:
        rules = prepare(self.ctx.workspace, self.global_path, self.store).rules
        if trust:
            rules.trust_project()
        return rules


def _setup(ctx: ToolContext, tmp_path: Path) -> Setup:
    return Setup(ctx, tmp_path)


def _agent(
    ctx: ToolContext,
    script: Script,
    rules: Rules | None,
    decisions: list[Decision] | None = None,
    *,
    fetcher: FakeFetcher | None = None,
    task: bool = False,
    terminals: TerminalManager | None = None,
) -> tuple[Agent, ScriptedProvider, RecordingApprover]:
    provider = ScriptedProvider(script, context_window=100_000)
    approver = RecordingApprover(decisions)
    agent = Agent(
        provider,
        default_tools(terminals, fetcher=fetcher),
        approver,
        ctx,
        25,
        today=lambda: date(2026, 11, 14),
        recorder=MemoryRecorder(),
        rules=rules,
        task_tools=read_only_tools(fetcher) if task else None,
    )
    return agent, provider, approver


async def _run(agent: Agent, text: str = "go") -> list[AgentEvent]:
    events: list[AgentEvent] = []
    await agent.run_turn(text, events.append)
    return events


def _tool_message(agent: Agent, call_id: str) -> str:
    return next(m.content for m in agent.history if m.role == "tool" and m.tool_call_id == call_id)


def _page() -> FetchResponse:
    return FetchResponse(URL, 200, "OK", "text/plain", None, b"hello", 5)


async def test_a_call_that_an_allow_rule_matches_runs_with_no_approval_request(
    ctx: ToolContext, tmp_path: Path
) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, allow=["bash(echo one)"])
    agent, _, approver = _agent(ctx, [_call("c1", "bash", command="echo one"), _done()], setup.rules())

    await _run(agent)

    assert approver.requests == []
    assert "one" in _tool_message(agent, "c1")


async def test_a_call_no_rule_matches_still_asks(ctx: ToolContext, tmp_path: Path) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, allow=["bash(echo one)"])
    agent, _, approver = _agent(
        ctx, [_call("c1", "bash", command="echo one two"), _done()], setup.rules(), [Decision(Choice.ONCE)]
    )

    await _run(agent)

    assert [r.label for r in approver.requests] == ["bash echo one two"]


async def test_a_prefix_rule_runs_the_matching_command_but_a_chained_one_still_asks(
    ctx: ToolContext, tmp_path: Path
) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, allow=["bash(echo hi*)"])
    script = [_call("c1", "bash", command="echo hi there"), _call("c2", "bash", command="echo hi; echo x"), _done()]
    agent, _, approver = _agent(ctx, script, setup.rules(), [Decision(Choice.DENY)])

    await _run(agent)

    assert [r.label for r in approver.requests] == ["bash echo hi; echo x"]
    assert "hi there" in _tool_message(agent, "c1")


async def test_a_deny_rule_refuses_without_a_dialog_and_names_the_rule_and_its_source(
    ctx: ToolContext, tmp_path: Path
) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, allow=["bash"])
    setup.write(setup.project_path, deny=["bash(touch*)"])
    agent, _, approver = _agent(ctx, [_call("c1", "bash", command="touch made"), _done()], setup.rules())

    events = await _run(agent)

    assert approver.requests == []
    assert not (ctx.workspace / "made").exists()
    assert _tool_message(agent, "c1") == "denied by rule bash(touch*) (project)"
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.result.is_error is True
    assert finished.applied is False


async def test_a_deny_prefix_refuses_the_denied_command_inside_a_chain_or_after_whitespace(
    ctx: ToolContext, tmp_path: Path
) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, allow=["bash"], deny=["bash(touch*)"])
    script = [
        _call("c1", "bash", command="echo hi; touch made"),
        _call("c2", "bash", command="  touch made"),
        _call("c3", "bash", command="echo $(touch made)"),
        _done(),
    ]
    agent, _, approver = _agent(ctx, script, setup.rules())

    await _run(agent)

    assert approver.requests == []
    assert not (ctx.workspace / "made").exists()
    assert [_tool_message(agent, c) for c in ("c1", "c2", "c3")] == ["denied by rule bash(touch*) (global)"] * 3


async def test_denying_every_bash_command_refuses_a_chain_too(ctx: ToolContext, tmp_path: Path) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, deny=["bash(*)"])
    agent, _, _ = _agent(ctx, [_call("c1", "bash", command="echo hi\n  echo there"), _done()], setup.rules())

    await _run(agent)

    assert _tool_message(agent, "c1") == "denied by rule bash(*) (global)"


async def test_a_deny_rule_for_term_start_refuses_a_chained_command(ctx: ToolContext, tmp_path: Path) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, deny=["term_start(rm*)"])
    script = [_term_start("echo x; rm y"), _done()]
    agent, _, approver = _agent(ctx, script, setup.rules(), terminals=TerminalManager(ctx.workspace))

    await _run(agent)

    assert approver.requests == []
    assert _tool_message(agent, "c1") == "denied by rule term_start(rm*) (global)"


async def test_an_allow_prefix_for_term_start_does_not_approve_a_chained_command(
    ctx: ToolContext, tmp_path: Path
) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, allow=["term_start(npm run dev*)"])
    script = [_term_start("npm run dev; rm y"), _done()]
    agent, _, approver = _agent(
        ctx, script, setup.rules(), [Decision(Choice.DENY)], terminals=TerminalManager(ctx.workspace)
    )

    await _run(agent)

    assert [r.tool for r in approver.requests] == ["term_start"]


async def test_a_deny_rule_also_refuses_a_tool_that_needs_no_approval(ctx: ToolContext, tmp_path: Path) -> None:
    (ctx.workspace / "a.txt").write_text("secret")
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, deny=["read"])
    agent, _, _ = _agent(ctx, [_call("c1", "read", path="a.txt"), _done()], setup.rules())

    await _run(agent)

    assert _tool_message(agent, "c1") == "denied by rule read (global)"


async def test_a_deny_rule_applies_in_plan_mode_too(ctx: ToolContext, tmp_path: Path) -> None:
    (ctx.workspace / "a.txt").write_text("secret")
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, deny=["read"])
    agent, _, _ = _agent(ctx, [_call("c1", "read", path="a.txt"), _done()], setup.rules())
    agent.set_mode(Mode.PLAN)

    await _run(agent)

    assert _tool_message(agent, "c1") == "denied by rule read (global)"


async def test_in_plan_mode_an_allow_rule_does_not_approve_bash_nor_show_a_hidden_tool(
    ctx: ToolContext, tmp_path: Path
) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, allow=["bash", "write"])
    script = [_call("c1", "bash", command="touch made"), _call("c2", "write", path="w.txt", content="x"), _done()]
    agent, provider, approver = _agent(ctx, script, setup.rules(), [Decision(Choice.DENY)])
    agent.set_mode(Mode.PLAN)

    await _run(agent)

    assert "write" not in provider.requests[0][1]  # an allow rule does not make a hidden tool visible
    assert "plan mode" in _tool_message(agent, "c2")
    assert [r.label for r in approver.requests] == ["bash touch made"]  # the rule did not approve bash
    assert not (ctx.workspace / "made").exists()
    assert not (ctx.workspace / "w.txt").exists()


async def test_in_plan_mode_an_allow_rule_does_not_approve_a_web_fetch(ctx: ToolContext, tmp_path: Path) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, allow=["web_fetch"])
    fetcher = FakeFetcher(_page())
    agent, provider, approver = _agent(
        ctx, [_call("c1", "web_fetch", url=URL), _done()], setup.rules(), [Decision(Choice.DENY)], fetcher=fetcher
    )
    agent.set_mode(Mode.PLAN)

    await _run(agent)

    assert "web_fetch" in provider.requests[0][1]
    assert len(approver.requests) == 1  # the rule did not approve it
    assert fetcher.calls == []


async def test_an_allow_rule_runs_a_web_fetch_in_build_mode(ctx: ToolContext, tmp_path: Path) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, allow=["web_fetch(https://example.com/*)"])
    fetcher = FakeFetcher(_page())
    agent, _, approver = _agent(ctx, [_call("c1", "web_fetch", url=URL), _done()], setup.rules(), fetcher=fetcher)

    await _run(agent)

    assert approver.requests == []
    assert fetcher.calls == [URL]


async def test_an_untrusted_project_allow_rule_does_not_approve_but_trusting_does(
    ctx: ToolContext, tmp_path: Path
) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.project_path, allow=["bash(echo one)"])
    asking, _, approver = _agent(
        ctx, [_call("c1", "bash", command="echo one"), _done()], setup.rules(), [Decision(Choice.DENY)]
    )
    await _run(asking)
    assert len(approver.requests) == 1

    trusting, _, approver = _agent(ctx, [_call("c1", "bash", command="echo one"), _done()], setup.rules(trust=True))
    await _run(trusting)
    assert approver.requests == []


async def test_rules_survive_a_reset_while_grants_do_not(ctx: ToolContext, tmp_path: Path) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, allow=["bash(echo one)"])
    script = [_call("c1", "bash", command="echo one"), _done(), _call("c2", "bash", command="echo two"), _done()]
    agent, _, approver = _agent(ctx, script, setup.rules(), [Decision(Choice.ALWAYS), Decision(Choice.DENY)])
    agent.grants.add("bash", {"command": "echo two"})

    agent.reset(recorder=MemoryRecorder())
    await _run(agent)  # the rule still covers `echo one`
    assert approver.requests == []
    await _run(agent)  # the reset forgot the grant for `echo two`
    assert [r.label for r in approver.requests] == ["bash echo two"]


async def test_editing_the_settings_file_during_the_run_changes_nothing_until_relaunch(
    ctx: ToolContext, tmp_path: Path
) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, allow=["bash(echo one)"])
    script = [_call("c1", "bash", command="echo one"), _done(), _call("c2", "bash", command="echo two"), _done()]
    agent, _, approver = _agent(ctx, script, setup.rules(), [Decision(Choice.DENY)])

    await _run(agent)
    setup.write(setup.global_path, allow=["bash"], deny=["bash(echo one)"])
    await _run(agent)

    assert "one" in _tool_message(agent, "c1")  # not denied by the new file
    assert [r.label for r in approver.requests] == ["bash echo two"]  # nor was the new allow read


async def test_a_deny_rule_on_read_also_stops_the_task_subagent(ctx: ToolContext, tmp_path: Path) -> None:
    (ctx.workspace / "a.txt").write_text("secret")
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, deny=["read"])
    script: Script = [
        _call("t1", "task", description="look", prompt="read a.txt"),
        _call("r1", "read", path="a.txt"),  # the child's request
        [TextDelta("it was refused"), Done(1, 1)],  # the child's report
        _done(),
    ]
    agent, provider, _ = _agent(ctx, script, setup.rules(), task=True)

    await _run(agent)

    child_request = provider.requests[2][0]
    refused = next(m for m in child_request if m.role == "tool")
    assert refused.content == "denied by rule read (global)"
    assert "secret" not in "".join(m.content for m in child_request)


async def test_a_task_subagent_uses_the_parents_allow_rules(ctx: ToolContext, tmp_path: Path) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, allow=["web_fetch"])
    fetcher = FakeFetcher(_page())
    script: Script = [
        _call("t1", "task", description="look", prompt="fetch it"),
        _call("w1", "web_fetch", url=URL),
        [TextDelta("fetched"), Done(1, 1)],
        _done(),
    ]
    agent, _, approver = _agent(ctx, script, setup.rules(), fetcher=fetcher, task=True)

    await _run(agent)

    assert approver.requests == []
    assert fetcher.calls == [URL]


async def test_without_rules_the_loop_behaves_as_before(ctx: ToolContext) -> None:
    agent, _, approver = _agent(
        ctx, [_call("c1", "bash", command="echo one"), _done()], None, [Decision(Choice.ONCE)]
    )

    await _run(agent)

    assert len(approver.requests) == 1
    assert approver.requests[0].allow_save is False


# "Always here"


def _notices(events: list[AgentEvent]) -> list[Notice]:
    return [e for e in events if isinstance(e, Notice)]


async def test_the_dialog_offers_always_here_when_the_rules_can_be_saved(ctx: ToolContext, tmp_path: Path) -> None:
    agent, _, approver = _agent(
        ctx, [_call("c1", "bash", command="echo one"), _done()], _setup(ctx, tmp_path).rules(), [Decision(Choice.ONCE)]
    )

    await _run(agent)

    assert approver.requests[0].allow_save is True
    assert approver.requests[0].allow_always is True


async def test_in_plan_mode_the_dialog_offers_neither_always_nor_always_here(ctx: ToolContext, tmp_path: Path) -> None:
    fetcher = FakeFetcher(_page())
    agent, _, approver = _agent(
        ctx,
        [_call("c1", "web_fetch", url=URL), _done()],
        _setup(ctx, tmp_path).rules(),
        [Decision(Choice.ONCE)],
        fetcher=fetcher,
    )
    agent.set_mode(Mode.PLAN)

    await _run(agent)

    assert (approver.requests[0].allow_always, approver.requests[0].allow_save) == (False, False)


async def test_always_here_saves_the_exact_rule_runs_the_call_and_the_next_one_does_not_ask(
    ctx: ToolContext, tmp_path: Path
) -> None:
    setup = _setup(ctx, tmp_path)
    script = [_call("c1", "bash", command="echo one"), _call("c2", "bash", command="echo one"), _done()]
    agent, _, approver = _agent(ctx, script, setup.rules(), [Decision(Choice.PROJECT)])

    events = await _run(agent)

    assert len(approver.requests) == 1
    assert "one" in _tool_message(agent, "c1")
    assert json.loads(setup.project_path.read_text())["permissions"]["allow"] == ["bash(echo one)"]
    assert _notices(events) == []
    assert setup.rules().untrusted_allow == ()  # the next launch: trusted, so no prompt


async def test_always_here_on_a_tool_without_a_key_argument_saves_the_bare_tool(
    ctx: ToolContext, tmp_path: Path
) -> None:
    setup = _setup(ctx, tmp_path)
    script = [_call("c1", "write", path="a.txt", content="x"), _done()]
    agent, _, _ = _agent(ctx, script, setup.rules(), [Decision(Choice.PROJECT)])

    await _run(agent)

    assert json.loads(setup.project_path.read_text())["permissions"]["allow"] == ["write"]


async def test_always_here_for_a_command_with_a_star_runs_it_this_run_only_and_warns(
    ctx: ToolContext, tmp_path: Path
) -> None:
    """`bash(echo x*)` would allow more than the one command that was approved, so nothing is saved."""
    setup = _setup(ctx, tmp_path)
    script = [_call("c1", "bash", command="echo x*"), _call("c2", "bash", command="echo x*"), _done()]
    agent, _, approver = _agent(ctx, script, setup.rules(), [Decision(Choice.PROJECT)])

    events = await _run(agent)

    assert len(approver.requests) == 1  # granted for this run
    assert not setup.project_path.exists()
    (notice,) = _notices(events)
    assert notice.level == "warn"
    assert "allowed for this run only" in notice.text


async def test_always_here_with_a_file_that_cannot_be_parsed_leaves_it_and_still_runs_the_call(
    ctx: ToolContext, tmp_path: Path
) -> None:
    setup = _setup(ctx, tmp_path)
    setup.project_path.parent.mkdir(parents=True)
    setup.project_path.write_text("{oops")
    script = [_call("c1", "bash", command="echo one"), _call("c2", "bash", command="echo one"), _done()]
    agent, _, approver = _agent(ctx, script, setup.rules(), [Decision(Choice.PROJECT)])

    events = await _run(agent)

    assert setup.project_path.read_text() == "{oops"
    assert len(approver.requests) == 1
    assert "one" in _tool_message(agent, "c1")
    (notice,) = _notices(events)
    assert notice.level == "warn"
    assert ".agent/settings.json" in notice.text
    assert "allowed for this run only" in notice.text


async def test_the_denial_text_escapes_the_rule_text_from_the_file(ctx: ToolContext, tmp_path: Path) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.project_path, deny=["bash(a\x1b[2J\nfake: all allowed)"])
    agent, _, _ = _agent(ctx, [_call("c1", "bash", command="a\x1b[2J\nfake: all allowed"), _done()], setup.rules())

    await _run(agent)

    text = _tool_message(agent, "c1")
    assert text == "denied by rule bash(a\\x1b[2J\\x0afake: all allowed) (project)"
    assert text.isprintable()


async def test_a_failed_save_notice_escapes_what_the_error_says(
    ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rules = _setup(ctx, tmp_path).rules()

    def broken(rule_text: str) -> None:
        raise OSError("cannot write /x/\x1b[2J.agent\nsettings.json")

    monkeypatch.setattr(rules, "save_project", broken)
    agent, _, _ = _agent(ctx, [_call("c1", "bash", command="echo one"), _done()], rules, [Decision(Choice.PROJECT)])

    events = await _run(agent)

    (notice,) = _notices(events)
    assert notice.text.isprintable()
    assert "\\x1b[2J.agent\\x0asettings.json" in notice.text


async def test_a_rule_saved_but_not_remembered_as_trusted_says_exactly_that(ctx: ToolContext, tmp_path: Path) -> None:
    setup = _setup(ctx, tmp_path)
    setup.store = TrustStore(ctx.workspace / "trust.json")  # refused: inside the workspace
    agent, _, _ = _agent(ctx, [_call("c1", "bash", command="echo one"), _done()], setup.rules(), [Decision(Choice.PROJECT)])

    events = await _run(agent)

    (notice,) = _notices(events)
    assert notice.level == "warn"
    assert notice.text.startswith("saved to .agent/settings.json but could not be remembered as trusted")
    assert notice.text.endswith("the next launch will ask")
    assert "allowed for this run only" not in notice.text
    assert json.loads(setup.project_path.read_text())["permissions"]["allow"] == ["bash(echo one)"]


# what the "Always here" option says it saves


async def test_the_request_carries_the_rule_it_would_save(ctx: ToolContext, tmp_path: Path) -> None:
    script = [_call("c1", "bash", command="echo one"), _call("c2", "write", path="a.txt", content="x"), _done()]
    agent, _, approver = _agent(ctx, script, _setup(ctx, tmp_path).rules(), [Decision(Choice.ONCE)] * 2)

    await _run(agent)

    assert [(r.save_rule, r.allow_save) for r in approver.requests] == [("bash(echo one)", True), ("write", True)]


async def test_the_option_is_not_offered_for_a_command_no_rule_can_say_exactly(ctx: ToolContext, tmp_path: Path) -> None:
    agent, _, approver = _agent(
        ctx, [_call("c1", "bash", command="ls *.py"), _done()], _setup(ctx, tmp_path).rules(), [Decision(Choice.ONCE)]
    )

    await _run(agent)

    assert (approver.requests[0].save_rule, approver.requests[0].allow_save) == (None, False)
    assert approver.requests[0].allow_always is True  # "Always" (per launch) is unaffected


# the settings files themselves


@pytest.mark.parametrize("target", [".agent/settings.json", "./.agent/../.agent/mcp.json", "{abs}/.agent/settings.json"])
async def test_a_write_to_an_agent_settings_file_always_shows_the_dialog_without_always_or_always_here(
    ctx: ToolContext, tmp_path: Path, target: str
) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, allow=["write", "edit"])
    path = target.replace("{abs}", str(ctx.workspace))
    script = [_call("c1", "write", path=path, content="{}"), _call("c2", "write", path=path, content="{}"), _done()]
    agent, _, approver = _agent(ctx, script, setup.rules(), [Decision(Choice.ALWAYS), Decision(Choice.PROJECT)])
    agent.grants.add("write", {})  # even a grant for the whole tool does not skip it

    await _run(agent)

    assert len(approver.requests) == 2  # asked both times: neither the rule, the grant nor "Always" covers it
    assert all((r.allow_always, r.allow_save, r.save_rule) == (False, False, None) for r in approver.requests)
    saved = ctx.workspace / ".agent" / "settings.json"
    assert not saved.exists() or "write" not in json.loads(saved.read_text()).get("permissions", {}).get("allow", [])


async def test_an_edit_of_the_mcp_file_asks_even_with_an_allow_rule(ctx: ToolContext, tmp_path: Path) -> None:
    mcp = ctx.workspace / ".agent" / "mcp.json"
    mcp.parent.mkdir()
    mcp.write_text('{"servers": {}}')
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, allow=["edit"])
    script = [_call("c1", "edit", path=".agent/mcp.json", old_string="{}", new_string='{"a": 1}'), _done()]
    agent, _, approver = _agent(ctx, script, setup.rules(), [Decision(Choice.DENY)])

    await _run(agent)

    assert [r.tool for r in approver.requests] == ["edit"]
    assert mcp.read_text() == '{"servers": {}}'


async def test_a_deny_rule_still_refuses_a_write_to_an_agent_settings_file(ctx: ToolContext, tmp_path: Path) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, deny=["write"])
    agent, _, approver = _agent(
        ctx, [_call("c1", "write", path=".agent/settings.json", content="{}"), _done()], setup.rules()
    )

    await _run(agent)

    assert approver.requests == []
    assert _tool_message(agent, "c1") == "denied by rule write (global)"


async def test_a_write_to_another_file_in_the_agent_folder_is_unaffected(ctx: ToolContext, tmp_path: Path) -> None:
    setup = _setup(ctx, tmp_path)
    setup.write(setup.global_path, allow=["write"])
    agent, _, approver = _agent(ctx, [_call("c1", "write", path=".agent/notes.md", content="x"), _done()], setup.rules())

    await _run(agent)

    assert approver.requests == []


async def test_the_notice_comes_before_the_calls_own_events(ctx: ToolContext, tmp_path: Path) -> None:
    setup = _setup(ctx, tmp_path)
    setup.project_path.parent.mkdir(parents=True)
    setup.project_path.write_text("{oops")
    agent, _, _ = _agent(ctx, [_call("c1", "bash", command="echo one"), _done()], setup.rules(), [Decision(Choice.PROJECT)])

    events = await _run(agent)

    kinds = [type(e).__name__ for e in events]
    assert kinds.index("Notice") < kinds.index("ToolFinished")
