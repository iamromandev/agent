"""`Hooks.run`: which entries run, how a run is read (exit codes, JSON), and what a decision carries. No processes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.engine.events import Notice
from src.engine.hooks.config import DEFAULT_TIMEOUT, Event, HookEntry
from src.engine.hooks.runner import REASON_CHARS, HookRun
from src.engine.hooks.setup import Hooks
from src.engine.permissions.rules import Source

from tests.fakes import FakeRunner, exit_run, ok_run

WORKSPACE = Path("/work/space")


def entry(command: str = "x", event: Event = "PreToolUse", matcher: str | None = None, source: Source = "global") -> HookEntry:
    return HookEntry(event, command, matcher, DEFAULT_TIMEOUT, source)


def make(runner: FakeRunner, *entries: HookEntry, active: bool = True) -> Hooks:
    return Hooks(entries, [], runner, lambda: active, WORKSPACE)


async def test_stdin_is_the_event_object_and_the_environment_names_the_event_and_project() -> None:
    runner = FakeRunner()
    hooks = make(runner, entry("cmd"))

    await hooks.run("PreToolUse", {"tool": "edit", "args": {"path": "a"}}, tool="edit")

    assert runner.calls == [
        ("cmd", {"event": "PreToolUse", "tool": "edit", "args": {"path": "a"}},
         {"AGENT_HOOK_EVENT": "PreToolUse", "AGENT_PROJECT_DIR": str(WORKSPACE)})
    ]


async def test_exit_2_blocks_with_the_stderr_as_the_reason() -> None:
    hooks = make(FakeRunner([exit_run(2, "  nope\n")]), entry())

    decision = await hooks.run("PreToolUse", {}, tool="edit")

    assert (decision.blocked, decision.reason) == (True, "nope")
    assert [o.outcome for o in decision.outcomes] == ["blocked"]


async def test_exit_2_with_empty_stderr_blocks_with_the_default_reason() -> None:
    decision = await make(FakeRunner([exit_run(2)]), entry()).run("PreToolUse", {}, tool="edit")

    assert (decision.blocked, decision.reason) == (True, "blocked by hook")


async def test_a_long_reason_is_cut() -> None:
    decision = await make(FakeRunner([exit_run(2, "x" * 5000)]), entry()).run("PreToolUse", {}, tool="edit")

    assert len(decision.reason) == REASON_CHARS


async def test_other_exit_codes_warn_and_do_not_block() -> None:
    decision = await make(FakeRunner([exit_run(1, "lint failed")]), entry()).run("PreToolUse", {}, tool="edit")

    assert not decision.blocked
    [outcome] = decision.outcomes
    assert outcome.outcome == "warning" and outcome.text == "exit 1: lint failed"


async def test_a_missing_command_warns() -> None:
    decision = await make(FakeRunner([exit_run(127, "sh: x: not found")]), entry()).run("PreToolUse", {}, tool="edit")

    assert not decision.blocked and [o.outcome for o in decision.outcomes] == ["warning"]


async def test_a_start_error_warns_with_its_text() -> None:
    run = HookRun(None, "", "", False, "could not start: no shell")

    decision = await make(FakeRunner([run]), entry()).run("PreToolUse", {}, tool="edit")

    [outcome] = decision.outcomes
    assert not decision.blocked and (outcome.outcome, outcome.text) == ("warning", "could not start: no shell")


async def test_a_timeout_is_its_own_outcome_and_does_not_block() -> None:
    run = HookRun(None, "", "", True, None)

    decision = await make(FakeRunner([run]), entry()).run("PreToolUse", {}, tool="edit")

    [outcome] = decision.outcomes
    assert not decision.blocked and outcome.outcome == "timeout" and "60" in outcome.text


async def test_stdout_json_with_decision_block_blocks_with_its_reason() -> None:
    out = json.dumps({"decision": "block", "reason": "x"})

    decision = await make(FakeRunner([ok_run(out)]), entry()).run("PreToolUse", {}, tool="edit")

    assert (decision.blocked, decision.reason) == (True, "x")


async def test_decision_allow_does_not_block_and_exit_2_wins_over_it() -> None:
    allow = json.dumps({"decision": "allow"})

    allowed = await make(FakeRunner([ok_run(allow)]), entry()).run("PreToolUse", {}, tool="edit")
    both = await make(FakeRunner([exit_run(2, "no", allow)]), entry()).run("PreToolUse", {}, tool="edit")

    assert not allowed.blocked and [o.outcome for o in allowed.outcomes] == ["ok"]
    assert (both.blocked, both.reason) == (True, "no")


@pytest.mark.parametrize("stdout", ["not json", "[1, 2]", '"str"', "42", "null", '{"decision": ', '{"context": 5}', "{}", ""])
async def test_stdout_that_is_not_a_json_object_with_usable_fields_is_ignored(stdout: str) -> None:
    decision = await make(FakeRunner([ok_run(stdout)]), entry()).run("PreToolUse", {}, tool="edit")

    assert not decision.blocked and decision.context == [] and [o.outcome for o in decision.outcomes] == ["ok"]


async def test_context_is_collected_in_run_order() -> None:
    runner = FakeRunner([ok_run(json.dumps({"context": "one"})), ok_run(), ok_run(json.dumps({"context": "two"}))])
    hooks = make(runner, entry("a", "PostToolUse"), entry("b", "PostToolUse"), entry("c", "PostToolUse"))

    decision = await hooks.run("PostToolUse", {}, tool="edit")

    assert decision.context == ["one", "two"]
    assert [o.outcome for o in decision.outcomes] == ["context", "ok", "context"]


async def test_the_first_block_ends_a_blocking_event() -> None:
    runner = FakeRunner([exit_run(2, "first"), exit_run(2, "second")])
    hooks = make(runner, entry("a"), entry("b"))

    decision = await hooks.run("PreToolUse", {}, tool="edit")

    assert [c[0] for c in runner.calls] == ["a"]
    assert decision.reason == "first" and len(decision.outcomes) == 1


@pytest.mark.parametrize("event", ["UserPromptSubmit", "Stop"])
async def test_stop_and_prompt_events_block_too(event: Event) -> None:
    runner = FakeRunner([exit_run(2, "no"), ok_run()])
    hooks = make(runner, entry("a", event), entry("b", event))

    decision = await hooks.run(event, {})

    assert decision.blocked and len(runner.calls) == 1


async def test_posttooluse_exit_2_is_context_not_a_block_and_later_hooks_still_run() -> None:
    runner = FakeRunner([exit_run(2, "E1: bad"), ok_run(json.dumps({"context": "more"}))])
    hooks = make(runner, entry("a", "PostToolUse"), entry("b", "PostToolUse"))

    decision = await hooks.run("PostToolUse", {}, tool="edit")

    assert not decision.blocked and decision.context == ["E1: bad", "more"] and len(runner.calls) == 2


async def test_posttooluse_decision_block_is_its_reason_as_context() -> None:
    out = json.dumps({"decision": "block", "reason": "fix it"})

    decision = await make(FakeRunner([ok_run(out)]), entry(event="PostToolUse")).run("PostToolUse", {}, tool="edit")

    assert not decision.blocked and decision.context == ["fix it"]


async def test_posttooluse_decision_block_with_a_reason_and_a_context_joins_both() -> None:
    out = json.dumps({"decision": "block", "reason": "fix it", "context": "E1 on line 3"})

    decision = await make(FakeRunner([ok_run(out)]), entry(event="PostToolUse")).run("PostToolUse", {}, tool="edit")

    assert not decision.blocked and decision.context == ["fix it\nE1 on line 3"]


async def test_pretooluse_context_is_unused_and_shows_as_a_warning() -> None:
    out = json.dumps({"context": "remember this"})
    hooks = make(FakeRunner([ok_run(out)]), entry())

    decision = await hooks.run("PreToolUse", {}, tool="edit")

    assert decision.context == [] and not decision.blocked
    [outcome] = decision.outcomes
    assert outcome.outcome == "warning" and "PreToolUse context is unused" in outcome.text
    assert hooks.notices(decision)[0].level == "warn"


async def test_a_pretooluse_block_still_blocks_when_it_also_has_a_context() -> None:
    out = json.dumps({"decision": "block", "reason": "no", "context": "why"})

    decision = await make(FakeRunner([ok_run(out)]), entry()).run("PreToolUse", {}, tool="edit")

    assert decision.blocked and decision.reason == "no" and decision.context == []


async def test_a_sessionstart_block_is_a_warning_and_nothing_else() -> None:
    hooks = make(FakeRunner([exit_run(2, "no")]), entry(event="SessionStart"))

    decision = await hooks.run("SessionStart", {"source": "startup"})

    assert not decision.blocked and decision.context == []
    [outcome] = decision.outcomes
    assert outcome.outcome == "warning" and "cannot block" in outcome.text


async def test_a_runner_that_raises_is_a_warning_never_an_exception() -> None:
    def boom(command: str, payload: dict) -> HookRun:
        raise RuntimeError("kaput")

    decision = await make(FakeRunner(boom), entry()).run("PreToolUse", {}, tool="edit")

    [outcome] = decision.outcomes
    assert not decision.blocked and outcome.outcome == "warning" and "kaput" in outcome.text


async def test_global_entries_run_before_project_ones_each_in_file_order() -> None:
    runner = FakeRunner()
    hooks = make(runner, entry("p1", source="project"), entry("g1"), entry("p2", source="project"), entry("g2"))

    await hooks.run("PreToolUse", {}, tool="edit")

    assert [c[0] for c in runner.calls] == ["g1", "g2", "p1", "p2"]


async def test_project_entries_are_never_spawned_while_inactive() -> None:
    active = False
    runner = FakeRunner()
    hooks = Hooks([entry("g"), entry("p", source="project")], [], runner, lambda: active, WORKSPACE)

    assert hooks.has("PreToolUse", "edit")
    await hooks.run("PreToolUse", {}, tool="edit")
    assert [c[0] for c in runner.calls] == ["g"]
    assert [e.command for e in hooks.untrusted()] == ["p"]

    active = True
    await hooks.run("PreToolUse", {}, tool="edit")
    assert [c[0] for c in runner.calls] == ["g", "g", "p"]
    assert hooks.untrusted() == ()


async def test_only_entries_of_the_event_and_matching_the_tool_run() -> None:
    runner = FakeRunner()
    hooks = make(
        runner, entry("pre-edit", matcher="edit"), entry("pre-all"), entry("post", "PostToolUse"), entry("pre-write", matcher="write")
    )

    await hooks.run("PreToolUse", {}, tool="edit")

    assert [c[0] for c in runner.calls] == ["pre-edit", "pre-all"]


def test_has_is_false_without_active_matching_entries() -> None:
    inactive = Hooks([entry("p", matcher="edit", source="project")], [], FakeRunner(), lambda: False, WORKSPACE)
    hooks = make(FakeRunner(), entry(matcher="edit"), entry(event="Stop"))

    assert hooks.has("PreToolUse", "edit") and not hooks.has("PreToolUse", "write")
    assert hooks.has("Stop") and not hooks.has("PostToolUse", "edit")
    assert not inactive.has("PreToolUse", "edit")
    assert not Hooks.empty().has("PreToolUse", "edit")


def test_empty_has_no_entries_and_no_warnings() -> None:
    hooks = Hooks.empty()

    assert hooks.entries == () and hooks.warnings == () and hooks.untrusted() == ()


async def test_notices_name_event_source_outcome_and_the_escaped_cut_text() -> None:
    runner = FakeRunner([exit_run(2, "bad\x1b[31m" + "y" * 500), exit_run(1, "oops"), ok_run(), HookRun(None, "", "", True, None)])
    hooks = make(runner, entry("a", event="PostToolUse"), entry("b", event="PostToolUse", source="project"),
                 entry("c", event="PostToolUse"), entry("d", event="PostToolUse", source="project"))

    decision = await hooks.run("PostToolUse", {}, tool="edit")  # runs a, c (global), then b, d (project)
    notices = hooks.notices(decision)

    assert [n.level for n in notices] == ["info", "warn", "warn"]
    assert notices[0].text.startswith("hook PostToolUse (global) context: bad\\x1b[31m")
    assert "\x1b" not in notices[0].text and len(notices[0].text) < 260
    assert notices[1] == Notice("warn", "hook PostToolUse (global) warning: exit 1: oops")
    assert notices[2].text.startswith("hook PostToolUse (project) timeout")


async def test_notices_redact_the_text_before_it_is_escaped_and_cut() -> None:
    token = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
    hooks = make(FakeRunner([exit_run(2, "x" * 190 + token)]), entry())
    decision = await hooks.run("PreToolUse", {}, tool="edit")

    [notice] = hooks.notices(decision, lambda text: text.replace(token, "[redacted]"))

    assert token not in notice.text and "[redacted]" in notice.text  # redacted whole, then cut


async def test_a_notice_is_cut_before_it_is_escaped_so_an_escape_is_never_split() -> None:
    hooks = make(FakeRunner([exit_run(2, "\x1b" * 500)]), entry())
    decision = await hooks.run("PreToolUse", {}, tool="edit")

    [notice] = hooks.notices(decision)

    text = notice.text.split(": ", 1)[1]
    assert text == "\\x1b" * 200


async def test_a_blocked_notice_is_shown_and_ok_ones_are_not() -> None:
    hooks = make(FakeRunner([exit_run(2, "nope")]), entry())

    decision = await hooks.run("PreToolUse", {}, tool="edit")

    assert hooks.notices(decision) == [Notice("info", "hook PreToolUse (global) blocked: nope")]
    assert hooks.notices(await make(FakeRunner(), entry()).run("PreToolUse", {}, tool="edit")) == []


def test_the_listing_shows_every_entry_with_its_source_inactive_ones_and_the_warnings() -> None:
    entries = [
        entry("lint\x1b[31m", "PostToolUse", "edit"),
        entry("all", "PreToolUse", source="project"),
        entry("bye", "Stop"),
    ]
    hooks = Hooks(entries, ["hooks (global): unknown event"], FakeRunner(), lambda: False, WORKSPACE)

    assert hooks.listing() == [
        "PostToolUse [edit] (global, 60s) lint\\x1b[31m",
        "PreToolUse [*] (project, 60s, inactive: untrusted) all",
        "Stop [*] (global, 60s) bye",
        "hooks (global): unknown event",
    ]


def test_the_listing_of_trusted_project_hooks_has_no_inactive_mark_and_none_means_empty() -> None:
    active = Hooks([entry("p", source="project")], [], FakeRunner(), lambda: True, WORKSPACE)

    assert active.listing() == ["PreToolUse [*] (project, 60s) p"]
    assert Hooks.empty().listing() == []
