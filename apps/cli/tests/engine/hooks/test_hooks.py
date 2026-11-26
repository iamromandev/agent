"""`Hooks.run`: which entries run, how a run is read (exit codes, JSON), and what a decision carries. No processes."""

from __future__ import annotations

import json
from pathlib import Path

from src.engine.hooks.config import DEFAULT_TIMEOUT, Event, HookEntry
from src.engine.hooks.runner import HookRun
from src.engine.hooks.setup import Hooks
from src.engine.permissions.rules import Source

from tests.fakes import FakeRunner, exit_run, ok_run

WORKSPACE = Path("/work/space")


def entry(command: str = "x", event: Event = "PreToolUse", matcher: str | None = None, source: Source = "global") -> HookEntry:
    return HookEntry(event, command, matcher, DEFAULT_TIMEOUT, source)


def make(runner: FakeRunner, *entries: HookEntry, active: bool = True) -> Hooks:
    return Hooks(entries, [], runner, lambda: active, WORKSPACE)


async def test_global_hooks_stop_running_when_the_global_gate_closes() -> None:
    runner = FakeRunner([ok_run(), ok_run()])
    gate = {"on": True}
    hooks = Hooks((entry("g", source="global"), entry("p", source="project")), [], runner, lambda: True, WORKSPACE, global_active=lambda: gate["on"])

    await hooks.run("PreToolUse", {}, tool="edit")
    gate["on"] = False
    assert hooks.has("PreToolUse", "edit")  # the project entry is still active
    await hooks.run("PreToolUse", {}, tool="edit")

    assert [c[0] for c in runner.calls] == ["g", "p", "p"]


async def test_other_exit_codes_warn_and_do_not_block() -> None:
    decision = await make(FakeRunner([exit_run(1, "lint failed")]), entry()).run("PreToolUse", {}, tool="edit")

    assert not decision.blocked
    [outcome] = decision.outcomes
    assert outcome.outcome == "warning" and outcome.text == "exit 1: lint failed"


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


async def test_posttooluse_decision_block_is_its_reason_as_context() -> None:
    out = json.dumps({"decision": "block", "reason": "fix it"})

    decision = await make(FakeRunner([ok_run(out)]), entry(event="PostToolUse")).run("PostToolUse", {}, tool="edit")

    assert not decision.blocked and decision.context == ["fix it"]


async def test_pretooluse_context_is_unused_and_shows_as_a_warning() -> None:
    out = json.dumps({"context": "remember this"})
    hooks = make(FakeRunner([ok_run(out)]), entry())

    decision = await hooks.run("PreToolUse", {}, tool="edit")

    assert decision.context == [] and not decision.blocked
    [outcome] = decision.outcomes
    assert outcome.outcome == "warning" and "PreToolUse context is unused" in outcome.text
    assert hooks.notices(decision)[0].level == "warn"


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
