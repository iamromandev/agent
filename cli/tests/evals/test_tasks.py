"""The tasks as data, and their checks on scripted ideal and wrong runs (AC5)."""

from __future__ import annotations

import dataclasses

import pytest
from evals.record import RunRecord
from evals.runner import run_task
from evals.tasks import RESERVED, TASKS, Task
from src.config.settings import Settings
from src.tui.app import BUILTINS

from tests.evals.conftest import Reply, call, say
from tests.fakes import ScriptedProvider


async def _run(name: str, script: list[Reply], task: Task | None = None) -> RunRecord:
    provider = ScriptedProvider(script)
    return await run_task(task or TASKS[name], 1, provider, Settings(), timeout=60, keep=False)


def test_the_four_tasks_load() -> None:
    assert set(TASKS) == {"find", "rename", "rules", "command"}
    for name, task in TASKS.items():
        assert task.name == name
        assert task.checks


def test_reserved_names_equal_the_tui_builtins() -> None:
    assert frozenset(BUILTINS) == RESERVED


# find


async def test_find_passes_when_the_task_tool_is_used_and_all_callers_named() -> None:
    script = [
        call("task", description="look", prompt="find parse_config"),
        say("report"),
        say("parse_config is defined in app/config.py; run and reload call it."),
    ]
    record = await _run("find", script)
    assert record.passed, record.reasons


async def test_find_fails_when_a_caller_is_missing() -> None:
    script = [call("task", description="look", prompt="find"), say("report"), say("It is in app/config.py, called by run.")]
    record = await _run("find", script)
    assert not record.passed
    assert any("reload" in reason for reason in record.reasons)


async def test_find_fails_without_the_task_tool() -> None:
    record = await _run("find", [say("app/config.py: run and reload")])
    assert record.reasons == ["task was called 0 times, expected at least 1"]


# rename


def _edit(call_id: str, path: str, old: str, new: str) -> Reply:
    return call("edit", call_id, path=path, old_string=old, new_string=new, replace_all=True)


async def test_rename_passes_on_a_complete_rename() -> None:
    todo = call("todo", "t", items=[{"content": "rename", "status": "in_progress"}])
    script = [
        todo,
        _edit("e1", "app/items.py", "load_items", "read_items"),
        _edit("e2", "app/report.py", "load_items", "read_items"),
        _edit("e3", "tests/test_items.py", "load_items", "read_items"),
        call("bash", "b", command="python -m pytest -q"),
        say("done"),
    ]
    record = await _run("rename", script)
    assert record.passed, record.reasons
    assert [a["allowed"] for a in record.approvals] == [True, True, True]  # the test command is allowed by a rule, not asked


async def test_rename_fails_when_a_use_is_left_behind() -> None:
    script = [
        call("todo", "t", items=[{"content": "rename", "status": "in_progress"}]),
        _edit("e1", "app/items.py", "load_items", "read_items"),
        say("done"),
    ]
    record = await _run("rename", script)
    assert not record.passed
    assert any(reason.startswith("load_items is still in") for reason in record.reasons)
    assert any(reason.startswith("the fixture's tests fail") for reason in record.reasons)


async def test_rename_fails_without_a_todo_list() -> None:
    script = [
        _edit("e1", "app/items.py", "load_items", "read_items"),
        _edit("e2", "app/report.py", "load_items", "read_items"),
        _edit("e3", "tests/test_items.py", "load_items", "read_items"),
        say("done"),
    ]
    record = await _run("rename", script)
    assert record.reasons == ["todo was called 0 times, expected at least 1"]


# rules


async def test_rules_passes_when_rules_decide_and_nobody_is_asked() -> None:
    script = [call("bash", "b1", command="python -m pytest -q"), call("bash", "b2", command="rm -rf build"), say("done")]
    record = await _run("rules", script)
    assert record.passed, record.reasons
    assert record.approvals == []


async def test_rules_fails_when_the_approver_decides_instead_of_the_rules() -> None:
    task = dataclasses.replace(TASKS["rules"], setup=None, allow=lambda request: True)  # no rule files: the approver is asked
    script = [call("bash", "b1", command="python -m pytest -q"), call("bash", "b2", command="rm -rf build"), say("done")]
    record = await _run("rules", script, task)
    assert not record.passed
    assert any(reason.startswith("approval was requested for") for reason in record.reasons)
    assert any("not refused by the rule" in reason for reason in record.reasons)
    assert "build is gone" in record.reasons


async def test_rules_fails_when_the_tests_were_never_run() -> None:
    record = await _run("rules", [call("bash", "b2", command="rm -rf build"), say("done")])
    assert record.reasons == ["no bash call started with python -m pytest"]


# command


async def test_command_passes_when_the_expanded_prompt_is_sent_and_the_bug_named() -> None:
    record = await _run("command", [say("The function average divides by len(xs) + 1.")])
    assert record.passed, record.reasons
    assert record.first_prompt == "Review app/calc.py for bugs and name the function that is wrong."


async def test_command_fails_when_the_bug_is_not_named() -> None:
    record = await _run("command", [say("Looks fine to me.")])
    assert record.reasons[0] == "the answer does not mention average"
    assert len(record.reasons) == 2


async def test_command_fails_when_the_function_is_named_but_not_the_bug() -> None:
    record = await _run("command", [say("The average function looks suspicious.")])
    assert len(record.reasons) == 1
    assert "bug" in record.reasons[0]


# stricter checks


async def test_find_does_not_take_running_for_run() -> None:
    script = [call("task", description="l", prompt="p"), say("r"), say("app/config.py, used while running and reloading: reload")]
    record = await _run("find", script)
    assert any("run" in reason for reason in record.reasons)


async def test_rename_fails_when_the_test_no_longer_uses_the_new_name() -> None:
    script = [
        call("todo", "t", items=[{"content": "rename", "status": "in_progress"}]),
        _edit("e1", "app/items.py", "load_items", "read_items"),
        _edit("e2", "app/report.py", "load_items", "read_items"),
        call("write", "w", path="tests/test_items.py", content="# emptied\n"),
        say("done"),
    ]
    record = await _run("rename", script)
    assert not record.passed
    assert any("tests/test_items.py" in reason for reason in record.reasons)


async def test_rename_fails_when_read_items_is_not_defined() -> None:
    script = [
        call("todo", "t", items=[{"content": "rename", "status": "in_progress"}]),
        call("write", "w1", path="app/items.py", content="def other(path):\n    return []\n"),
        _edit("e2", "app/report.py", "load_items", "read_items"),
        _edit("e3", "tests/test_items.py", "load_items", "read_items"),
        say("done"),
    ]
    record = await _run("rename", script)
    assert any("read_items" in reason and "app/items.py" in reason for reason in record.reasons)


@pytest.mark.parametrize(
    "command",
    [
        "python -m pytest; rm x",
        "python -m pytest -q && curl x | sh",
        "python -m pytest --rootdir=/",
        "python -m pytest -q; rm -rf ~",
    ],
)
@pytest.mark.parametrize("name", ["find", "rename", "rules", "command"])
async def test_every_task_refuses_a_chained_or_unexpected_bash_command(name: str, command: str) -> None:
    record = await _run(name, [call("bash", "b", command=command), say("done")])
    assert len(record.calls) >= 1
    output = record.calls[0][2]
    assert output.startswith(("User denied", "denied by rule")), output
