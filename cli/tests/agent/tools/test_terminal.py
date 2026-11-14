from __future__ import annotations

import signal
from pathlib import Path
from typing import Any

import pytest
from src.agent.terminal import manager as manager_module
from src.agent.terminal.manager import Terminal, TerminalManager
from src.agent.tools import default_tools
from src.agent.tools.base import Prepared, Tool, ToolContext, ToolResult, check_args
from src.agent.tools.terminal import TermRead, TermStart, TermStop
from src.agent.tools.text import TOOL_OUTPUT_LIMIT

from tests.helpers import assert_all_gone, wait_for_pids, wait_until


async def _prepare(tool: Tool, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
    return await tool.prepare(args, ctx)


async def _start(manager: TerminalManager, ctx: ToolContext, name: str, command: str) -> ToolResult:
    prepared = await _prepare(TermStart(manager), {"name": name, "command": command}, ctx)
    assert isinstance(prepared, Prepared)
    return await prepared.execute()


async def _read(manager: TerminalManager, ctx: ToolContext, name: str, **args: Any) -> ToolResult:
    result = await _prepare(TermRead(manager), {"name": name, **args}, ctx)
    assert isinstance(result, ToolResult)
    return result


async def _stop(manager: TerminalManager, ctx: ToolContext, name: str) -> ToolResult:
    prepared = await _prepare(TermStop(manager), {"name": name}, ctx)
    assert isinstance(prepared, Prepared)
    return await prepared.execute()


def _text(terminal: Terminal) -> str:
    return "\n".join(terminal.buffer.tail(100))


# registry


async def test_the_terminal_tools_are_added_only_with_a_manager(manager: TerminalManager) -> None:
    base = ["read", "glob", "grep", "write", "edit", "bash"]

    assert [t.name for t in default_tools()] == base
    assert [t.name for t in default_tools(manager)] == [*base, "term_start", "term_read", "term_stop"]


async def test_start_and_stop_are_approved_and_mutating_but_read_is_neither(manager: TerminalManager) -> None:
    flags = {t.name: (t.needs_approval, t.mutating) for t in default_tools(manager) if t.name.startswith("term_")}

    assert flags == {"term_start": (True, True), "term_read": (False, False), "term_stop": (True, True)}


async def test_the_terminal_specs_describe_every_parameter(manager: TerminalManager) -> None:
    specs = {t.name: t.spec() for t in default_tools(manager) if t.name.startswith("term_")}

    assert specs["term_start"].parameters["required"] == ["name", "command"]
    assert specs["term_read"].parameters["required"] == ["name"]
    assert specs["term_stop"].parameters["required"] == ["name"]
    for spec in specs.values():
        assert spec.description
        for prop in spec.parameters["properties"].values():
            assert prop["type"] in {"string", "integer"}
            assert prop["description"]


async def test_the_lines_argument_must_be_an_integer(manager: TerminalManager) -> None:
    schema = TermRead(manager).parameters

    assert check_args(schema, {"name": "a", "lines": "5"}) == '"lines" must be an integer'
    assert check_args(schema, {"name": "a"}) is None


# term_start


async def test_start_previews_the_command_and_starts_nothing_before_it_runs(
    manager: TerminalManager, ctx: ToolContext
) -> None:
    prepared = await _prepare(TermStart(manager), {"name": "api", "command": "python -m http.server"}, ctx)

    assert isinstance(prepared, Prepared)
    assert prepared.preview is not None
    assert prepared.preview.kind == "command"
    assert prepared.preview.title == "term_start api"
    assert prepared.preview.body == "python -m http.server"
    assert manager.names() == []


async def test_start_runs_the_command_and_reports_the_pid(manager: TerminalManager, ctx: ToolContext) -> None:
    result = await _start(manager, ctx, "api", "sleep 30")

    terminal = manager.get("api")
    assert terminal is not None
    assert result.output == f"started api (pid {terminal.process.pid})"
    assert not result.is_error


async def test_start_returns_at_once_for_a_command_that_fails_at_once(
    manager: TerminalManager, ctx: ToolContext
) -> None:
    result = await _start(manager, ctx, "bad", "exit 3")

    assert result.output.startswith("started bad (pid ")
    assert not result.is_error


async def test_start_refuses_an_invalid_name_before_approval(manager: TerminalManager, ctx: ToolContext) -> None:
    result = await _prepare(TermStart(manager), {"name": "bad name", "command": "true"}, ctx)

    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "name" in result.output
    assert manager.names() == []


async def test_start_refuses_a_name_that_is_running_before_approval(
    manager: TerminalManager, ctx: ToolContext
) -> None:
    await _start(manager, ctx, "api", "sleep 30")

    result = await _prepare(TermStart(manager), {"name": "api", "command": "sleep 31"}, ctx)

    assert isinstance(result, ToolResult)
    assert result.is_error
    assert result.output == "terminal api is already running"


async def test_start_refuses_if_the_name_was_taken_while_waiting_for_approval(
    manager: TerminalManager, ctx: ToolContext
) -> None:
    prepared = await _prepare(TermStart(manager), {"name": "api", "command": "sleep 30"}, ctx)
    assert isinstance(prepared, Prepared)
    await _start(manager, ctx, "api", "sleep 31")

    result = await prepared.execute()

    assert result.is_error
    assert "already running" in result.output


async def test_start_can_reuse_the_name_of_an_exited_terminal(manager: TerminalManager, ctx: ToolContext) -> None:
    await _start(manager, ctx, "api", "exit 0")
    old = manager.get("api")
    assert old is not None
    await wait_until(lambda: old.status[0] == "exited")

    result = await _start(manager, ctx, "api", "sleep 30")

    assert not result.is_error
    assert manager.get("api") is not old


# term_read


async def test_read_returns_the_last_lines_and_running(manager: TerminalManager, ctx: ToolContext) -> None:
    await _start(
        manager, ctx, "api", "for i in 1 2 3 4 5 6 7 8; do printf '\\033[32mline %s\\033[0m\\n' $i; done; sleep 30"
    )
    terminal = manager.get("api")
    assert terminal is not None
    await wait_until(lambda: "line 8" in _text(terminal))

    result = await _read(manager, ctx, "api", lines=5)

    assert result.output == "line 4\nline 5\nline 6\nline 7\nline 8\n\nrunning"
    assert not result.is_error
    assert "\x1b" not in result.output


async def test_read_defaults_to_the_last_100_lines(manager: TerminalManager, ctx: ToolContext) -> None:
    await _start(manager, ctx, "api", "i=1; while [ $i -le 150 ]; do echo n$i; i=$((i+1)); done; sleep 30")
    terminal = manager.get("api")
    assert terminal is not None
    await wait_until(lambda: "n150" in _text(terminal))

    lines = (await _read(manager, ctx, "api")).output.split("\n")

    assert lines[0] == "n51"
    assert lines[99] == "n150"
    assert lines[-1] == "running"


@pytest.mark.parametrize(("asked", "expected"), [(0, 1), (-5, 1), (10**9, 5_000)])
async def test_read_clamps_the_line_count(
    manager: TerminalManager, ctx: ToolContext, asked: int, expected: int
) -> None:
    await _start(manager, ctx, "api", "i=1; while [ $i -le 20 ]; do echo n$i; i=$((i+1)); done; sleep 30")
    terminal = manager.get("api")
    assert terminal is not None
    await wait_until(lambda: "n20" in _text(terminal))

    result = await _read(manager, ctx, "api", lines=asked)

    assert len(result.output.split("\n")) == min(expected, 20) + 2


async def test_read_of_a_terminal_without_output_is_just_the_status(
    manager: TerminalManager, ctx: ToolContext
) -> None:
    await _start(manager, ctx, "quiet", "sleep 30")

    assert (await _read(manager, ctx, "quiet")).output == "running"


async def test_read_after_a_normal_exit_gives_the_code(manager: TerminalManager, ctx: ToolContext) -> None:
    await _start(manager, ctx, "job", "echo done; exit 3")
    terminal = manager.get("job")
    assert terminal is not None
    await wait_until(lambda: terminal.status[0] == "exited")

    assert (await _read(manager, ctx, "job")).output == "done\n\nexited 3"


async def test_read_after_a_signal_death_names_the_signal(manager: TerminalManager, ctx: ToolContext) -> None:
    await _start(manager, ctx, "job", "kill -KILL $$")
    terminal = manager.get("job")
    assert terminal is not None
    await wait_until(lambda: terminal.status == ("exited", -signal.SIGKILL))

    assert (await _read(manager, ctx, "job")).output == "exited: SIGKILL"


async def test_read_keeps_the_status_when_the_output_is_huge(manager: TerminalManager, ctx: ToolContext) -> None:
    await _start(manager, ctx, "big", "i=0; while [ $i -lt 400 ]; do printf '%0100d\\n' $i; i=$((i+1)); done; sleep 30")
    terminal = manager.get("big")
    assert terminal is not None
    await wait_until(lambda: terminal.buffer.tail(1) == ["0" * 97 + "399"])

    text = (await _read(manager, ctx, "big", lines=5_000)).output

    assert len(text) <= TOOL_OUTPUT_LIMIT
    assert text.endswith("\n\nrunning")
    assert text.split("\n")[-3] == "0" * 97 + "399"
    assert text.startswith("[")  # says how many earlier lines were left out


async def test_read_of_an_unknown_name_lists_the_existing_ones(manager: TerminalManager, ctx: ToolContext) -> None:
    await _start(manager, ctx, "api", "sleep 30")
    await _start(manager, ctx, "web", "sleep 30")

    result = await _read(manager, ctx, "nope")

    assert result.is_error
    assert result.output == "no terminal named nope; terminals: api, web"


async def test_read_of_an_unknown_name_with_no_terminals_says_none(manager: TerminalManager, ctx: ToolContext) -> None:
    result = await _read(manager, ctx, "nope")

    assert result.is_error
    assert result.output == "no terminal named nope; terminals: none"


# term_stop


async def test_stop_previews_what_it_stops(manager: TerminalManager, ctx: ToolContext) -> None:
    await _start(manager, ctx, "api", "sleep 30")

    prepared = await _prepare(TermStop(manager), {"name": "api"}, ctx)

    assert isinstance(prepared, Prepared)
    assert prepared.preview is not None
    assert prepared.preview.kind == "command"
    assert prepared.preview.title == "term_stop api"
    assert prepared.preview.body == "stop api"
    terminal = manager.get("api")
    assert terminal is not None and terminal.status == ("running", None)


async def test_stop_ends_the_whole_group_and_reports_how(
    manager: TerminalManager, ctx: ToolContext, workspace: Path
) -> None:
    shell_pid, child_pid = workspace / "shell.pid", workspace / "child.pid"
    await _start(manager, ctx, "api", f"echo $$ > {shell_pid}; sleep 30 & echo $! > {child_pid}; wait")
    pids = await wait_for_pids(shell_pid, child_pid)

    result = await _stop(manager, ctx, "api")

    assert result.output == "stopped api (exited: SIGTERM)"
    assert not result.is_error
    await assert_all_gone(*pids)
    terminal = manager.get("api")
    assert terminal is not None and terminal.status[0] == "exited"  # the tab stays until it is closed


async def test_stop_of_a_process_that_survives_sigkill_says_it_is_still_running(
    manager: TerminalManager, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manager_module, "_KILL_WAIT", 0.2)
    await _start(manager, ctx, "api", "sleep 30")
    terminal = manager.get("api")
    assert terminal is not None
    real_poll = terminal.process.poll
    stuck = True
    monkeypatch.setattr(terminal.process, "poll", lambda: None if stuck else real_poll())

    result = await _stop(manager, ctx, "api")

    assert result.is_error
    assert result.output == "could not stop api: it is still running after SIGKILL"
    stuck = False
    await wait_until(lambda: terminal.status[0] == "exited", timeout=5)


async def test_stop_says_so_when_a_process_of_the_terminal_could_not_be_killed(
    manager: TerminalManager, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manager_module, "_KILL_WAIT", 0.2)
    await _start(manager, ctx, "api", "sleep 30")
    terminal = manager.get("api")
    assert terminal is not None
    real_alive = terminal.process.group_alive
    survivor = True
    monkeypatch.setattr(terminal.process, "group_alive", lambda: survivor or real_alive())

    result = await _stop(manager, ctx, "api")

    assert result.is_error
    assert result.output == "stopped api (exited: SIGTERM), but some of its processes are still running"
    survivor = False


async def test_stop_of_an_exited_terminal_says_so(manager: TerminalManager, ctx: ToolContext) -> None:
    await _start(manager, ctx, "job", "exit 2")
    terminal = manager.get("job")
    assert terminal is not None
    await wait_until(lambda: terminal.status[0] == "exited")

    result = await _stop(manager, ctx, "job")

    assert result.output == "job already exited (exited 2)"
    assert not result.is_error


async def test_stop_of_an_unknown_name_lists_the_existing_ones(manager: TerminalManager, ctx: ToolContext) -> None:
    await _start(manager, ctx, "api", "sleep 30")

    result = await _prepare(TermStop(manager), {"name": "nope"}, ctx)

    assert isinstance(result, ToolResult)
    assert result.is_error
    assert result.output == "no terminal named nope; terminals: api"


async def test_stop_of_a_terminal_closed_while_waiting_for_approval_is_an_error(
    manager: TerminalManager, ctx: ToolContext
) -> None:
    await _start(manager, ctx, "api", "sleep 30")
    prepared = await _prepare(TermStop(manager), {"name": "api"}, ctx)
    assert isinstance(prepared, Prepared)
    await manager.close("api")

    result = await prepared.execute()

    assert result.is_error
    assert "no terminal named api" in result.output


async def test_every_terminal_tool_shows_its_name_in_the_label(manager: TerminalManager) -> None:
    assert TermStart(manager).label({"name": "api", "command": "x"}) == "term_start api"
    assert TermRead(manager).label({"name": "api"}) == "term_read api"
    assert TermStop(manager).label({"name": "api"}) == "term_stop api"


def test_always_on_term_start_covers_only_that_command() -> None:
    from src.agent.approval import Grants
    from src.agent.mode import Mode
    from src.agent.tools.terminal import TermStart

    grants = Grants()
    grants.add("term_start", {"name": "web", "command": "make dev"}, TermStart.grant_arg)
    assert grants.covers("term_start", {"name": "api", "command": "make dev"}, Mode.BUILD, TermStart.grant_arg)
    assert not grants.covers("term_start", {"name": "web", "command": "rm -rf ~"}, Mode.BUILD, TermStart.grant_arg)
