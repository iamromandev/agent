from __future__ import annotations

from typing import Any

import pytest
from src.engine.terminal import manager as manager_module
from src.engine.terminal.manager import Terminal, TerminalManager
from src.engine.tools.base import Prepared, Tool, ToolContext, ToolResult
from src.engine.tools.terminal import TermRead, TermStart, TermStop
from src.engine.tools.text import TOOL_OUTPUT_LIMIT

from tests.helpers import wait_until


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


# term_start


async def test_start_refuses_an_invalid_name_before_approval(manager: TerminalManager, ctx: ToolContext) -> None:
    result = await _prepare(TermStart(manager), {"name": "bad name", "command": "true"}, ctx)

    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "name" in result.output
    assert manager.names() == []


async def test_start_refuses_if_the_name_was_taken_while_waiting_for_approval(
    manager: TerminalManager, ctx: ToolContext
) -> None:
    prepared = await _prepare(TermStart(manager), {"name": "api", "command": "sleep 30"}, ctx)
    assert isinstance(prepared, Prepared)
    await _start(manager, ctx, "api", "sleep 31")

    result = await prepared.execute()

    assert result.is_error
    assert "already running" in result.output


# term_read


async def test_read_of_a_terminal_without_output_is_just_the_status(
    manager: TerminalManager, ctx: ToolContext
) -> None:
    await _start(manager, ctx, "quiet", "sleep 30")

    assert (await _read(manager, ctx, "quiet")).output == "running"


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


async def test_read_of_an_unknown_name_with_no_terminals_says_none(manager: TerminalManager, ctx: ToolContext) -> None:
    result = await _read(manager, ctx, "nope")

    assert result.is_error
    assert result.output == "no terminal named nope; terminals: none"


# term_stop


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
