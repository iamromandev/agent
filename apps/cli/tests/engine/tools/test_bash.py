from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from src.engine.tools.base import Prepared, ToolContext, ToolResult
from src.engine.tools.shell import BashTool, _Capture
from src.engine.tools.text import strip_ansi


@pytest.fixture(autouse=True)
def _posix_sh(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool runs $SHELL -c; pin it so the tests do not depend on the developer's shell."""
    monkeypatch.setenv("SHELL", "/bin/sh")


def _ctx(workspace: Path, timeout: float = 5.0) -> ToolContext:
    return ToolContext(workspace=workspace, bash_timeout=timeout)


async def _prepare(command: str, ctx: ToolContext) -> Prepared:
    result = await BashTool().prepare({"command": command}, ctx)
    assert isinstance(result, Prepared)
    return result


async def _execute(prepared: Prepared) -> ToolResult:
    return await prepared.execute()  # create_task wants a coroutine, execute() is an Awaitable


async def _run(command: str, ctx: ToolContext) -> ToolResult:
    return await (await _prepare(command, ctx)).execute()


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


async def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.02)


async def _wait_for_pids(*files: Path) -> list[int]:
    await _wait_until(lambda: all(f.exists() and f.read_text().strip() for f in files))
    return [int(f.read_text().strip()) for f in files]


async def _assert_all_gone(*pids: int) -> None:
    # A killed orphan stays a zombie until init reaps it, so allow a moment.
    await _wait_until(lambda: all(_gone(p) for p in pids), timeout=3.0)
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


# A shell (group leader) plus a background grandchild in the same group.
def _spawn_tree(workspace: Path) -> tuple[str, Path, Path]:
    shell_pid, child_pid = workspace / "shell.pid", workspace / "child.pid"
    command = f"echo $$ > {shell_pid}; sleep 30 & echo $! > {child_pid}; wait"
    return command, shell_pid, child_pid


async def test_a_timeout_keeps_the_colored_partial_output(workspace: Path) -> None:
    result = await _run(r"printf '\033[31mpartial\033[0m'; sleep 30", _ctx(workspace, timeout=1.0))

    assert result.output.startswith("partial\nCommand timed out")
    assert result.display is not None
    assert result.display.startswith("\x1b[31mpartial\x1b[0m\nCommand timed out")


async def test_a_command_that_finishes_leaves_no_background_children_alive_after_timeout(
    workspace: Path,
) -> None:
    """The shell exits at once but a background child keeps the pipe open until the timeout."""
    child_pid_file = workspace / "bg.pid"
    command = f"sleep 30 & echo $! > {child_pid_file}"

    result = await _run(command, _ctx(workspace, timeout=1.0))

    assert "timed out after 1s" in result.output
    (pid,) = await _wait_for_pids(child_pid_file)
    await _assert_all_gone(pid)


async def test_a_missing_shell_is_an_error_result(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", "/nonexistent/shell")

    result = await _run("echo ok", _ctx(workspace))

    assert result.is_error
    assert result.output.startswith("Could not start shell:")
    assert result.summary == "failed to start"


def _plain(display: str | None) -> str:
    assert display is not None
    return strip_ansi(display)


async def test_the_display_copy_has_no_partial_escape_where_head_and_tail_meet(workspace: Path) -> None:
    result = await _run(r'''yes "$(printf '\033[38;5;123mabcdefghi\033[0m')" | head -n 20000''', _ctx(workspace))

    display = result.display or ""
    assert re.search(r"\x1b(?!\[[0-9;:]*m)", display) is None
    assert re.search(r"\[\d+ bytes omitted\]\n[0-9;]*m", display) is None
    assert _plain(display) == result.output


def _captured(*chunks: bytes) -> _Capture:
    capture = _Capture()
    for chunk in chunks:
        capture.feed(chunk)
    return capture


def test_an_overlong_osc_is_dropped_whole() -> None:
    assert _captured(b"a\x1b]0;" + b"t" * 5000 + b"\x07b").raw() == "ab"
