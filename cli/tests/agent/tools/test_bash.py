from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from src.agent.tools.base import Prepared, ToolContext, ToolResult
from src.agent.tools.shell import BashTool


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


async def test_prepare_returns_a_command_preview(workspace: Path) -> None:
    prepared = await _prepare("uv run pytest -q", _ctx(workspace))

    assert prepared.preview is not None
    assert prepared.preview.kind == "command"
    assert prepared.preview.title == "bash uv run pytest -q"
    assert prepared.preview.body == "uv run pytest -q"


async def test_output_and_exit_code_are_returned_and_non_zero_is_not_an_error(workspace: Path) -> None:
    result = await _run("echo hi; exit 3", _ctx(workspace))

    assert "hi" in result.output
    assert result.output.endswith("exit 3")
    assert result.summary == "exit 3"
    assert not result.is_error


async def test_a_successful_command_reports_exit_0(workspace: Path) -> None:
    result = await _run("true", _ctx(workspace))

    assert result.output == "exit 0"
    assert result.summary == "exit 0"


async def test_stderr_is_merged_into_the_output(workspace: Path) -> None:
    result = await _run("echo out; echo err >&2", _ctx(workspace))

    assert "out" in result.output
    assert "err" in result.output


async def test_the_command_runs_in_the_workspace(workspace: Path) -> None:
    result = await _run("pwd", _ctx(workspace))

    assert result.output.splitlines()[0] == str(workspace)


async def test_colored_output_comes_back_without_ansi_escapes(workspace: Path) -> None:
    result = await _run(r"printf '\033[31mred\033[0m'", _ctx(workspace))

    assert "red" in result.output
    assert "\x1b" not in result.output


async def test_no_color_is_set_for_the_command(workspace: Path) -> None:
    result = await _run("echo $NO_COLOR", _ctx(workspace))

    assert result.output.splitlines()[0] == "1"


async def test_stdin_is_closed_so_cat_returns_at_once(workspace: Path) -> None:
    start = time.perf_counter()
    result = await asyncio.wait_for(_run("cat", _ctx(workspace)), timeout=3)

    assert time.perf_counter() - start < 2
    assert result.output == "exit 0"


async def test_the_command_runs_in_its_own_session_and_process_group(workspace: Path) -> None:
    result = await _run("echo $$; ps -o pgid= -p $$", _ctx(workspace))

    pid, pgid = result.output.splitlines()[:2]
    assert pid.strip() == pgid.strip()
    assert int(pgid) != os.getpgrp()


async def test_timeout_kills_the_command_and_reports_it(workspace: Path) -> None:
    command, shell_pid_file, child_pid_file = _spawn_tree(workspace)

    start = time.perf_counter()
    result = await _run(command, _ctx(workspace, timeout=1.0))

    assert time.perf_counter() - start < 3
    assert "timed out after 1s" in result.output
    assert result.is_error
    shell_pid, child_pid = await _wait_for_pids(shell_pid_file, child_pid_file)
    await _assert_all_gone(shell_pid, child_pid)


async def test_sleep_30_with_a_1s_timeout_finishes_under_3s(workspace: Path) -> None:
    start = time.perf_counter()
    result = await _run("sleep 30", _ctx(workspace, timeout=1.0))

    assert time.perf_counter() - start < 3
    assert "timed out after 1s" in result.output


async def test_cancel_kills_the_whole_process_group_and_reraises(workspace: Path) -> None:
    command, shell_pid_file, child_pid_file = _spawn_tree(workspace)
    prepared = await _prepare(command, _ctx(workspace, timeout=60.0))

    task = asyncio.create_task(_execute(prepared))
    shell_pid, child_pid = await _wait_for_pids(shell_pid_file, child_pid_file)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await _assert_all_gone(shell_pid, child_pid)


async def test_cancel_kills_a_plain_sleep(workspace: Path) -> None:
    pid_file = workspace / "sleep.pid"
    prepared = await _prepare(f"echo $$ > {pid_file}; exec sleep 30", _ctx(workspace, timeout=60.0))

    task = asyncio.create_task(_execute(prepared))
    (pid,) = await _wait_for_pids(pid_file)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await _assert_all_gone(pid)


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


async def test_endless_output_is_bounded_and_marks_the_omitted_part(workspace: Path) -> None:
    result = await _run("yes", _ctx(workspace, timeout=2.0))

    assert len(result.output) < 30_000
    assert re.search(r"\[\d+ bytes omitted\]", result.output)
    assert "timed out after 2s" in result.output
    assert result.is_error


async def test_output_captured_before_a_timeout_is_kept(workspace: Path) -> None:
    result = await _run("echo partial-output; sleep 30", _ctx(workspace, timeout=1.0))

    assert "partial-output" in result.output
    assert "timed out after 1s" in result.output
    assert result.output.index("partial-output") < result.output.index("timed out")


async def test_large_output_keeps_its_head_and_tail(workspace: Path) -> None:
    result = await _run("seq 1 200000", _ctx(workspace, timeout=10.0))

    lines = result.output.splitlines()
    assert lines[0] == "1"
    assert lines[-2] == "200000"
    assert lines[-1] == "exit 0"
    assert len(result.output) < 30_000
    assert re.search(r"\[\d+ bytes omitted\]", result.output)


async def test_small_output_has_no_omitted_marker(workspace: Path) -> None:
    result = await _run("seq 1 100", _ctx(workspace))

    assert "omitted" not in result.output
    assert result.output.splitlines()[-2] == "100"


async def test_an_empty_shell_variable_falls_back_to_bin_sh(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHELL", "")

    result = await _run("echo ok", _ctx(workspace))

    assert not result.is_error
    assert result.output.splitlines()[0] == "ok"


async def test_a_missing_shell_is_an_error_result(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", "/nonexistent/shell")

    result = await _run("echo ok", _ctx(workspace))

    assert result.is_error
    assert result.output.startswith("Could not start shell:")
    assert result.summary == "failed to start"


async def test_a_deleted_workspace_is_an_error_result(workspace: Path) -> None:
    prepared = await _prepare("echo ok", _ctx(workspace))
    workspace.rmdir()

    result = await _execute(prepared)

    assert result.is_error
    assert result.output.startswith("Could not start shell:")
    assert result.summary == "failed to start"


def test_bash_flags_and_label() -> None:
    tool = BashTool()
    assert tool.needs_approval
    assert not tool.mutating
    assert tool.label({"command": "ls -la"}) == "bash ls -la"


