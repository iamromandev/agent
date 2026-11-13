from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from src.agent.tools.base import Prepared, ToolContext, ToolResult
from src.agent.tools.shell import BashTool, _Capture
from src.agent.tools.text import strip_ansi


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


async def test_the_command_is_asked_for_color_and_not_told_to_avoid_it(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")  # even when the parent has it set
    result = await _run('printf "%s:%s:%s" "${NO_COLOR-unset}" "$FORCE_COLOR" "$CLICOLOR_FORCE"', _ctx(workspace))

    assert result.output.splitlines()[0] == "unset:1:1"


async def test_the_model_gets_plain_text_and_the_screen_gets_the_colors(workspace: Path) -> None:
    result = await _run(r"printf '\033[32mok\033[0m'", _ctx(workspace))

    assert result.output == "ok\nexit 0"
    assert result.display == "\x1b[32mok\x1b[0m\nexit 0"


async def test_output_without_color_has_no_display_copy(workspace: Path) -> None:
    result = await _run("echo plain", _ctx(workspace))

    assert result.output == "plain\nexit 0"
    assert result.display is None


async def test_cursor_movement_is_not_kept_in_the_display_copy(workspace: Path) -> None:
    result = await _run(r"printf '\033[2K\033[1A\033[32mok\033[0m'", _ctx(workspace))

    assert result.display == "\x1b[32mok\x1b[0m\nexit 0"
    assert "\x1b[2K" not in (result.display or "")


async def test_a_timeout_keeps_the_colored_partial_output(workspace: Path) -> None:
    result = await _run(r"printf '\033[31mpartial\033[0m'; sleep 30", _ctx(workspace, timeout=1.0))

    assert result.output.startswith("partial\nCommand timed out")
    assert result.display is not None
    assert result.display.startswith("\x1b[31mpartial\x1b[0m\nCommand timed out")


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


def _plain(display: str | None) -> str:
    assert display is not None
    return strip_ansi(display)


async def test_a_newline_before_a_trailing_reset_does_not_make_the_copies_differ(workspace: Path) -> None:
    result = await _run(r"printf 'text\n\033[0m'", _ctx(workspace))

    assert result.output == "text\nexit 0"
    assert _plain(result.display) == result.output


async def test_long_colored_output_is_cut_at_the_same_visible_text_in_both_copies(workspace: Path) -> None:
    command = r"""seq 1 200000 | awk '{printf "\033[32m%s\033[0m\n", $1}'"""
    result = await _run(command, _ctx(workspace, 30.0))

    assert re.search(r"\[\d+ bytes omitted\]", result.output)
    assert _plain(result.display) == result.output


async def test_colored_output_near_the_cap_keeps_as_much_text_as_plain_output(workspace: Path) -> None:
    colored = await _run(r"yes $'\033[32mabcdefghi\033[0m' | head -n 5000", _ctx(workspace))
    plain = await _run("yes abcdefghi | head -n 5000", _ctx(workspace))

    assert colored.output == plain.output
    assert _plain(colored.display) == colored.output


async def test_the_display_copy_has_no_partial_escape_where_head_and_tail_meet(workspace: Path) -> None:
    result = await _run(r"yes $'\033[38;5;123mabcdefghi\033[0m' | head -n 20000", _ctx(workspace))

    display = result.display or ""
    assert re.search(r"\x1b(?!\[[0-9;:]*m)", display) is None
    assert re.search(r"\[\d+ bytes omitted\]\n[0-9;]*m", display) is None
    assert _plain(display) == result.output


async def test_an_escape_split_across_chunks_is_still_one_sequence(workspace: Path) -> None:
    result = await _run(r"printf '\033[3'; sleep 0.2; printf '2mok\033[0m'", _ctx(workspace))

    assert result.output == "ok\nexit 0"
    assert result.display == "\x1b[32mok\x1b[0m\nexit 0"


def _captured(*chunks: bytes) -> _Capture:
    capture = _Capture()
    for chunk in chunks:
        capture.feed(chunk)
    return capture


@pytest.mark.parametrize("run", ["\n" * 20_000 + "x", "\x1b[0m" * 20_000 + "x"])
def test_a_long_run_of_trailing_newlines_or_styles_is_trimmed_in_linear_time(run: str) -> None:
    capture = _captured(run.encode())
    started = time.perf_counter()
    capture.raw()
    capture.text()
    assert time.perf_counter() - started < 0.05


def test_trailing_newlines_are_dropped_and_the_trailing_reset_is_kept() -> None:
    assert _captured(b"a\x1b[32mb\n\n\x1b[0m\n").raw() == "a\x1b[32mb\x1b[0m"


def test_oversized_style_tokens_are_dropped_and_do_not_leak_as_text() -> None:
    big = b"\x1b[" + b"1;" * 5000 + b"m"
    capture = _captured(b"a" + big + b"b" + big * 100 + b"c")
    assert capture.raw() == "abc"
    assert capture.text() == "abc"


def test_an_overlong_osc_is_dropped_whole() -> None:
    assert _captured(b"a\x1b]0;" + b"t" * 5000 + b"\x07b").raw() == "ab"


def test_style_tokens_are_bounded_in_bytes() -> None:
    capture = _captured(*[b"\x1b[1m\x1b[" + b"9" * 500 + b"m"] * 2000)
    assert len(capture.raw()) <= 2000 * 4 + 10


def test_the_display_copy_resets_styles_before_the_omitted_marker() -> None:
    capture = _captured(b"\x1b[31m" + b"a" * 40_000 + b"\x1b[0m")
    assert "\x1b[0m\n[" in capture.raw().split("omitted")[0]
    assert capture.text().startswith("a" * 10_000 + "\n[")
    assert "\x1b" not in capture.text()
