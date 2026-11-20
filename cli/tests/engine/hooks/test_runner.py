"""ProcessRunner with real `/bin/sh` commands: stdin, environment, caps, timeout, cancellation, sandbox wrap."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest
from src.engine.guardrails.sandbox import SANDBOX_EXEC
from src.engine.guardrails.setup import Guardrails, prepare
from src.engine.hooks.runner import MAX_STREAM, ProcessRunner

from tests.helpers import assert_all_gone, wait_for_pids


@pytest.fixture(autouse=True)
def sh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", "/bin/sh")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "ws"
    path.mkdir()
    return path


@pytest.fixture
def runner(workspace: Path) -> ProcessRunner:
    return ProcessRunner(workspace, Guardrails.off(workspace))


async def test_a_large_stdin_the_command_never_reads_does_not_hang(runner: ProcessRunner) -> None:
    run = await runner.run("true", b"x" * 2_000_000, 10, {})

    assert run.exit_code == 0


async def test_a_missing_shell_is_a_start_error(runner: ProcessRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", "/nonexistent/shell")

    run = await runner.run("true", b"", 10, {})

    assert run.exit_code is None and run.error and not run.timed_out


async def test_ten_megabytes_of_output_are_capped_and_the_process_is_not_stuck(runner: ProcessRunner) -> None:
    started = time.monotonic()

    run = await runner.run("yes x | head -c 10000000; yes y | head -c 10000000 >&2", b"", 30, {})

    assert run.exit_code == 0
    assert len(run.stdout.encode()) == MAX_STREAM and set(run.stdout.replace("\n", "")) == {"x"}
    assert len(run.stderr.encode()) == MAX_STREAM
    assert time.monotonic() - started < 15


async def test_a_command_that_outlives_its_timeout_is_killed_with_its_group(runner: ProcessRunner, workspace: Path) -> None:
    pidfile = workspace / "pid"
    started = time.monotonic()

    run = await runner.run("sleep 30 & echo $! > pid; echo started; wait", b"", 1, {})

    assert time.monotonic() - started < 10
    assert run.timed_out and run.exit_code is None and run.stdout.strip() == "started"
    await assert_all_gone(*await wait_for_pids(pidfile))


async def test_cancelling_the_run_kills_the_group(runner: ProcessRunner, workspace: Path) -> None:
    pidfile = workspace / "pid"
    task = asyncio.ensure_future(runner.run("sleep 30 & echo $! > pid; wait", b"", 60, {}))
    pids = await wait_for_pids(pidfile)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await assert_all_gone(*pids)


needs_sandbox = pytest.mark.skipif(
    sys.platform != "darwin" or not SANDBOX_EXEC.exists(), reason="needs macOS and /usr/bin/sandbox-exec"
)


def real_guard(workspace: Path, home: Path) -> Guardrails:
    return prepare(workspace, None, None, {}, "Darwin", home=home, sessions=home / "sessions")


async def test_kill_all_kills_the_process_group_of_a_running_hook(runner: ProcessRunner, workspace: Path) -> None:
    task = asyncio.ensure_future(runner.run("sleep 30 & echo $! > pid; wait", b"", 60, {}))
    pids = await wait_for_pids(workspace / "pid")

    runner.kill_all()

    await assert_all_gone(*pids)
    run = await asyncio.wait_for(task, 5)
    assert run.exit_code != 0 and not run.timed_out


async def test_a_hook_asked_for_after_kill_all_does_not_start(runner: ProcessRunner, workspace: Path) -> None:
    runner.kill_all()

    run = await runner.run("touch started", b"", 10, {})

    assert run.exit_code is None and run.error is not None and "closed" in run.error
    assert not (workspace / "started").exists()
