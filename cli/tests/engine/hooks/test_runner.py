"""ProcessRunner with real `/bin/sh` commands: stdin, environment, caps, timeout, cancellation, sandbox wrap."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import statistics
import sys
import time
import uuid
from pathlib import Path

import pytest
from src.engine.guardrails.sandbox import SANDBOX_EXEC
from src.engine.guardrails.setup import Guardrails, prepare
from src.engine.hooks.runner import MAX_STREAM, ProcessRunner

from tests.fakes import recording_sandbox
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


async def test_the_command_reads_the_event_json_on_stdin(runner: ProcessRunner) -> None:
    payload = json.dumps({"event": "PreToolUse", "tool": "edit"}).encode()

    run = await runner.run("cat", payload, 10, {})

    assert (run.exit_code, run.stdout, run.stderr, run.timed_out, run.error) == (0, payload.decode(), "", False, None)


async def test_a_large_stdin_the_command_never_reads_does_not_hang(runner: ProcessRunner) -> None:
    run = await runner.run("true", b"x" * 2_000_000, 10, {})

    assert run.exit_code == 0


async def test_extra_variables_are_set_and_the_command_runs_in_the_workspace(runner: ProcessRunner, workspace: Path) -> None:
    run = await runner.run('printf "%s|%s|%s" "$AGENT_HOOK_EVENT" "$AGENT_PROJECT_DIR" "$PWD"', b"", 10, {
        "AGENT_HOOK_EVENT": "Stop",
        "AGENT_PROJECT_DIR": str(workspace),
    })

    event, project, cwd = run.stdout.split("|")
    assert (event, Path(project).resolve(), Path(cwd).resolve()) == ("Stop", workspace.resolve(), workspace.resolve())


async def test_secret_variables_are_not_passed(runner: ProcessRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_SERVICE_TOKEN", "hunter2hunter2")
    monkeypatch.setenv("PLAIN_NAME", "visible")

    run = await runner.run('printf "%s|%s" "${MY_SERVICE_TOKEN-unset}" "$PLAIN_NAME"', b"", 10, {})

    assert run.stdout == "unset|visible"


async def test_stdout_and_stderr_and_the_exit_code_are_kept_apart(runner: ProcessRunner) -> None:
    run = await runner.run("echo out; echo err >&2; exit 2", b"", 10, {})

    assert (run.exit_code, run.stdout.strip(), run.stderr.strip()) == (2, "out", "err")


async def test_a_missing_command_exits_127_without_an_error(runner: ProcessRunner) -> None:
    run = await runner.run("definitely-not-a-command-xyz", b"", 10, {})

    assert run.exit_code == 127 and run.error is None and not run.timed_out


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


async def test_invalid_utf8_is_replaced(runner: ProcessRunner) -> None:
    run = await runner.run(r"printf '\377\376ok'", b"", 10, {})

    assert run.stdout == "��ok"


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


async def test_the_command_is_started_behind_the_sandbox_wrap(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard, log = recording_sandbox(tmp_path, monkeypatch, workspace)

    run = await ProcessRunner(workspace, guard).run("echo wrapped", b"", 10, {})

    assert run.stdout.strip() == "wrapped"
    assert log.read_text().splitlines() == ["(version 1) (fake profile)"]


needs_sandbox = pytest.mark.skipif(
    sys.platform != "darwin" or not SANDBOX_EXEC.exists(), reason="needs macOS and /usr/bin/sandbox-exec"
)


def real_guard(workspace: Path, home: Path) -> Guardrails:
    return prepare(workspace, None, None, {}, "Darwin", home=home, sessions=home / "sessions")


@needs_sandbox
async def test_a_hook_cannot_write_outside_the_workspace(workspace: Path, tmp_path: Path) -> None:
    home = (tmp_path / "home").resolve()
    home.mkdir()
    outside = Path.home() / f".agent-hook-test-{uuid.uuid4().hex}"  # a place the sandbox does not allow
    runner = ProcessRunner(workspace, real_guard(workspace, home))
    try:
        run = await runner.run(f"echo x > inside; echo y > '{outside}'", b"", 20, {})

        assert (workspace / "inside").exists() and not outside.exists()
        assert run.exit_code != 0
    finally:
        outside.unlink(missing_ok=True)


@needs_sandbox
async def test_a_noop_hook_costs_under_100_ms_with_the_sandbox(workspace: Path, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    runner = ProcessRunner(workspace, real_guard(workspace, home))
    await runner.run("true", b"", 20, {})  # warm up

    times = []
    for _ in range(7):
        started = time.perf_counter()
        await runner.run("true", b"{}", 20, {})
        times.append(time.perf_counter() - started)

    assert statistics.median(times) <= 0.1


async def test_a_hook_that_backgrounds_a_process_returns_when_the_shell_exits_and_leaves_the_daemon_alone(
    runner: ProcessRunner, workspace: Path
) -> None:
    pidfile = workspace / "pid"
    started = time.perf_counter()

    run = await runner.run("sleep 30 & echo $! > pid; echo hi; exit 3", b"", 20, {})

    [pid] = await wait_for_pids(pidfile)
    try:
        assert time.perf_counter() - started < 5  # not the 20 s timeout: the daemon holds the pipes open
        assert (run.exit_code, run.stdout.strip(), run.timed_out) == (3, "hi", False)
        os.kill(pid, 0)  # the daemon the user started is still there
    finally:
        os.kill(pid, signal.SIGKILL)
    await assert_all_gone(pid)


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


async def test_kill_all_leaves_a_daemon_of_a_finished_hook_alone(runner: ProcessRunner, workspace: Path) -> None:
    await runner.run("sleep 30 & echo $! > pid", b"", 10, {})
    [pid] = await wait_for_pids(workspace / "pid")

    try:
        runner.kill_all()
        await asyncio.sleep(0.1)
        os.kill(pid, 0)  # still there: the user started it on purpose
    finally:
        os.kill(pid, signal.SIGKILL)
    await assert_all_gone(pid)
