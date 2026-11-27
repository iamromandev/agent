"""ProcessRunner with real `/bin/sh` commands: stdin, environment, caps, timeout, cancellation."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from core.engine.configwatch import Change, ConfigWatch, WatchedSet
from core.engine.guardrails.setup import Guardrails
from core.engine.hooks.runner import MAX_STREAM, ProcessRunner
from core.testing.helpers import assert_all_gone, wait_for_pids


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


def _watch(workspace: Path, tmp_path: Path) -> tuple[ConfigWatch, list[Change]]:
    seen: list[Change] = []
    xdg = {"XDG_CONFIG_HOME": str(tmp_path / "config"), "XDG_DATA_HOME": str(tmp_path / "data")}
    return ConfigWatch(WatchedSet.for_run(workspace, xdg, tmp_path / "home"), seen.append), seen


async def test_a_hook_that_rewrites_a_watched_file_is_reported_with_the_hook_event(workspace: Path, tmp_path: Path) -> None:
    (workspace / ".agent").mkdir()
    watch, seen = _watch(workspace, tmp_path)
    runner = ProcessRunner(workspace, Guardrails.off(workspace), watch)

    run = await runner.run("echo x > .agent/settings.json", b"", 10, {}, event="PreToolUse")

    assert run.exit_code == 0
    assert [c.cause.text for c in seen] == ["hook PreToolUse"]


async def test_a_quiet_hook_is_not_reported(workspace: Path, tmp_path: Path) -> None:
    watch, seen = _watch(workspace, tmp_path)
    runner = ProcessRunner(workspace, Guardrails.off(workspace), watch)

    await runner.run("true", b"", 10, {}, event="Stop")

    assert seen == []
