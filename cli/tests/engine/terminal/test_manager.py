from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import Generator
from pathlib import Path

import pytest
from src.engine.terminal import manager as manager_module
from src.engine.terminal.manager import MAX_TERMINALS, Terminal, TerminalError, TerminalManager

from tests.helpers import assert_all_gone, wait_for_pids, wait_until


def _text(terminal: Terminal) -> str:
    return "\n".join(terminal.buffer.tail(100))


async def _exited(terminal: Terminal) -> None:
    await wait_until(lambda: terminal.status[0] == "exited")


async def test_a_failing_subscriber_does_not_stop_the_others_or_the_buffer(
    manager: TerminalManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = asyncio.get_running_loop()
    errors: list[object] = []
    monkeypatch.setattr(loop, "call_exception_handler", errors.append)
    terminal = manager.start("a", "cat")
    seen = bytearray()

    def boom(data: bytes) -> None:
        raise RuntimeError("boom")

    terminal.subscribe(boom)
    terminal.subscribe(seen.extend)
    terminal.write(b"hi\n")

    await wait_until(lambda: b"hi" in seen and "hi" in _text(terminal))
    assert errors


async def test_an_exit_listener_added_after_the_exit_is_called(manager: TerminalManager) -> None:
    terminal = manager.start("a", "exit 0")
    await _exited(terminal)
    exits: list[Terminal] = []

    terminal.on_exit(exits.append)

    await wait_until(lambda: exits == [terminal])


async def test_a_missing_workspace_is_refused(tmp_path: Path) -> None:
    broken = TerminalManager(tmp_path / "gone")

    with pytest.raises(TerminalError, match="workspace"):
        broken.start("a", "true")


async def test_closing_an_unknown_name_is_quiet(manager: TerminalManager) -> None:
    await manager.close("nope")


# --- members of the group that outlive the leader ------------------------------------------------


@contextlib.contextmanager
def _killed_after(*files: Path) -> Generator[None]:
    """Whatever the files name is killed when the block ends, so a failing test leaks no process."""
    try:
        yield
    finally:
        for file in files:
            with contextlib.suppress(ValueError, OSError):
                os.kill(int(file.read_text().strip()), signal.SIGKILL)


async def test_a_leader_that_exits_on_its_own_does_not_leave_a_child_that_ignores_hup(
    manager: TerminalManager, workspace: Path
) -> None:
    child = workspace / "child.pid"
    with _killed_after(child):
        terminal = manager.start("a", f"(trap '' HUP; exec sleep 60) & echo $! > {child}; exit 0")
        (pid,) = await wait_for_pids(child)

        await _exited(terminal)

        assert terminal.status == ("exited", 0)
        await assert_all_gone(pid)


async def test_stop_kills_a_foreground_job_of_the_user_shell_that_ignores_term_and_hup(
    manager: TerminalManager, workspace: Path
) -> None:
    """With job control the job has a group of its own: signalling the shell's group misses it."""
    job = workspace / "job.pid"
    with _killed_after(job):
        terminal = manager.start("shell", None)
        terminal.write(f"sh -c 'trap \"\" TERM HUP; echo $$ > {job}; exec sleep 60'\n".encode())
        (pid,) = await wait_for_pids(job)
        if os.getpgid(pid) == terminal.process.pid:
            pytest.skip("this shell has no job control: the job shares the shell's group")

        await terminal.stop(grace=0.3)

        assert terminal.status[0] == "exited"
        await assert_all_gone(pid)


# --- close keeps a terminal tracked until it is stopped ---------------------------------------------


async def test_close_all_during_a_close_still_waits_for_that_terminal(manager: TerminalManager) -> None:
    terminal = manager.start("a", "trap '' TERM; echo ready; while :; do sleep 0.1; done")
    await wait_until(lambda: "ready" in _text(terminal))
    closing = asyncio.create_task(manager.close("a"))
    await asyncio.sleep(0.05)  # the close is waiting out the grace

    assert manager.get("a") is terminal
    await manager.close_all()

    assert terminal.status[0] == "exited"
    await closing


# --- the number of terminals ---------------------------------------------------------------------


async def test_a_new_terminal_is_refused_when_all_the_slots_run(manager: TerminalManager) -> None:
    for index in range(MAX_TERMINALS):
        manager.start(f"t{index}", "sleep 30")

    with pytest.raises(TerminalError, match="too many"):
        manager.check_name("one-more")
    with pytest.raises(TerminalError, match="too many"):
        manager.start("one-more", "sleep 30")

    assert len(manager.names()) == MAX_TERMINALS


async def test_a_new_terminal_evicts_the_oldest_exited_one_when_full(manager: TerminalManager) -> None:
    manager.start("old", "exit 0")
    manager.start("newer", "exit 0")
    for index in range(MAX_TERMINALS - 2):
        manager.start(f"t{index}", "sleep 30")
    for name in ("old", "newer"):
        terminal = manager.get(name)
        assert terminal is not None
        await _exited(terminal)
    events: list[tuple[str, str]] = []
    manager.listen(lambda name, event: events.append((name, event)))

    manager.start("fresh", "sleep 30")

    assert manager.get("old") is None
    assert manager.get("newer") is not None
    assert len(manager.names()) == MAX_TERMINALS
    assert events == [("old", "closed"), ("fresh", "started")]


async def test_a_replaced_terminal_is_reported_closed_after_the_new_one_is_registered(
    manager: TerminalManager,
) -> None:
    old = manager.start("a", "exit 0")
    await _exited(old)
    seen: list[Terminal | None] = []
    manager.listen(lambda name, event: seen.append(manager.get(name)) if event == "closed" else None)

    new = manager.start("a", "sleep 30")

    assert seen == [new]


# --- a leader that does not go ----------------------------------------------------------------------


# --- jobs of the user's shell that leave its process group --------------------------------------------

_BACKGROUND_JOBS = (
    "nohup sleep 60 >/dev/null 2>&1 & echo $! > {plain}\n"
    "(trap '' HUP TERM; exec sleep 60) & echo $! > {stubborn}\n"
)


async def _start_shell_with_background_jobs(
    manager: TerminalManager, workspace: Path
) -> tuple[Terminal, list[int], tuple[Path, Path]]:
    plain, stubborn = workspace / "plain.pid", workspace / "stubborn.pid"
    terminal = manager.start("shell", None)
    terminal.write(_BACKGROUND_JOBS.format(plain=plain, stubborn=stubborn).encode())
    return terminal, await wait_for_pids(plain, stubborn), (plain, stubborn)


async def test_start_without_a_running_loop_starts_no_process(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned: list[object] = []
    monkeypatch.setattr(manager_module.PtyProcess, "spawn", lambda *args: spawned.append(args))
    manager = TerminalManager(workspace)

    def start_off_loop() -> None:
        manager.start("a", "sleep 30")

    with pytest.raises(RuntimeError):
        await asyncio.to_thread(start_off_loop)  # a thread has no running loop

    assert spawned == []


async def test_an_impossible_size_is_a_terminal_error_not_a_value_error(workspace: Path) -> None:
    manager = TerminalManager(workspace, cols=100_000)

    with pytest.raises(TerminalError, match="could not start terminal"):
        manager.start("a", "sleep 30")

    assert manager.names() == []


async def test_kill_all_kills_every_group_at_once_without_the_event_loop(
    manager: TerminalManager, workspace: Path
) -> None:
    files: list[Path] = []
    for index in range(2):
        shell, child = workspace / f"s{index}.pid", workspace / f"c{index}.pid"
        files += [shell, child]
        manager.start(f"t{index}", f"trap '' TERM HUP; echo $$ > {shell}; sleep 1000 & echo $! > {child}; wait")
    pids = await wait_for_pids(*files)

    manager.kill_all()  # a plain call: nothing is awaited, so it works when the loop is gone

    await assert_all_gone(*pids)
    with pytest.raises(TerminalError):
        manager.start("late", "true")  # no more terminals
