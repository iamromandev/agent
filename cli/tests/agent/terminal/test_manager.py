from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import time
from collections.abc import Generator
from pathlib import Path

import pytest
from src.agent.terminal import manager as manager_module
from src.agent.terminal import pty as pty_module
from src.agent.terminal.manager import GRACE, MAX_TERMINALS, Terminal, TerminalError, TerminalManager

from tests.helpers import assert_all_gone, wait_for_pids, wait_until
from tests.helpers import gone as gone_pid


def _text(terminal: Terminal) -> str:
    return "\n".join(terminal.buffer.tail(100))


async def _exited(terminal: Terminal) -> None:
    await wait_until(lambda: terminal.status[0] == "exited")


async def test_a_started_terminal_runs_and_is_listed(manager: TerminalManager) -> None:
    terminal = manager.start("api", "sleep 30")

    assert terminal.name == "api"
    assert terminal.command == "sleep 30"
    assert terminal.status == ("running", None)
    assert manager.get("api") is terminal
    assert manager.names() == ["api"]
    assert manager.get("other") is None


async def test_the_output_is_kept_as_plain_lines(manager: TerminalManager) -> None:
    terminal = manager.start("a", "printf '\\033[31mred\\033[0m\\nplain\\n'")

    await wait_until(lambda: terminal.buffer.tail(5) == ["red", "plain"])


async def test_the_command_runs_in_the_workspace(manager: TerminalManager, workspace: Path) -> None:
    terminal = manager.start("a", "pwd -P")

    await wait_until(lambda: str(workspace) in _text(terminal))


async def test_the_environment_asks_for_color_and_is_a_terminal(
    manager: TerminalManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "9")
    monkeypatch.setenv("LINES", "9")
    terminal = manager.start("a", 'echo "[$TERM][$FORCE_COLOR][${NO_COLOR-unset}][${COLUMNS-unset}][${LINES-unset}]"')

    await wait_until(lambda: "[xterm-256color][1][unset][unset][unset]" in _text(terminal))


async def test_the_terminal_is_sized_from_the_manager(workspace: Path) -> None:
    sized = TerminalManager(workspace, cols=100, rows=30, grace=0.3)
    try:
        terminal = sized.start("a", "stty size")
        await wait_until(lambda: "30 100" in _text(terminal))
    finally:
        await sized.close_all()


async def test_without_a_command_the_user_shell_runs_interactively(manager: TerminalManager) -> None:
    terminal = manager.start("shell", None)
    assert terminal.command is None

    terminal.write(b"echo from-the-shell\n")

    await wait_until(lambda: "from-the-shell" in _text(terminal))
    assert terminal.status == ("running", None)


async def test_an_unusable_shell_falls_back_to_bin_sh(
    manager: TerminalManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHELL", "/no/such/shell")
    terminal = manager.start("a", "echo fell-back")

    await wait_until(lambda: "fell-back" in _text(terminal))


async def test_a_subscriber_gets_the_raw_bytes_until_it_unsubscribes(manager: TerminalManager) -> None:
    terminal = manager.start("a", "cat")
    seen = bytearray()
    unsubscribe = terminal.subscribe(seen.extend)

    terminal.write(b"one\n")
    await wait_until(lambda: b"one" in seen)
    unsubscribe()
    unsubscribe()  # twice is fine
    mark = len(seen)
    terminal.write(b"two\n")
    await wait_until(lambda: "two" in _text(terminal))

    assert len(seen) == mark


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


async def test_a_normal_exit_gives_its_code_and_all_its_output(manager: TerminalManager) -> None:
    terminal = manager.start("a", "echo a; echo b; exit 3")
    exits: list[Terminal] = []
    terminal.on_exit(exits.append)

    await _exited(terminal)

    assert terminal.status == ("exited", 3)
    assert terminal.buffer.tail(5) == ["a", "b"]
    await wait_until(lambda: exits == [terminal])


async def test_a_signal_death_is_a_negative_code(manager: TerminalManager) -> None:
    terminal = manager.start("a", "kill -KILL $$")

    await _exited(terminal)

    assert terminal.status == ("exited", -signal.SIGKILL)


async def test_a_command_that_cannot_run_exits_127_with_its_message(manager: TerminalManager) -> None:
    terminal = manager.start("a", "definitely-not-a-program-xyz")

    await _exited(terminal)

    assert terminal.status == ("exited", 127)
    assert "not found" in _text(terminal)


async def test_an_exit_listener_added_after_the_exit_is_called(manager: TerminalManager) -> None:
    terminal = manager.start("a", "exit 0")
    await _exited(terminal)
    exits: list[Terminal] = []

    terminal.on_exit(exits.append)

    await wait_until(lambda: exits == [terminal])


async def test_written_bytes_reach_the_process(manager: TerminalManager) -> None:
    terminal = manager.start("a", "cat")

    terminal.write(b"typed\n")

    await wait_until(lambda: "typed" in _text(terminal))


async def test_resize_is_seen_by_the_process(manager: TerminalManager) -> None:
    terminal = manager.start("a", "trap 'stty size' WINCH; echo ready; while :; do sleep 0.05; done")
    await wait_until(lambda: "ready" in _text(terminal))

    terminal.resize(90, 33)

    await wait_until(lambda: "33 90" in _text(terminal))


async def test_stop_ends_a_process_that_obeys_sigterm(manager: TerminalManager) -> None:
    terminal = manager.start("a", "sleep 30")

    status = await terminal.stop()

    assert status == ("exited", -signal.SIGTERM)
    assert terminal.status == status


async def test_stop_kills_a_process_that_ignores_sigterm_after_the_grace(manager: TerminalManager) -> None:
    terminal = manager.start("a", "trap '' TERM; echo ready; while :; do sleep 0.1; done")
    await wait_until(lambda: "ready" in _text(terminal))

    started = time.monotonic()
    status = await terminal.stop(grace=0.3)

    assert status == ("exited", -signal.SIGKILL)
    assert 0.25 <= time.monotonic() - started < 2.5


async def test_stop_takes_the_children_down_too(manager: TerminalManager, workspace: Path) -> None:
    shell_pid, child_pid = workspace / "shell.pid", workspace / "child.pid"
    terminal = manager.start("a", f"echo $$ > {shell_pid}; sleep 30 & echo $! > {child_pid}; wait")
    pids = await wait_for_pids(shell_pid, child_pid)

    await terminal.stop()

    await assert_all_gone(*pids)


async def test_stop_of_an_exited_terminal_reports_its_status(manager: TerminalManager) -> None:
    terminal = manager.start("a", "exit 4")
    await _exited(terminal)

    assert await terminal.stop() == ("exited", 4)


async def test_stop_works_when_the_leader_exited_but_a_child_keeps_the_terminal(
    manager: TerminalManager, workspace: Path
) -> None:
    child_pid = workspace / "child.pid"
    terminal = manager.start("a", f"sleep 30 & echo $! > {child_pid}; exit 0")
    (pid,) = await wait_for_pids(child_pid)

    status = await terminal.stop()

    assert status[0] == "exited"
    await assert_all_gone(pid)


async def test_stop_right_after_start_is_not_missed(manager: TerminalManager) -> None:
    for index in range(10):
        terminal = manager.start(f"t{index}", "sleep 30")

        assert await terminal.stop() == ("exited", -signal.SIGTERM)


async def test_the_default_grace_is_three_seconds() -> None:
    assert GRACE == 3.0


@pytest.mark.parametrize("name", ["", "has space", "a/b", "x" * 33, "né", "a.b", "a\n"])
async def test_an_invalid_name_is_refused(manager: TerminalManager, name: str) -> None:
    with pytest.raises(TerminalError, match="name"):
        manager.start(name, "true")

    assert manager.names() == []


@pytest.mark.parametrize("name", ["a", "api-1", "Dev_server", "x" * 32])
async def test_a_valid_name_is_accepted(manager: TerminalManager, name: str) -> None:
    manager.start(name, "true")


async def test_a_name_in_use_by_a_running_terminal_is_refused(manager: TerminalManager) -> None:
    first = manager.start("api", "sleep 30")

    with pytest.raises(TerminalError, match="already running"):
        manager.start("api", "sleep 31")

    assert manager.get("api") is first


async def test_a_name_of_an_exited_terminal_can_be_started_again(manager: TerminalManager) -> None:
    old = manager.start("api", "exit 1")
    await _exited(old)

    new = manager.start("api", "sleep 30")

    assert manager.get("api") is new
    assert new is not old
    assert manager.names() == ["api"]


async def test_a_missing_workspace_is_refused(tmp_path: Path) -> None:
    broken = TerminalManager(tmp_path / "gone")

    with pytest.raises(TerminalError, match="workspace"):
        broken.start("a", "true")


async def test_the_scrollback_stays_bounded(workspace: Path) -> None:
    small = TerminalManager(workspace, scrollback=50, grace=0.3)
    try:
        terminal = small.start("a", "i=0; while [ $i -lt 2000 ]; do echo line $i; i=$((i+1)); done")
        await _exited(terminal)

        assert len(terminal.buffer) <= 50
        assert terminal.buffer.tail(1) == ["line 1999"]
    finally:
        await small.close_all()


async def test_listeners_hear_started_exited_and_closed(manager: TerminalManager) -> None:
    events: list[tuple[str, str]] = []
    manager.listen(lambda name, event: events.append((name, event)))

    terminal = manager.start("a", "exit 0")
    await wait_until(lambda: ("a", "exited") in events)
    await manager.close("a")

    assert events == [("a", "started"), ("a", "exited"), ("a", "closed")]
    assert terminal.status == ("exited", 0)


async def test_restarting_an_exited_name_reports_the_old_one_closed(manager: TerminalManager) -> None:
    old = manager.start("a", "exit 0")
    await _exited(old)
    events: list[tuple[str, str]] = []
    manager.listen(lambda name, event: events.append((name, event)))

    manager.start("a", "sleep 30")

    assert events == [("a", "closed"), ("a", "started")]


async def test_close_stops_a_running_terminal_and_forgets_it(manager: TerminalManager) -> None:
    terminal = manager.start("a", "sleep 30")

    await manager.close("a")

    assert terminal.status[0] == "exited"
    assert manager.get("a") is None
    assert manager.names() == []


async def test_closing_an_unknown_name_is_quiet(manager: TerminalManager) -> None:
    await manager.close("nope")


async def test_close_all_leaves_no_process_behind(manager: TerminalManager, workspace: Path) -> None:
    files: list[Path] = []
    for index in range(3):
        shell, child = workspace / f"s{index}.pid", workspace / f"c{index}.pid"
        files += [shell, child]
        manager.start(f"t{index}", f"echo $$ > {shell}; sleep 1000 & echo $! > {child}; sleep 1000; wait")
    pids = await wait_for_pids(*files)

    await manager.close_all()

    await assert_all_gone(*pids)
    assert manager.names() == []


async def test_close_all_waits_one_grace_for_all_the_terminals(manager: TerminalManager) -> None:
    for index in range(4):
        manager.start(f"t{index}", "trap '' TERM; echo ready; while :; do sleep 0.1; done")
    terminals = [t for i in range(4) if (t := manager.get(f"t{i}")) is not None]
    await wait_until(lambda: all("ready" in _text(t) for t in terminals))

    started = time.monotonic()
    await manager.close_all()

    assert time.monotonic() - started < 0.3 * 2 + 1.0  # one grace (0.3), not four


async def test_a_manager_that_was_closed_starts_nothing(manager: TerminalManager) -> None:
    await manager.close_all()

    with pytest.raises(TerminalError, match="closed"):
        manager.start("late", "sleep 30")


async def test_stopping_the_user_shell_does_not_wait_out_the_grace(workspace: Path) -> None:
    """An interactive shell ignores SIGTERM, so it is also sent SIGHUP, as closing a terminal window does."""
    patient = TerminalManager(workspace, grace=10.0)
    try:
        terminal = patient.start("shell", None)
        terminal.write(b"echo ready\n")
        await wait_until(lambda: "ready" in _text(terminal))

        started = time.monotonic()
        status = await terminal.stop(grace=10.0)

        assert status[0] == "exited"
        assert time.monotonic() - started < 3.0
    finally:
        await patient.close_all()


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


async def test_stop_kills_a_child_that_ignores_term_and_hup_after_the_leader_died(
    manager: TerminalManager, workspace: Path
) -> None:
    child = workspace / "child.pid"
    command = f"(trap '' TERM HUP; exec sleep 60) & echo $! > {child}; wait"
    with _killed_after(child):
        terminal = manager.start("a", command)
        (pid,) = await wait_for_pids(child)

        status = await terminal.stop(grace=0.3)

        assert status == ("exited", -signal.SIGTERM)
        await assert_all_gone(pid)


async def test_close_all_kills_a_child_that_ignores_term_and_hup(manager: TerminalManager, workspace: Path) -> None:
    child = workspace / "child.pid"
    command = f"(trap '' TERM HUP; exec sleep 60) & echo $! > {child}; wait"
    with _killed_after(child):
        terminal = manager.start("a", command)
        (pid,) = await wait_for_pids(child)

        await manager.close_all()

        assert terminal.status == ("exited", -signal.SIGTERM)
        await assert_all_gone(pid)


async def test_a_child_is_given_the_grace_to_stop_on_term_before_it_is_killed(
    manager: TerminalManager, workspace: Path
) -> None:
    child, mark = workspace / "child.pid", workspace / "mark"
    # HUP is ignored: when the leader exits at once the kernel hangs up the tty, which would otherwise kill the
    # child before its TERM handler has written the mark (a race of the test, not of the terminal)
    command = f"(trap '' HUP; trap 'echo term > {mark}; exit 0' TERM; echo $$ > {child}; while :; do sleep 0.05; done) & wait"
    with _killed_after(child):
        terminal = manager.start("a", command)
        await wait_for_pids(child)

        await terminal.stop(grace=2.0)

        # the leader dies at once; its child got SIGTERM too (so it could clean up) and was not killed first
        await wait_until(mark.exists)


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


@pytest.mark.parametrize("shell", ["bash", "zsh"])
async def test_stop_kills_a_foreground_job_of_an_interactive_bash_or_zsh(
    shell: str, manager: TerminalManager, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = shutil.which(shell)
    if path is None:
        pytest.skip(f"{shell} is not installed")
    monkeypatch.setenv("SHELL", path)
    monkeypatch.setenv("HOME", str(workspace))  # no user rc files
    job = workspace / "job.pid"
    with _killed_after(job):
        terminal = manager.start("shell", None)
        terminal.write(f"sh -c 'trap \"\" TERM HUP; echo $$ > {job}; exec sleep 60'\n".encode())
        (pid,) = await wait_for_pids(job)

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


async def test_close_forgets_the_terminal_and_reports_it_closed_once(manager: TerminalManager) -> None:
    events: list[tuple[str, str]] = []
    manager.listen(lambda name, event: events.append((name, event)))
    manager.start("a", "trap '' TERM; echo ready; while :; do sleep 0.1; done")
    await wait_until(lambda: ("a", "started") in events)

    await asyncio.gather(manager.close("a"), manager.close("a"))

    assert events.count(("a", "closed")) == 1
    assert manager.names() == []


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


async def test_a_leader_that_is_not_reaped_after_sigkill_is_not_reported_dead_until_it_is(
    manager: TerminalManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manager_module, "_KILL_WAIT", 0.2)
    monkeypatch.setattr(manager_module, "_POLL_MAX", 0.1)
    terminal = manager.start("a", "sleep 30")
    real_poll = terminal.process.poll
    stuck = True
    monkeypatch.setattr(terminal.process, "poll", lambda: None if stuck else real_poll())

    status = await terminal.stop(grace=0.1)

    assert status == ("running", None)
    assert terminal.status == ("running", None)
    stuck = False
    await _exited(terminal)  # the reaper keeps polling in the background
    assert terminal.status == ("exited", -signal.SIGTERM)  # what it really died of


async def test_the_reaper_backs_off_while_the_leader_is_not_reapable(
    manager: TerminalManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal = manager.start("a", "sleep 30")
    real_poll = terminal.process.poll
    polls = 0
    stuck = True

    def poll() -> int | None:
        nonlocal polls
        polls += 1
        return None if stuck else real_poll()

    monkeypatch.setattr(terminal.process, "poll", poll)
    terminal.process.signal_group(signal.SIGKILL)  # the pty closes: the reaper starts waiting for the leader

    await asyncio.sleep(1.0)

    assert polls < 15  # at a fixed 20 ms that would be about 50
    stuck = False
    await wait_until(lambda: terminal.status[0] == "exited", timeout=5)


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


async def test_stop_kills_the_background_jobs_of_the_user_shell(manager: TerminalManager, workspace: Path) -> None:
    terminal, pids, files = await _start_shell_with_background_jobs(manager, workspace)
    with _killed_after(*files):
        await terminal.stop(grace=0.3)

        assert terminal.status[0] == "exited"
        await assert_all_gone(*pids)


async def test_close_all_kills_the_background_jobs_of_the_user_shell(manager: TerminalManager, workspace: Path) -> None:
    terminal, pids, files = await _start_shell_with_background_jobs(manager, workspace)
    with _killed_after(*files):
        await manager.close_all()

        assert terminal.status[0] == "exited"
        await assert_all_gone(*pids)


@pytest.mark.parametrize("shell", ["bash", "zsh"])
async def test_stop_kills_the_background_jobs_of_an_interactive_bash_or_zsh(
    shell: str, manager: TerminalManager, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = shutil.which(shell)
    if path is None:
        pytest.skip(f"{shell} is not installed")
    monkeypatch.setenv("SHELL", path)
    monkeypatch.setenv("HOME", str(workspace))  # no user rc files
    terminal, pids, files = await _start_shell_with_background_jobs(manager, workspace)
    with _killed_after(*files):
        await terminal.stop(grace=0.3)

        assert terminal.status[0] == "exited"
        await assert_all_gone(*pids)


async def test_the_exit_hook_kills_the_background_jobs_of_the_user_shell(
    manager: TerminalManager, workspace: Path
) -> None:
    terminal, pids, files = await _start_shell_with_background_jobs(manager, workspace)
    with _killed_after(*files):
        pty_module._kill_live_groups()

        await wait_until(lambda: terminal.status[0] == "exited")
        await assert_all_gone(*pids)


async def test_a_shell_that_exits_on_its_own_does_not_leave_its_background_jobs(
    manager: TerminalManager, workspace: Path
) -> None:
    terminal, pids, files = await _start_shell_with_background_jobs(manager, workspace)
    with _killed_after(*files):
        terminal.write(b"exit\n")

        await wait_until(lambda: terminal.status[0] == "exited")
        await assert_all_gone(*pids)


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


async def test_a_leftover_member_is_killed_when_the_leader_is_reaped_after_a_stuck_stop(
    manager: TerminalManager, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manager_module, "_KILL_WAIT", 0.2)
    monkeypatch.setattr(manager_module, "_POLL_MAX", 0.1)
    child = workspace / "child.pid"
    with _killed_after(child):
        terminal = manager.start("a", f"(trap '' TERM HUP; exec sleep 60) & echo $! > {child}; wait")
        (pid,) = await wait_for_pids(child)
        process = terminal.process
        real_poll, real_signal_tree = process.poll, process.signal_tree
        stuck = True
        monkeypatch.setattr(process, "poll", lambda: None if stuck else real_poll())
        # only the leader is signalled, so the child outlives the stop
        monkeypatch.setattr(process, "signal_tree", lambda sig: os.kill(process.pid, sig))

        assert await terminal.stop(grace=0.1) == ("running", None)
        assert not gone_pid(pid)

        monkeypatch.setattr(process, "signal_tree", real_signal_tree)
        stuck = False
        await _exited(terminal)  # reaped by the background reaper, which now kills what is left

        await assert_all_gone(pid)


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


async def test_kill_all_with_nothing_running_is_quiet(manager: TerminalManager) -> None:
    manager.kill_all()
    manager.kill_all()  # twice is fine
