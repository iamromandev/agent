from __future__ import annotations

import asyncio
import contextlib
import os
import resource
import signal
import time
from pathlib import Path

import pytest
from src.agent.terminal import pty as pty_module
from src.agent.terminal.pty import PtyProcess

from tests.agent.terminal.conftest import SpawnFn, read_until, wait_exit
from tests.helpers import assert_all_gone, gone, wait_for_pids, wait_until


def test_the_process_output_is_read_from_the_master(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sh", "-c", "echo hello"])

    assert b"hello" in read_until(process, b"hello")


def test_the_process_runs_in_the_given_folder(spawn_pty: SpawnFn, tmp_path: Path) -> None:
    folder = tmp_path / "work"
    folder.mkdir()
    process = spawn_pty(["sh", "-c", "pwd -P"], cwd=folder)

    assert str(folder.resolve()).encode() in read_until(process, folder.name.encode())


def test_the_process_gets_exactly_the_given_environment(spawn_pty: SpawnFn, pty_env: dict[str, str]) -> None:
    process = spawn_pty(["sh", "-c", 'echo "[$MARK][${HOME-unset}]"'], env={**pty_env, "MARK": "m1"})

    assert b"[m1][unset]" in read_until(process, b"[m1][unset]")


def test_the_child_sees_a_terminal(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sh", "-c", "[ -t 0 ] && [ -t 1 ] && echo tty-yes || echo tty-no"])

    assert b"tty-yes" in read_until(process, b"tty-")


def test_the_pty_starts_at_the_requested_size(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sh", "-c", "stty size"], cols=100, rows=30)

    assert b"30 100" in read_until(process, b"30 100")


def test_resize_changes_what_the_process_sees(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sh", "-c", "trap 'stty size' WINCH; echo ready; while :; do sleep 0.05; done"])
    read_until(process, b"ready")

    process.resize(120, 40)

    assert b"40 120" in read_until(process, b"40 120")


def test_bytes_written_reach_the_process(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["cat"])
    process.write(b"ping\n")

    assert b"ping" in read_until(process, b"ping")


async def test_a_large_write_is_delivered_whole(spawn_pty: SpawnFn) -> None:
    # raw mode: in canonical mode the line discipline holds back input that has no newline
    process = spawn_pty(["sh", "-c", "stty raw -echo; echo ready; head -c 200000 > /dev/null; echo done-reading"])
    read_until(process, b"ready")

    process.write(b"x" * 200_000)  # far more than the pty queue takes: the rest waits for the loop to flush

    seen = bytearray()

    def finished() -> bool:
        seen.extend(_drain(process))
        return b"done-reading" in seen

    await wait_until(finished)


def _drain(process: PtyProcess) -> bytes:
    """What the process has written so far (nothing when it has written nothing)."""
    with contextlib.suppress(OSError):  # nothing to read now, or EIO
        return os.read(process.fd, 65_536)
    return b""


async def test_a_write_to_a_process_that_does_not_read_returns_at_once(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sh", "-c", "stty raw -echo; echo ready; sleep 30"])
    read_until(process, b"ready")

    started = time.monotonic()
    process.write(b"y" * 100_000)

    assert time.monotonic() - started < 0.5


async def test_queued_input_keeps_its_order_across_writes(spawn_pty: SpawnFn, tmp_path: Path) -> None:
    out = tmp_path / "received"
    process = spawn_pty(["sh", "-c", f"stty raw -echo; echo ready; sleep 0.3; head -c 150003 > {out}; echo done"])
    read_until(process, b"ready")
    payload = b"".join(b"%06d" % i for i in range(25_000)) + b"end"

    process.write(payload[:100_000])
    process.write(payload[100_000:])

    await wait_until(lambda: out.exists() and out.stat().st_size == len(payload), timeout=10)
    assert out.read_bytes() == payload
    read_until(process, b"done")  # an unread tty output holds back the shell's exit on macOS


async def test_input_beyond_the_cap_is_refused_with_a_clear_error(
    spawn_pty: SpawnFn, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pty_module, "MAX_PENDING", 10_000)
    process = spawn_pty(["sh", "-c", "stty raw -echo; echo ready; sleep 30"])
    read_until(process, b"ready")

    with pytest.raises(pty_module.WriteOverflow, match="input dropped"):
        for _ in range(100):
            process.write(b"b" * 4_000)


async def test_an_oversized_write_sends_nothing_and_a_normal_one_is_delivered(
    spawn_pty: SpawnFn, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pty_module, "MAX_PENDING", 10_000)
    out = tmp_path / "received"
    process = spawn_pty(["sh", "-c", f"stty raw -echo; echo ready; exec cat > {out}"])
    read_until(process, b"ready")

    with pytest.raises(pty_module.WriteOverflow, match="input dropped"):
        process.write(b"x" * 20_000)  # not a single byte of it may go out
    await asyncio.sleep(0.2)
    assert not out.exists() or out.read_bytes() == b""

    process.write(b"fine")  # under the cap: delivered

    await wait_until(lambda: out.exists() and out.read_bytes() == b"fine")


async def test_closing_drops_the_queued_input_and_stops_watching(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sh", "-c", "stty raw -echo; echo ready; sleep 30"])
    read_until(process, b"ready")
    process.write(b"z" * 100_000)

    process.close()

    process.write(b"late")  # dropped quietly
    await asyncio.sleep(0.05)  # a stale writer on a closed descriptor would blow up the loop


def test_a_normal_exit_gives_its_code(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sh", "-c", "exit 3"])

    assert wait_exit(process) == 3
    assert process.poll() == 3  # asked again, it says the same


def test_a_signal_death_is_a_negative_code(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sh", "-c", "kill -TERM $$; sleep 5"])

    assert wait_exit(process) == -signal.SIGTERM


def test_a_missing_program_exits_127(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["definitely-not-a-program-xyz"])

    assert wait_exit(process) == 127


def test_poll_is_none_while_the_process_runs(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sleep", "5"])

    assert process.poll() is None


async def test_signal_group_reaches_the_children_too(spawn_pty: SpawnFn, tmp_path: Path) -> None:
    shell_pid, child_pid = tmp_path / "shell.pid", tmp_path / "child.pid"
    process = spawn_pty(["sh", "-c", f"echo $$ > {shell_pid}; sleep 30 & echo $! > {child_pid}; wait"])
    pids = await wait_for_pids(shell_pid, child_pid)

    process.signal_group(signal.SIGKILL)

    assert wait_exit(process) == -signal.SIGKILL
    await assert_all_gone(*pids)


def test_the_process_leads_its_own_group_and_session(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sleep", "5"])

    assert os.getpgid(process.pid) == process.pid
    assert os.getsid(process.pid) == process.pid != os.getsid(0)


def test_signalling_a_gone_group_is_quiet(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sh", "-c", "exit 0"])
    wait_exit(process)

    process.signal_group(signal.SIGTERM)


def test_writing_after_the_process_exited_is_dropped(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sh", "-c", "exit 0"])
    wait_exit(process)

    process.write(b"late input\n")


def test_close_closes_the_master_once(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sleep", "5"])
    fd = process.fd

    process.close()
    process.close()  # no error, and it must not close an unrelated descriptor twice

    with pytest.raises(OSError):
        os.fstat(fd)


def test_the_master_is_not_inherited_by_other_children(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sleep", "5"])

    assert not os.get_inheritable(process.fd)
    assert not os.get_blocking(process.fd)


def test_the_exit_hook_kills_the_groups_still_alive(spawn_pty: SpawnFn) -> None:
    process = spawn_pty(["sleep", "30"])

    pty_module._kill_live_groups()

    assert wait_exit(process) == -signal.SIGKILL


def test_a_closed_process_whose_group_is_alive_is_still_killed_by_the_exit_hook(spawn_pty: SpawnFn) -> None:
    # closing the master hangs the terminal up, so the process must ignore SIGHUP to still be there
    process = spawn_pty(["sh", "-c", "trap '' HUP; echo ready; sleep 30"])
    read_until(process, b"ready")
    process.close()

    pty_module._kill_live_groups()

    assert wait_exit(process) == -signal.SIGKILL


def test_a_gone_group_is_never_signalled_again(spawn_pty: SpawnFn, monkeypatch: pytest.MonkeyPatch) -> None:
    process = spawn_pty(["sh", "-c", "exit 0"])
    wait_exit(process)
    assert not process.group_alive()  # reaped and empty: the id may now belong to somebody else
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))

    pty_module._kill_live_groups()
    process.signal_group(signal.SIGKILL)
    process.signal_tree(signal.SIGKILL)

    assert sent == []


def test_group_alive_sees_a_child_that_outlives_the_leader(spawn_pty: SpawnFn, tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    process = spawn_pty(["sh", "-c", f"(trap '' HUP TERM; exec sleep 30) & echo $! > {pid_file}"])
    assert wait_exit(process) == 0
    deadline = time.monotonic() + 5
    while not (pid_file.exists() and pid_file.read_text().strip()):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    child = int(pid_file.read_text())
    try:
        assert process.group_alive()

        process.signal_tree(signal.SIGKILL)

        deadline = time.monotonic() + 5
        while process.group_alive():
            assert time.monotonic() < deadline, "the group did not go"
            time.sleep(0.01)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child, signal.SIGKILL)


def test_signal_tree_reaches_a_foreground_job_in_its_own_group(spawn_pty: SpawnFn, tmp_path: Path) -> None:
    pid_file = tmp_path / "job.pid"
    process = spawn_pty(["sh", "-i"])
    process.write(f"sh -c 'echo $$ > {pid_file}; exec sleep 30'\n".encode())
    deadline = time.monotonic() + 5
    while not (pid_file.exists() and pid_file.read_text().strip()):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    job = int(pid_file.read_text())
    try:
        if os.getpgid(job) == process.pid:
            pytest.skip("this shell has no job control")

        process.signal_tree(signal.SIGKILL)

        # reaping the shell lets init adopt, and then reap, the job; on macOS a shell with unread output cannot exit
        deadline = time.monotonic() + 5
        while (process.poll() is None or not gone(job)) and time.monotonic() < deadline:
            _drain(process)
            time.sleep(0.01)
        assert gone(job)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(job, signal.SIGKILL)


def test_group_alive_and_signal_tree_see_a_job_in_its_own_group_of_the_session(
    spawn_pty: SpawnFn, tmp_path: Path
) -> None:
    pid_file = tmp_path / "job.pid"
    process = spawn_pty(["sh", "-c", f"set -m; (trap '' HUP TERM; exec sleep 30) & echo $! > {pid_file}; wait"])
    deadline = time.monotonic() + 5
    while not (pid_file.exists() and pid_file.read_text().strip()):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    job = int(pid_file.read_text())
    try:
        if os.getpgid(job) == process.pid:
            pytest.skip("this shell has no job control")
        os.kill(process.pid, signal.SIGKILL)  # the leader goes; the job is left in a group of its own
        assert wait_exit(process) == -signal.SIGKILL
        assert os.getsid(job) == process.pid

        assert process.group_alive()
        process.signal_tree(signal.SIGKILL)

        deadline = time.monotonic() + 5
        while process.group_alive():
            assert time.monotonic() < deadline, "the session did not go"
            time.sleep(0.01)
        assert gone(job)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(job, signal.SIGKILL)


def test_a_session_without_members_is_gone_for_good(spawn_pty: SpawnFn, monkeypatch: pytest.MonkeyPatch) -> None:
    process = spawn_pty(["sh", "-c", "exit 0"])
    wait_exit(process)
    assert not process.group_alive()
    scans: list[int] = []
    monkeypatch.setattr(pty_module, "_all_pids", lambda: scans.append(1) or [])

    assert not process.group_alive()  # reaped and empty: the id may be reused, nothing is looked at any more

    assert scans == []


def test_spawning_works_when_descriptors_are_numbered_above_1024(
    spawn_pty: SpawnFn,
) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = 1_300
    if hard != resource.RLIM_INFINITY and hard < wanted:
        pytest.skip("the descriptor limit cannot be raised enough")
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, wanted), hard))
    held: list[int] = []
    try:
        while not held or held[-1] < 1_100:
            held.append(os.dup(0))
        process = spawn_pty(["sh", "-c", "echo hello"])

        assert b"hello" in read_until(process, b"hello")
    finally:
        for fd in held:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


# --- spawn failing ---------------------------------------------------------------------------------


def _open_fds() -> int:
    return len(os.listdir("/dev/fd"))


def _record_forks(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    pids: list[int] = []
    real_fork = os.fork

    def fork() -> int:
        pid = real_fork()
        if pid:
            pids.append(pid)
        return pid

    monkeypatch.setattr(os, "fork", fork)
    return pids


def test_a_child_that_never_execs_is_killed_and_reaped_after_the_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pty_env: dict[str, str]
) -> None:
    monkeypatch.setattr(pty_module, "_SPAWN_WAIT", 0.3)
    monkeypatch.setattr(os, "execvpe", lambda *_: time.sleep(60))  # the child hangs before exec
    pids = _record_forks(monkeypatch)
    fds = _open_fds()

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="did not start"):
        PtyProcess.spawn(["sleep", "30"], tmp_path, pty_env, 80, 24)

    assert time.monotonic() - started < 3
    assert len(pids) == 1
    assert gone(pids[0])
    assert _open_fds() == fds


def test_an_interrupted_spawn_kills_and_reaps_the_child_and_closes_its_fds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pty_env: dict[str, str]
) -> None:
    pids = _record_forks(monkeypatch)
    fds = _open_fds()

    def interrupt(fd: int, timeout: float) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(pty_module, "_wait_readable", interrupt)

    with pytest.raises(KeyboardInterrupt):
        PtyProcess.spawn(["sleep", "30"], tmp_path, pty_env, 80, 24)

    assert len(pids) == 1
    assert gone(pids[0])
    assert _open_fds() == fds


def test_a_failing_pipe_does_not_leak_the_pty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pty_env: dict[str, str]) -> None:
    def broken() -> tuple[int, int]:
        raise OSError("no pipes")

    monkeypatch.setattr(os, "pipe", broken)
    fds = _open_fds()

    with pytest.raises(OSError, match="no pipes"):
        PtyProcess.spawn(["sleep", "30"], tmp_path, pty_env, 80, 24)

    assert _open_fds() == fds


def test_a_signal_sent_right_after_spawn_is_not_missed(spawn_pty: SpawnFn) -> None:
    for _ in range(20):
        process = spawn_pty(["sleep", "30"])
        process.signal_group(signal.SIGKILL)

        assert wait_exit(process, timeout=2) == -signal.SIGKILL
