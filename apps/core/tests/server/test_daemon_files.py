"""The daemon's files: a private run folder, a 0600 socket, one daemon per workspace (AC3, NFR1)."""

from __future__ import annotations

import os
import socket
import stat
from pathlib import Path

import pytest
from core.server.files import AlreadyRunning, PidLock, bind_socket
from core.server.paths import RunDirError, log_path, pid_path, run_dir, socket_path
from core.server.serve import serve


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def test_the_run_folder_is_made_private_and_holds_the_files(data_home: Path, tmp_path: Path) -> None:
    folder = run_dir()
    assert folder == data_home / "agent" / "run" and _mode(folder) == 0o700
    ws = tmp_path / "ws"
    for path in (socket_path(ws), pid_path(ws), log_path(ws)):
        assert path.parent == folder
    assert socket_path(ws).name.endswith(".sock") and len(socket_path(ws).stem) == 16
    assert socket_path(tmp_path / "other") != socket_path(ws)


@pytest.mark.parametrize("problem", ["symlink", "mode", "owner"])
def test_a_run_folder_that_is_not_ours_alone_is_refused(
    data_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    folder = data_home / "agent" / "run"
    folder.parent.mkdir(parents=True)
    if problem == "symlink":
        (tmp_path / "real").mkdir(mode=0o700)
        folder.symlink_to(tmp_path / "real")
    else:
        folder.mkdir(mode=0o700)
        if problem == "mode":
            folder.chmod(0o755)
        else:
            uid = os.getuid()
            monkeypatch.setattr(os, "getuid", lambda: uid + 1)
    with pytest.raises(RunDirError):
        run_dir()


def test_a_long_data_folder_falls_back_to_a_private_temp_folder(
    monkeypatch: pytest.MonkeyPatch, data_home: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home / ("x" * 90)))
    monkeypatch.setenv("TMPDIR", str(data_home))
    folder = run_dir()
    assert folder == data_home / f"agent-{os.getuid()}" and _mode(folder) == 0o700
    assert len(str(socket_path(tmp_path))) <= 100


def test_the_socket_is_0600_and_unix_only(data_home: Path, tmp_path: Path) -> None:
    path = socket_path(tmp_path)
    sock = bind_socket(path)
    try:
        assert _mode(path) == 0o600 and stat.S_ISSOCK(os.lstat(path).st_mode)
        assert sock.family == socket.AF_UNIX
    finally:
        sock.close()


def test_agent_serve_exits_1_naming_the_running_daemon(data_home: Path, tmp_path: Path, capsys) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    held = PidLock.acquire(pid_path(ws))
    try:
        assert serve(ws) == 1
    finally:
        held.release()
    assert f"agent daemon already running for {ws.resolve()} (pid {os.getpid()})" in capsys.readouterr().err
    assert not socket_path(ws).exists()


def test_a_second_daemon_for_the_folder_is_refused_naming_the_pid(data_home: Path, tmp_path: Path) -> None:
    first = PidLock.acquire(pid_path(tmp_path))
    try:
        assert pid_path(tmp_path).read_text().strip() == str(os.getpid())
        with pytest.raises(AlreadyRunning) as info:
            PidLock.acquire(pid_path(tmp_path))  # another descriptor: flock conflicts even in this process
        assert info.value.pid == os.getpid()
    finally:
        first.release()
    PidLock.acquire(pid_path(tmp_path)).release()  # free again
