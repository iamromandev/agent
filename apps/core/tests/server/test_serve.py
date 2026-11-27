"""Starting and stopping the daemon process: what runs in the background, how `stop` knows it is gone, and the
cleanup a signal during shutdown must not cut short."""

from __future__ import annotations

import asyncio
import os
import signal
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from core.engine.commands.builtins import BUILTINS
from core.runtime import build_runtime
from core.server import serve as served
from core.server import serve as served_module
from core.server.daemon import Daemon
from core.server.files import PidLock
from core.server.paths import log_path, pid_path, socket_path
from core.testing.fakes import FakeCatalog, FakeConnector, ScriptedProvider


def _plant(ws: Path, marker: Path) -> None:
    """Modules a hostile workspace could hold to run code if the daemon's interpreter looked in its folder."""
    code = f"open({str(marker)!r}, 'a').write(__name__ + '\\n')\n"
    (ws / "core" / "server").mkdir(parents=True)
    (ws / "core" / "__init__.py").write_text(code)
    (ws / "core" / "server" / "__init__.py").write_text(code)
    (ws / "core" / "server" / "__main__.py").write_text(code)
    for name in ("uvicorn.py", "httpx.py", "argparse.py"):
        (ws / name).write_text(code)


def test_the_daemon_command_never_imports_from_the_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    marker = tmp_path / "ran"
    _plant(ws, marker)
    argv = served.daemon_argv(ws)
    assert argv[:4] == [sys.executable, "-P", "-m", "core.server"]

    # `--help` stops before any runtime is built, after `core.server.__main__` and argparse were imported
    done = subprocess.run([*argv[:4], "--help"], cwd=ws, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and "serve" in done.stdout
    assert not marker.exists(), marker.read_text()


def test_start_background_starts_the_daemon_detached_with_its_log_and_waits_for_it(
    data_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    started: list[dict[str, object]] = []

    class FakeProcess:
        def __init__(self, argv: list[str], **kwargs: object) -> None:
            started.append({"argv": argv, **kwargs})

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(served.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(served, "hello", lambda workspace, timeout=1.0: {"v": 1} if started else None)

    assert served.start_background(ws) == socket_path(ws)
    (call,) = started
    assert call["argv"] == served.daemon_argv(ws.resolve())
    assert call["start_new_session"] is True and call["cwd"] == ws.resolve() and call["stdin"] == subprocess.DEVNULL
    assert stat.S_IMODE(os.stat(log_path(ws)).st_mode) == 0o600


def test_start_background_gives_up_naming_the_log(data_home: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(served.subprocess, "Popen", lambda *a, **k: type("P", (), {"wait": lambda self: 0})())
    monkeypatch.setattr(served, "hello", lambda workspace, timeout=1.0: None)
    monkeypatch.setattr(served, "START_WAIT", 0.1)
    with pytest.raises(TimeoutError, match=str(log_path(tmp_path))):
        served.start_background(tmp_path)


def _holder(pid_file: Path, seconds: float) -> subprocess.Popen[bytes]:
    """A process that holds the daemon's pid lock for `seconds`, then exits (and is never waited for: a zombie)."""
    code = textwrap.dedent(
        f"""
        import time
        from pathlib import Path
        from core.server.files import PidLock
        lock = PidLock.acquire(Path({str(pid_file)!r}))
        print("locked", flush=True)
        time.sleep({seconds})
        lock.release()
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    assert proc.stdout is not None and proc.stdout.readline() == b"locked\n"
    return proc


def test_stop_does_not_wait_on_a_daemon_that_exited_but_was_not_reaped(
    data_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(served, "hello", lambda workspace, timeout=1.0: {"v": 1})
    monkeypatch.setattr(served.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(served, "STOP_WAIT", 5.0)
    proc = _holder(pid_path(tmp_path), 0.3)  # a zombie from 0.3 s on: `kill(pid, 0)` would still succeed
    try:
        begin = time.monotonic()
        assert served.stop(tmp_path)
        assert time.monotonic() - begin < 3.0 and killed == []
    finally:
        proc.wait()


def test_stop_sends_sigterm_to_a_daemon_that_does_not_stop(
    data_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(served, "hello", lambda workspace, timeout=1.0: {"v": 1})
    monkeypatch.setattr(served.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(served, "STOP_WAIT", 0.2)
    lock = PidLock.acquire(pid_path(tmp_path))  # held the whole time: it never stops
    try:
        assert served.stop(tmp_path)
    finally:
        lock.release()
    assert killed == [(os.getpid(), signal.SIGTERM)]


def test_a_signal_during_shutdown_does_not_cut_the_cleanup_short(
    data_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("AGENT_DAEMON_IDLE", "0.2")  # it stops by itself
    built: list[object] = []
    kills: list[str] = []

    def fake_build(workspace: Path, builtins: object, **kwargs: object):
        rt = build_runtime(
            workspace, BUILTINS, terminals=False, provider=ScriptedProvider([]), catalog=FakeCatalog(["fake"]), connector=FakeConnector()
        )
        real_kill_all = rt.kill_all

        def kill_all() -> None:
            kills.append("kill_all")
            real_kill_all()

        rt.kill_all = kill_all  # ty: ignore[invalid-assignment]
        built.append(rt)
        return rt

    real_close = Daemon.close

    async def close(self: Daemon) -> None:
        signal.raise_signal(signal.SIGTERM)  # `agent stop` gave up waiting, or the user pressed ctrl+c twice
        signal.raise_signal(signal.SIGINT)
        await asyncio.sleep(0)
        await real_close(self)

    monkeypatch.setattr("core.runtime.build_runtime", fake_build)
    monkeypatch.setattr(Daemon, "close", close)
    before = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))

    assert served.serve(ws) == 0

    assert built and kills  # the runtime's processes were stopped
    assert not pid_path(ws).exists() and not socket_path(ws).exists()
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == before


def test_stop_with_no_run_folder_creates_none(data_home: Path, tmp_path: Path) -> None:
    from core.server.paths import _base

    assert not served.stop(tmp_path)
    assert not _base().exists()


def test_the_config_warnings_nobody_heard_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    class Guard:
        def undelivered(self) -> list[str]:
            return ["config changed by bash (x): .agent/settings.json"]

    with caplog.at_level("WARNING", logger="agent.daemon"):
        served.log_undelivered(Guard())
    assert [r.getMessage() for r in caplog.records] == ["warning: config changed by bash (x): .agent/settings.json"]


async def test_ensure_daemon_uses_the_running_daemon_and_starts_one_when_none_runs(serve, tmp_path, monkeypatch) -> None:
    served = await serve()
    started: list[Path] = []

    def fake_start(workspace: Path) -> Path:
        started.append(workspace)
        return workspace / "new.sock"

    monkeypatch.setattr(served_module, "start_background", fake_start)
    assert await asyncio.to_thread(served_module.ensure_daemon, served.workspace) == served.socket
    assert started == []
    other = tmp_path / "other"
    other.mkdir()
    assert await asyncio.to_thread(served_module.ensure_daemon, other) == other.resolve() / "new.sock"
    assert started == [other.resolve()]
