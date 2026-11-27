"""The clipboard client: the subprocess is faked, the real clipboard is never touched."""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from core.client import clipboard
from core.client.clipboard import ClipboardError, ClipboardUnsupported, PasteFolder, read_clipboard_image

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 64


class FakeProc:
    def __init__(self, code: int = 0, err: bytes = b"", write: bytes | None = None, hang: bool = False) -> None:
        self.returncode: int | None = code
        self.pid = 999_999_999  # never a real group: the kill is faked too
        self._err, self._write, self._hang = err, write, hang
        self.dest: str | None = None

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            await asyncio.sleep(60)
        if self._write is not None and self.dest is not None:
            Path(self.dest).write_bytes(self._write)
        return b"", self._err

    async def wait(self) -> int:
        return 0


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch):
    state: dict = {"calls": [], "killed": 0, "proc": FakeProc(write=PNG)}
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: None)

    async def create(*argv, **kwargs):
        state["calls"].append((argv, kwargs))
        proc = state["proc"]
        proc.dest = argv[-1]
        return proc

    async def kill(proc) -> None:
        state["killed"] += 1

    monkeypatch.setattr(clipboard.asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(clipboard, "kill_group", kill)
    return state


async def test_other_platforms_are_unsupported(fake, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(ClipboardUnsupported):
        await read_clipboard_image(tmp_path)
    assert fake["calls"] == []


async def test_osascript_gets_the_path_as_an_argument_not_in_the_script(fake, tmp_path: Path) -> None:
    dest = await read_clipboard_image(tmp_path)

    assert dest is not None and dest.read_bytes() == PNG
    argv, kwargs = fake["calls"][0]
    assert argv[0] == "osascript" and argv[-1] == str(dest)
    assert str(tmp_path) not in " ".join(argv[:-1])
    assert any("on run argv" in a for a in argv)
    assert kwargs["start_new_session"] is True


async def test_pngpaste_is_preferred_when_installed(fake, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: "/opt/homebrew/bin/pngpaste")
    dest = await read_clipboard_image(tmp_path)

    assert dest is not None
    assert fake["calls"][0][0][0] == "/opt/homebrew/bin/pngpaste"


@pytest.mark.parametrize(
    "err", [b"execution error: Can't make some data into the expected type. (-1700)", b"No PNG data on the pasteboard"]
)
async def test_no_image_on_the_clipboard_is_none(fake, tmp_path: Path, err: bytes) -> None:
    fake["proc"] = FakeProc(code=1, err=err)
    assert await read_clipboard_image(tmp_path) is None


async def test_another_failure_is_an_error_with_the_reason(fake, tmp_path: Path) -> None:
    fake["proc"] = FakeProc(code=1, err=b"boom happened")
    with pytest.raises(ClipboardError, match="boom happened"):
        await read_clipboard_image(tmp_path)


async def test_a_file_that_is_no_image_is_none(fake, tmp_path: Path) -> None:
    fake["proc"] = FakeProc(write=b"just text")
    assert await read_clipboard_image(tmp_path) is None


async def test_a_slow_command_is_killed_and_reported(fake, tmp_path: Path) -> None:
    fake["proc"] = FakeProc(hang=True)
    with pytest.raises(ClipboardError, match="timed out"):
        await read_clipboard_image(tmp_path, timeout=0.05)
    assert fake["killed"] == 1


def test_the_paste_folder_is_private_outside_the_workspace_and_removed(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    folder = PasteFolder(workspace)
    path = folder.path()

    assert path.is_dir() and stat.S_IMODE(path.stat().st_mode) == 0o700
    assert workspace.resolve() not in path.resolve().parents
    assert folder.path() == path
    folder.remove()
    assert not path.exists()
    folder.remove()  # idempotent


def test_a_temp_folder_inside_the_workspace_is_not_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "ws"
    (workspace / "tmp").mkdir(parents=True)
    monkeypatch.setattr(clipboard.tempfile, "gettempdir", lambda: str(workspace / "tmp"))
    monkeypatch.setattr(clipboard.Path, "home", lambda: tmp_path / "home")
    folder = PasteFolder(workspace)
    path = folder.path()

    assert workspace.resolve() not in path.resolve().parents
    assert stat.S_IMODE(path.stat().st_mode) == 0o700
    folder.remove()
    assert not path.exists()


@pytest.mark.skipif(sys.platform != "darwin" or shutil.which("osacompile") is None, reason="needs macOS osacompile")
def test_the_real_script_compiles(tmp_path: Path) -> None:
    source = tmp_path / "paste.applescript"
    source.write_text("\n".join(clipboard.SCRIPT) + "\n")
    done = subprocess.run(
        ["osacompile", "-o", str(tmp_path / "paste.scpt"), str(source)], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


# real processes: a fake `pngpaste` on PATH, nothing is mocked


def _fake_pngpaste(directory: Path, body: str) -> None:
    directory.mkdir(exist_ok=True)
    tool = directory / "pngpaste"
    tool.write_text("#!/bin/sh\n" + body + "\n")
    tool.chmod(0o755)


@pytest.fixture
def real(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("PATH", f"{bin_dir}:/bin:/usr/bin")
    return bin_dir


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_a_hanging_pngpaste_times_out_and_its_whole_group_is_killed(real: Path, tmp_path: Path) -> None:
    pids = tmp_path / "pids"
    _fake_pngpaste(real, f'[ "$1" = --warm ] && exit 0\necho $$ > "{pids}"\nsleep 30 &\necho $! >> "{pids}"\nsleep 30')
    # macOS checks a newly written executable on its first exec (about 0.3 s, more under load), which could use up the
    # 0.5 s timeout before the script wrote its pids; one run first makes the timed run start at once.
    subprocess.run([real / "pngpaste", "--warm"], check=True)
    with pytest.raises(ClipboardError, match="timed out"):
        await read_clipboard_image(tmp_path, timeout=0.5)
    shell, child = (int(p) for p in pids.read_text().split())
    for _ in range(50):
        if not _alive(shell) and not _alive(child):
            break
        await asyncio.sleep(0.1)
    assert not _alive(shell) and not _alive(child)


async def test_a_real_pngpaste_without_an_image_is_none(real: Path, tmp_path: Path) -> None:
    _fake_pngpaste(real, "echo 'No PNG data on the pasteboard' >&2\nexit 1")
    assert await read_clipboard_image(tmp_path) is None


async def test_a_real_pngpaste_that_writes_a_png_returns_the_path(real: Path, tmp_path: Path) -> None:
    _fake_pngpaste(real, 'printf "\\211PNG\\r\\n\\032\\n0000000000" > "$1"')
    dest = await read_clipboard_image(tmp_path)
    assert dest is not None and dest.read_bytes().startswith(b"\x89PNG")


def test_a_removed_paste_folder_refuses_new_paths(tmp_path: Path) -> None:
    folder = PasteFolder(tmp_path)
    folder.path()
    folder.remove()
    with pytest.raises(OSError):
        folder.path()


def test_stale_paste_folders_of_dead_pids_are_pruned(tmp_path: Path) -> None:
    old = time.time() - 3 * 86400
    dead = 2**22 + 12345 if not clipboard._alive(2**22 + 12345) else 0
    made = {
        "dead": tmp_path / f"agent-paste-{dead}-abc123",
        "live": tmp_path / f"agent-paste-{os.getpid()}-abc123",
        "fresh": tmp_path / f"agent-paste-{dead}-fresh1",
        "other": tmp_path / f"agent-paste-{dead}x-abc123",
        "cache": tmp_path / f"paste-{dead}-xyz789",
    }
    for folder in made.values():
        folder.mkdir()
    for name in ("dead", "live", "other", "cache"):
        os.utime(made[name], (old, old))
    (tmp_path / "target").mkdir()
    os.utime(tmp_path / "target", (old, old))
    link = tmp_path / f"agent-paste-{dead}-link12"
    link.symlink_to(tmp_path / "target")

    clipboard.prune_stale(tmp_path)

    assert not made["dead"].exists() and not made["cache"].exists()
    assert made["live"].exists() and made["fresh"].exists() and made["other"].exists()
    assert link.is_symlink() and (tmp_path / "target").exists()
