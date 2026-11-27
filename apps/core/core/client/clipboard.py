"""Reading an image from the macOS clipboard into a private folder (the app's own call, not the model's)."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path

from core.engine.checkpoints.setup import STALE_AFTER, _alive
from core.engine.process import kill_group
from core.engine.provider.images import sniff
from core.engine.text import escape

TIMEOUT = 2.0
NO_IMAGE = ("-1700", "no png", "no image")
# The destination arrives as an argument (`on run argv`), never inside the script text.
SCRIPT = (
    "on run argv",
    "set dest to POSIX file (item 1 of argv)",
    "set png to (the clipboard as «class PNGf»)",
    "set handle to open for access dest with write permission",
    "try",
    "set eof handle to 0",
    "write png to handle",
    "on error msg",
    "close access handle",
    "error msg",
    "end try",
    "close access handle",
    "end run",
)
_counter = itertools.count(1)


class ClipboardUnsupported(Exception):
    """Clipboard images are read on macOS only."""


class ClipboardError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_FOLDER_NAME = re.compile(r"(?:agent-paste|paste)-([0-9]+)-[A-Za-z0-9_]+", re.ASCII)


def prune_stale(*roots: Path, now: float | None = None) -> None:
    """Delete `agent-paste-<pid>-*` / `paste-<pid>-*` folders in `roots` (real folders owned by this user, never
    symlinks) whose process is gone and that are over a day old. Never raises."""
    now = time.time() if now is None else now
    for root in roots:
        try:
            entries = list(root.iterdir())
        except (OSError, ValueError):
            continue
        for entry in entries:
            try:
                found = _FOLDER_NAME.fullmatch(entry.name)
                if found is None:
                    continue
                info = entry.lstat()
                if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or now - info.st_mtime <= STALE_AFTER:
                    continue
                if _alive(int(found.group(1))):
                    continue
            except (OSError, ValueError, OverflowError):
                continue
            shutil.rmtree(entry, ignore_errors=True)


class PasteFolder:
    """The 0700 folder pasted images are written to: made on first use, outside the workspace, removed on quit."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve()
        self._path: Path | None = None
        self._closed = False

    def path(self) -> Path:
        if self._closed:
            raise OSError("the paste folder was removed")
        if self._path is None or not self._path.is_dir():
            base = Path(tempfile.gettempdir()).resolve()
            if base == self._workspace or self._workspace in base.parents:
                root = Path.home() / ".cache" / "agent"
                root.mkdir(parents=True, exist_ok=True)
                made = Path(tempfile.mkdtemp(prefix=f"paste-{os.getpid()}-", dir=root))
            else:
                made = Path(tempfile.mkdtemp(prefix=f"agent-paste-{os.getpid()}-"))
            made.chmod(0o700)
            self._path = made
        return self._path

    def remove(self) -> None:
        self._closed = True
        path, self._path = self._path, None
        if path is not None:
            shutil.rmtree(path, ignore_errors=True)


async def read_clipboard_image(dest_dir: Path, timeout: float = TIMEOUT) -> Path | None:
    """Write the clipboard's image to a new PNG file in `dest_dir` and return its path; None when the clipboard has
    no image. Raises `ClipboardUnsupported` off macOS and `ClipboardError` for any other failure or a timeout."""
    if sys.platform != "darwin":
        raise ClipboardUnsupported
    dest = dest_dir / f"{next(_counter)}-{os.getpid()}.png"
    tool = shutil.which("pngpaste")
    argv = (tool, str(dest)) if tool else ("osascript", *(a for line in SCRIPT for a in ("-e", line)), str(dest))
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        raise ClipboardError(escape(str(error))[:200]) from error
    try:
        _, err = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        await kill_group(proc)
        raise ClipboardError(f"timed out after {timeout:g} s") from None
    except BaseException:
        await kill_group(proc)
        raise
    text = err.decode("utf-8", "replace").strip()
    if proc.returncode != 0:
        _drop(dest)
        if any(m in text.lower() for m in NO_IMAGE):
            return None
        raise ClipboardError(escape(text)[:200] or f"exit code {proc.returncode}")
    try:
        with dest.open("rb") as f:
            head = f.read(12)
    except OSError:
        return None
    if sniff(head) is None:
        _drop(dest)
        return None
    return dest


def _drop(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()
