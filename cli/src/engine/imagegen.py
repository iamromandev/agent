"""Image generation with mflux: find its command, check its output and run it as a sandboxed subprocess.

mflux is never imported: it is an optional extra, and its console script runs in its own process group.
UI-free: no `textual`, no `httpx`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import platform as platform_module
import re
import struct
import sys
from asyncio.subprocess import DEVNULL, PIPE
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from src.engine.env import child_environment
from src.engine.process import kill_group
from src.engine.text import escape

if TYPE_CHECKING:
    from src.engine.guardrails.setup import Guardrails

COMMAND = "mflux-generate-z-image-turbo"
DEFAULT_MODEL = "filipstrand/Z-Image-Turbo-mflux-4bit"
EDIT_COMMAND = "mflux-generate-flux2-edit"
DEFAULT_EDIT_MODEL = "black-forest-labs/FLUX.2-klein-4B"
EDIT_EXTRA = ("--quantize", "4")
DEFAULT_EDIT_STEPS = 4
MIN_SIDE, MAX_SIDE, SIDE_STEP = 256, 2048, 16
MAX_STEPS, DEFAULT_STEPS, DEFAULT_SIDE = 50, 9, 1024
MAX_PROMPT = 2000
STDERR_LINES, STDERR_CHARS = 20, 2000
OUTPUT_NAME = "image.png"
_KEEP = 64 * 1024  # of stderr: the rest is read and dropped so the child never blocks
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PROGRESS = re.compile(r"it/s|s/it|%\|")


@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
    inputs: tuple[Path, ...] = ()  # edit commands: the source first, then the references


class ImageGenError(Exception):
    """The one-line reason a generation failed, as the model sees it."""


def find_mflux(
    system: str = sys.platform,
    machine: str | None = None,
    python: Path | None = None,
    command: str = COMMAND,
) -> Path | None:
    """The named mflux command next to the interpreter, on macOS arm64 only."""
    machine = platform_module.machine() if machine is None else machine
    if system != "darwin" or machine != "arm64":
        return None
    found = (Path(sys.executable) if python is None else python).parent / command
    return found if found.is_file() and os.access(found, os.X_OK) else None


def _hub_cache(env: Mapping[str, str]) -> Path:
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"])
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_cached(model: str, env: Mapping[str, str] = os.environ) -> bool:
    """True when the model is in the local Hugging Face cache (or is a folder that exists)."""
    if Path(model).is_dir():
        return True
    snapshots = _hub_cache(env) / f"models--{model.replace('/', '--')}" / "snapshots"
    try:
        return any(any(entry.iterdir()) for entry in snapshots.iterdir() if entry.is_dir())
    except OSError:
        return False


def png_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from the IHDR of a PNG, or None when `data` is no PNG."""
    if len(data) < 33 or not data.startswith(_PNG_SIGNATURE) or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return (width, height) if width and height else None


def _tail(raw: bytes) -> str:
    """The last lines of stderr that carry information: tqdm progress is dropped, `\\r` splits lines."""
    text = raw.decode("utf-8", errors="replace")
    lines = [line.strip() for line in re.split(r"[\r\n]", text)]
    lines = [line for line in lines if line and not _PROGRESS.search(line)]
    return escape("\n".join(lines[-STDERR_LINES:]))[-STDERR_CHARS:]


Phase = Literal["starting", "downloading", "loading", "steps", "saving"]
_STEP_LINE = re.compile(r"^\s*\d{1,3}%\|[^|]*\|\s*(\d{1,4})/(\d{1,4}) \[([0-9:]+)<([0-9:]+|\?)[,\]]")
_DOWNLOAD_LINE = re.compile(r"^(.+?):\s*(\d{1,3})%\|")
_TAIL_CAP = 4096
DISPLAY_LIMIT = 80


@dataclass(frozen=True)
class ImageProgress:
    phase: Phase
    step: int = 0  # steps phase: completed steps
    total: int = 0  # steps phase: mflux's total (the requested steps)
    left: int | None = None  # seconds, mflux's estimate; None when tqdm printed "?"
    percent: int | None = None  # downloading phase, 0-100, when readable


def _seconds(text: str) -> int | None:
    parts = text.split(":")
    if not 2 <= len(parts) <= 3 or not all(part.isdigit() for part in parts):
        return None
    total = 0
    for part in parts:
        total = total * 60 + int(part)
    return total


def clock(seconds: float) -> str:
    """`m:ss`, or `h:mm:ss` from one hour."""
    whole = max(0, int(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class ProgressParser:
    """Incremental: `feed` returns the progress values for the complete lines in the bytes so far."""

    def __init__(self, total: int, downloading: bool) -> None:
        self.total = total
        self.downloading = downloading
        self._tail = b""
        self._last = 0
        self._stepping = False
        self._in_download = False
        self._percent = -1

    def start(self) -> ImageProgress:
        return ImageProgress("downloading" if self.downloading else "loading")

    def feed(self, chunk: bytes) -> list[ImageProgress]:
        data = self._tail + chunk
        pieces = re.split(rb"([\r\n])", data)
        tail = pieces.pop()
        self._tail = tail[-_TAIL_CAP:]
        values: list[ImageProgress] = []
        for index in range(0, len(pieces), 2):
            if pieces[index]:
                values.extend(self._line(pieces[index].decode("utf-8", errors="replace"), pieces[index + 1] == b"\n"))
        # the newest bar has no terminator until the next one starts: read it now, without consuming it
        if tail:
            values.extend(self._line(self._tail.decode("utf-8", errors="replace"), None))
        return values

    def _line(self, line: str, newline: bool | None) -> list[ImageProgress]:
        """`newline` is None for the unfinished tail, else whether a `\\n` ended the line."""
        if match := _STEP_LINE.match(line):
            return self._step(int(match[1]), int(match[2]), match[4])
        if self.downloading and not self._stepping:
            if match := _DOWNLOAD_LINE.match(line):
                percent = int(match[2])
                if percent > 100:
                    return []
                self._in_download = True
                if newline:
                    self._in_download = False
                    seen, self._percent = self._percent == percent, percent
                    shown = [] if seen else [ImageProgress("downloading", percent=percent)]
                    return [*shown, ImageProgress("loading")]
                if percent == self._percent and newline is None:
                    return []
                self._percent = percent
                return [ImageProgress("downloading", percent=percent)]
            if self._in_download and newline is not None:
                self._in_download = False
                return [ImageProgress("loading")]
        return []

    def _step(self, step: int, total: int, left: str) -> list[ImageProgress]:
        if total != self.total or step > total or step < self._last:
            return []
        if step == 0:
            return []
        if step == self._last:
            return []
        self._last = step
        self._stepping = True
        values = [ImageProgress("steps", step, total, None if left == "?" else _seconds(left))]
        if step == total:
            values.append(ImageProgress("saving"))
        return values


def format_progress(p: ImageProgress, elapsed: float) -> str:
    """The one display line, at most 80 characters, built from numbers and fixed words only."""
    time_text = clock(elapsed)
    match p.phase:
        case "starting":
            text = f"starting… {time_text}"
        case "downloading":
            percent = f"{p.percent}% · " if p.percent is not None else ""
            text = f"downloading model… {percent}{time_text}"
        case "loading":
            text = f"loading model… {time_text}"
        case "steps":
            left = f" · ~{clock(p.left)} left" if p.left is not None else ""
            text = f"step {p.step}/{p.total} · {time_text} elapsed{left}"
        case "saving":
            text = f"saving… {time_text}"
    return text[:DISPLAY_LIMIT]


class MfluxGenerator:
    def __init__(
        self, command: Path, model: str, timeout: float, guard: Guardrails, extra: tuple[str, ...] = ()
    ) -> None:
        self.command = command
        self.extra = extra  # fixed arguments of this command, before --output
        # a folder is made absolute once: mflux runs with another cwd, and the cache check must see the same path
        self.model = str(Path(model).resolve()) if Path(model).is_dir() else model
        self.timeout = timeout
        self.guard = guard

    def argv(self, request: ImageRequest, out_dir: Path) -> list[str]:
        """Every value its own item: the prompt never goes through a shell."""
        return self.guard.sandbox.wrap(
            [
                str(self.command),
                "--model", self.model,
                f"--prompt={request.prompt}",
                *(("--image-paths", *map(str, request.inputs)) if request.inputs else ()),
                "--width", str(request.width),
                "--height", str(request.height),
                "--steps", str(request.steps),
                "--seed", str(request.seed),
                *self.extra,
                "--output", str(out_dir / OUTPUT_NAME),
                "--no-exif",
            ]
        )  # fmt: skip

    async def generate(
        self,
        request: ImageRequest,
        out_dir: Path,
        on_progress: Callable[[ImageProgress], None] | None = None,
    ) -> Path:
        """The PNG mflux wrote inside `out_dir`. Kills the whole process group on timeout and on cancel."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.argv(request, out_dir),
                cwd=out_dir,
                stdin=DEVNULL,
                stdout=DEVNULL,
                stderr=PIPE,
                start_new_session=True,
                env=self.guard.env.filter(child_environment()),
            )
        except OSError as exc:
            raise ImageGenError(f"Could not start mflux: {escape(str(exc))}") from exc
        kept = bytearray()
        parser = ProgressParser(request.steps, downloading=not model_cached(self.model))

        def report(value: ImageProgress) -> None:
            if on_progress is not None:
                with contextlib.suppress(Exception):
                    on_progress(value)

        report(parser.start())

        async def drain() -> None:
            assert proc.stderr is not None
            while chunk := await proc.stderr.read(65_536):
                kept.extend(chunk)
                if on_progress is not None:
                    with contextlib.suppress(Exception):
                        for value in parser.feed(chunk):
                            report(value)
                if len(kept) > _KEEP:
                    del kept[: len(kept) - _KEEP]
            await proc.wait()

        try:
            await asyncio.wait_for(drain(), self.timeout)
        except TimeoutError:
            await kill_group(proc)
            raise ImageGenError(f"timed out after {self.timeout:g} s") from None
        except BaseException:
            await kill_group(proc)
            raise
        if proc.returncode != 0:
            raise ImageGenError(f"mflux failed (exit {proc.returncode}): {_tail(bytes(kept))}")
        output = out_dir / OUTPUT_NAME
        if not output.is_file():
            raise ImageGenError("mflux wrote no image")
        return output
