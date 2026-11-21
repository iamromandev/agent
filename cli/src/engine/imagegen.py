"""Image generation with mflux: find its command, check its output and run it as a sandboxed subprocess.

mflux is never imported: it is an optional extra, and its console script runs in its own process group.
UI-free: no `textual`, no `httpx`.
"""

from __future__ import annotations

import asyncio
import os
import platform as platform_module
import re
import struct
import sys
from asyncio.subprocess import DEVNULL, PIPE
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.engine.env import child_environment
from src.engine.process import kill_group
from src.engine.text import escape

if TYPE_CHECKING:
    from src.engine.guardrails.setup import Guardrails

COMMAND = "mflux-generate-z-image-turbo"
DEFAULT_MODEL = "filipstrand/Z-Image-Turbo-mflux-4bit"
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


class ImageGenError(Exception):
    """The one-line reason a generation failed, as the model sees it."""


def find_mflux(
    system: str = sys.platform,
    machine: str | None = None,
    python: Path | None = None,
) -> Path | None:
    """The mflux command next to the interpreter, on macOS arm64 only."""
    machine = platform_module.machine() if machine is None else machine
    if system != "darwin" or machine != "arm64":
        return None
    command = (Path(sys.executable) if python is None else python).parent / COMMAND
    return command if command.is_file() and os.access(command, os.X_OK) else None


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


class MfluxGenerator:
    def __init__(self, command: Path, model: str, timeout: float, guard: Guardrails) -> None:
        self.command = command
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
                "--width", str(request.width),
                "--height", str(request.height),
                "--steps", str(request.steps),
                "--seed", str(request.seed),
                "--output", str(out_dir / OUTPUT_NAME),
                "--no-exif",
            ]
        )  # fmt: skip

    async def generate(self, request: ImageRequest, out_dir: Path) -> Path:
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

        async def drain() -> None:
            assert proc.stderr is not None
            while chunk := await proc.stderr.read(65_536):
                kept.extend(chunk)
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
