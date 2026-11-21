"""The generate_image tool: make a PNG in the workspace with mflux, after approval."""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, ClassVar

from src.engine.guardrails.setup import guard_of
from src.engine.imagegen import (
    DEFAULT_SIDE,
    DEFAULT_STEPS,
    MAX_PROMPT,
    MAX_SIDE,
    MAX_STEPS,
    MIN_SIDE,
    SIDE_STEP,
    ImageGenError,
    ImageProgress,
    ImageRequest,
    MfluxGenerator,
    format_progress,
    model_cached,
    png_size,
)
from src.engine.provider.images import size_text, sniff
from src.engine.text import escape_display

from .base import Prepared, Preview, Tool, ToolContext, ToolResult, check_args
from .fs import _refuse_outside
from .paths import InvalidPath, inside_workspace, invalid_path_result, resolve_path
from .protected import refuse_protected

SEED_LIMIT = 2**32
PROGRESS_TICK = 0.25  # seconds between progress lines
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_IEND = b"IEND\xaeB`\x82"
_FORMAT_NAMES = {"jpeg": "a JPEG", "gif": "a GIF", "webp": "a WebP image"}


def _invalid(message: str) -> ToolResult:
    return ToolResult(message, is_error=True, summary="invalid")


def _side_error(name: str, value: int) -> ToolResult | None:
    if MIN_SIDE <= value <= MAX_SIDE and value % SIDE_STEP == 0:
        return None
    return _invalid(f"{name} must be {MIN_SIDE}-{MAX_SIDE} and a multiple of {SIDE_STEP}")


async def _finish_placement(work: Any) -> ToolResult | None:
    """Wait for the placement thread even when cancelled, then re-raise the cancel. A thread cannot be stopped, so
    once placement started it finishes (file and journal entry together) and /undo covers the file."""
    task = asyncio.ensure_future(work)
    cancelled: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:
            if task.done() and not task.cancelled():
                result = task.result()
                cancelled = cancelled or exc
                break
            cancelled = exc
    if cancelled is not None:
        raise cancelled
    return result


class GenerateImageTool(Tool):
    name: ClassVar[str] = "generate_image"
    description: ClassVar[str] = (
        "Generate a PNG image from a text prompt with a local model (mflux) and save it at a path inside the workspace. "
        "It takes about a minute or more, replaces a file at the path, and every call needs user approval."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": f"What the image shows, 1-{MAX_PROMPT} characters."},
            "path": {"type": "string", "description": "Where to save it: a .png path inside the workspace."},
            "width": {"type": "integer", "description": f"Pixels, {MIN_SIDE}-{MAX_SIDE}, a multiple of {SIDE_STEP}. Default {DEFAULT_SIDE}."},
            "height": {"type": "integer", "description": f"Pixels, {MIN_SIDE}-{MAX_SIDE}, a multiple of {SIDE_STEP}. Default {DEFAULT_SIDE}."},
            "seed": {"type": "integer", "description": "Integer 0-4294967295 to make the same image again. Default random."},
            "steps": {"type": "integer", "description": f"Denoising steps, 1-{MAX_STEPS}. Default {DEFAULT_STEPS}."},
        },
        "required": ["prompt", "path"],
    }
    mutating: ClassVar[bool] = True  # hidden and refused in plan mode
    needs_approval: ClassVar[bool] = True  # every call asks; target stays None so accept-edits never skips it
    uncancellable: ClassVar[bool | None] = False  # Esc kills the process group
    grant_arg: ClassVar[str | None] = "prompt"

    def __init__(self, generator: MfluxGenerator) -> None:
        self.generator = generator

    def label(self, args: dict[str, Any]) -> str:
        """The file, not the prompt: a prompt can be 2,000 characters and would fill the block's one-line title."""
        path = args.get("path")
        return f"{self.name} {path}" if isinstance(path, str) and path else self.name

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        if (problem := check_args(self.parameters, args)) is not None:
            return _invalid(problem)
        raw: str = args["path"]
        prompt = args["prompt"].strip()
        width = args.get("width", DEFAULT_SIDE)
        height = args.get("height", DEFAULT_SIDE)
        steps = args.get("steps", DEFAULT_STEPS)
        seed = args.get("seed")
        if not 1 <= len(prompt) <= MAX_PROMPT:
            return _invalid(f"prompt must be 1-{MAX_PROMPT} characters")
        for name, value in (("width", width), ("height", height)):
            if (error := _side_error(name, value)) is not None:
                return error
        if not 1 <= steps <= MAX_STEPS:
            return _invalid(f"steps must be 1-{MAX_STEPS}")
        if seed is not None and not 0 <= seed < SEED_LIMIT:
            return _invalid(f"seed must be 0-{SEED_LIMIT - 1}")
        try:
            path = resolve_path(ctx, raw)
        except InvalidPath as exc:
            return invalid_path_result(exc)
        if not inside_workspace(path, ctx):
            return _refuse_outside()
        if guard_of(ctx).paths.is_protected(path):
            return refuse_protected()
        if path.is_dir():
            return ToolResult(f"{raw} is a directory", is_error=True, summary="directory")
        if path.exists() and not path.is_file():
            return ToolResult(f"{raw} is not a regular file", is_error=True, summary="not a regular file")
        if path.suffix.lower() != ".png":
            return ToolResult(f"{raw} must end in .png", is_error=True, summary="invalid")
        seed = secrets.randbelow(SEED_LIMIT) if seed is None else seed
        request = ImageRequest(prompt, width, height, steps, seed)
        model = self.generator.model
        lines = [
            f"prompt: {escape_display(prompt)}",
            f"size: {width}\u00d7{height}, {steps} steps, seed {seed}",
            f"model: {escape_display(model)}",
        ]
        if path.exists():
            lines.append("[overwrites the existing file]")
        if not model_cached(model):
            lines.append(f"[model {escape_display(model)} is not downloaded; this call may download several GB]")
        preview = Preview("content", f"generate image {escape_display(raw)}", "\n".join(lines))

        async def execute() -> ToolResult:
            return await self._run(request, path, raw, ctx)

        return Prepared(preview, execute)

    async def _generate(self, request: ImageRequest, out_dir: Path, ctx: ToolContext) -> Path:
        """Run mflux; with a progress sink, a ticker sends the display line when it changes (at most 4 a second)."""
        from src.engine.events import ToolProgress  # not at module level: events imports the tools package

        sink = ctx.progress
        if sink is None:
            return await self.generator.generate(request, out_dir)
        start = time.monotonic()
        latest = ImageProgress("starting")
        last: str | None = None

        def send() -> None:
            nonlocal last
            text = format_progress(latest, time.monotonic() - start)
            if text != last:
                last = text
                with contextlib.suppress(Exception):
                    sink(ToolProgress(text))

        def update(value: ImageProgress) -> None:
            nonlocal latest
            latest = value

        async def ticker() -> None:
            while True:
                send()
                await asyncio.sleep(PROGRESS_TICK)

        task = asyncio.create_task(ticker())
        try:
            return await self.generator.generate(request, out_dir, on_progress=update)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _run(self, request: ImageRequest, path: Path, raw: str, ctx: ToolContext) -> ToolResult:
        out_dir = Path(tempfile.mkdtemp(prefix="agent-image-"))
        try:
            try:
                generated = await self._generate(request, out_dir, ctx)
            except ImageGenError as exc:
                message = str(exc)
                return ToolResult(message, is_error=True, summary="timed out" if message.startswith("timed out") else "failed")
            try:
                if (await asyncio.to_thread(generated.stat)).st_size > MAX_OUTPUT_BYTES:
                    return ToolResult("mflux wrote an image over 64 MB", is_error=True, summary="failed")
                data = await asyncio.to_thread(generated.read_bytes)
            except OSError as exc:
                return ToolResult(f"Cannot read the image: {exc.strerror or exc}", is_error=True, summary="failed")
            size = png_size(data)
            if size is None:
                kind = _FORMAT_NAMES.get(sniff(data[:16]) or "", "an unknown file")
                return ToolResult(f"mflux wrote {kind} instead of a PNG", is_error=True, summary="failed")
            if size != (request.width, request.height):
                expected = f"{request.width}\u00d7{request.height}"
                return ToolResult(f"mflux wrote {size[0]}\u00d7{size[1]}, expected {expected}", is_error=True, summary="failed")
            if not data.endswith(_IEND):
                return ToolResult("mflux wrote an incomplete PNG", is_error=True, summary="failed")
            try:
                refused = await _finish_placement(asyncio.to_thread(self._place, path, data, ctx))
            except OSError as exc:
                return ToolResult(f"Cannot write {raw}: {exc.strerror or exc}", is_error=True, summary="write failed")
            if refused is not None:
                return refused
            described = f"PNG, {size[0]}\u00d7{size[1]}, {size_text(len(data))}"
            return ToolResult(f"generated {raw} ({described}, seed {request.seed})", summary=described)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    @staticmethod
    def _place(path: Path, data: bytes, ctx: ToolContext) -> ToolResult | None:
        """Write `data` beside `path` and rename it into place. A folder may have been swapped for a symlink during
        the minutes mflux ran, so the target is checked again right before the write."""

        def allowed() -> bool:
            final = path.parent.resolve() / path.name
            return inside_workspace(final, ctx) and not guard_of(ctx).paths.is_protected(final)

        if not allowed():
            return _refuse_outside()
        if ctx.journal is not None:
            ctx.journal.before(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not allowed():
                return _refuse_outside()
            try:
                mode = path.stat().st_mode & 0o7777
            except OSError:
                mode = None
            temp = path.with_name(f".{path.name}.agent-{secrets.token_hex(4)}.tmp")
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW
            try:
                fd = os.open(temp, flags, 0o666)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                    if mode is not None:
                        os.fchmod(handle.fileno(), mode)
                os.replace(temp, path)
            except BaseException:
                with contextlib.suppress(OSError):
                    temp.unlink()
                raise
        finally:
            if ctx.journal is not None:
                ctx.journal.after(path)
        return None

