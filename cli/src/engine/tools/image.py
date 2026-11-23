"""The generate_image tool: make a PNG in the workspace with mflux, after approval."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import secrets
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from src.engine.guardrails.setup import guard_of
from src.engine.image_inputs import MAX_REFERENCES, InputImage, default_size, resolve_input, stage
from src.engine.image_pull import REFUSED, ImagePuller
from src.engine.imagegen import (
    DEFAULT_EDIT_STEPS,
    DEFAULT_SIDE,
    DEFAULT_STEPS,
    MAX_PROMPT,
    MAX_SIDE,
    MAX_STEPS,
    MIN_SIDE,
    SIDE_STEP,
    ImageGenError,
    ImageRequest,
    MfluxGenerator,
    model_cached,
    png_size,
    run_with_ticker,
)
from src.engine.provider.images import size_text, sniff
from src.engine.text import escape_display

from .base import Prepared, Preview, Tool, ToolContext, ToolResult, check_args
from .fs import _refuse_outside
from .paths import InvalidPath, inside_workspace, invalid_path_result, resolve_path
from .protected import refuse_protected

SEED_LIMIT = 2**32
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_IEND = b"IEND\xaeB`\x82"
_FORMAT_NAMES = {"jpeg": "a JPEG", "gif": "a GIF", "webp": "a WebP image"}


def _invalid(message: str) -> ToolResult:
    return ToolResult(message, is_error=True, summary="invalid")


def _side_error(name: str, value: int) -> ToolResult | None:
    if MIN_SIDE <= value <= MAX_SIDE and value % SIDE_STEP == 0:
        return None
    return _invalid(f"{name} must be {MIN_SIDE}-{MAX_SIDE} and a multiple of {SIDE_STEP}")


async def _finish_placement(work: Any) -> Any:
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


def _check_output_path(raw: str, ctx: ToolContext) -> Path | ToolResult:
    """The resolved output path, or the refusal: inside the workspace, not protected, not a folder, a `.png`."""
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
    return path


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

    def __init__(self, generator: MfluxGenerator, pull: ImagePuller | None = None) -> None:
        self.generator = generator
        self.pull = pull

    def label(self, args: dict[str, Any]) -> str:
        """The file, not the prompt: a prompt can be 2,000 characters and would fill the block's one-line title."""
        path = args.get("path")
        return f"{self.name} {path}" if isinstance(path, str) and path else self.name

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        if self.pull is not None and self.pull.busy:
            return ToolResult(REFUSED, is_error=True, summary="refused")
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
        if isinstance(checked := _check_output_path(raw, ctx), ToolResult):
            return checked
        path = checked
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
            return await _run_mflux(self.generator, request, path, raw, ctx, "generated")

        return Prepared(preview, execute)


def _describe_input(item: InputImage) -> str:
    return f"{item.label} ({item.fmt.upper()}, {item.width}\u00d7{item.height}, {size_text(len(item.data))})"


class EditImageTool(Tool):
    name: ClassVar[str] = "edit_image"
    description: ClassVar[str] = (
        "Edit an image from a text instruction with a local model (mflux, FLUX.2 klein) and save the result as a PNG "
        "at a path inside the workspace. `image` is the source (a workspace path, or clipboard:N for an image pasted "
        f"into the user's message; clipboard:N refers only to the images pasted into the latest user message); `references` are up to {MAX_REFERENCES} more images the edit may use. "
        "It takes minutes, replaces a file at the path (the source included), and every call needs user approval."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": f"What to change, 1-{MAX_PROMPT} characters."},
            "image": {"type": "string", "description": "The source image: a workspace path or clipboard:N."},
            "path": {"type": "string", "description": "Where to save the result: a .png path inside the workspace."},
            "references": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_REFERENCES,
                "description": f"Up to {MAX_REFERENCES} reference images: workspace paths or clipboard:N.",
            },
            "width": {"type": "integer", "description": f"Pixels, {MIN_SIDE}-{MAX_SIDE}, a multiple of {SIDE_STEP}. Default the source's, scaled to a longer side of at most 1024."},
            "height": {"type": "integer", "description": f"Pixels, {MIN_SIDE}-{MAX_SIDE}, a multiple of {SIDE_STEP}. Default the source's, scaled to a longer side of at most 1024."},
            "seed": {"type": "integer", "description": "Integer 0-4294967295 to make the same edit again. Default random."},
            "steps": {"type": "integer", "description": f"Denoising steps, 1-{MAX_STEPS}. Default {DEFAULT_EDIT_STEPS}."},
        },
        "required": ["prompt", "image", "path"],
    }
    mutating: ClassVar[bool] = True  # hidden and refused in plan mode
    needs_approval: ClassVar[bool] = True  # every call asks; target stays None so accept-edits never skips it
    uncancellable: ClassVar[bool | None] = False  # Esc kills the process group
    grant_arg: ClassVar[str | None] = "prompt"

    def __init__(self, generator: MfluxGenerator, pull: ImagePuller | None = None) -> None:
        self.generator = generator
        self.pull = pull

    def label(self, args: dict[str, Any]) -> str:
        """The output file, not the prompt."""
        path = args.get("path")
        return f"{self.name} {path}" if isinstance(path, str) and path else self.name

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        if self.pull is not None and self.pull.busy:
            return ToolResult(REFUSED, is_error=True, summary="refused")
        if (problem := check_args(self.parameters, args)) is not None:
            return _invalid(problem)
        raw: str = args["path"]
        prompt = args["prompt"].strip()
        references = args.get("references", [])
        steps = args.get("steps", DEFAULT_EDIT_STEPS)
        seed = args.get("seed")
        if not 1 <= len(prompt) <= MAX_PROMPT:
            return _invalid(f"prompt must be 1-{MAX_PROMPT} characters")
        if not isinstance(references, list) or not all(isinstance(r, str) for r in references):
            return _invalid("references must be a list of strings")
        if len(references) > MAX_REFERENCES:
            return _invalid(f"references must have at most {MAX_REFERENCES} items")
        for name in ("width", "height"):
            if name in args and (error := _side_error(name, args[name])) is not None:
                return error
        if not 1 <= steps <= MAX_STEPS:
            return _invalid(f"steps must be 1-{MAX_STEPS}")
        if seed is not None and not 0 <= seed < SEED_LIMIT:
            return _invalid(f"seed must be 0-{SEED_LIMIT - 1}")
        if isinstance(checked := _check_output_path(raw, ctx), ToolResult):
            return checked
        path = checked
        inputs: list[InputImage] = []
        for ref in (args["image"], *references):
            found = await asyncio.to_thread(resolve_input, ref, ctx)
            if isinstance(found, ToolResult):
                return found
            if any(found.identity == other.identity for other in inputs):
                return _invalid(f"{escape_display(ref)} is given twice (the same image as {found.label})")
            inputs.append(found)
        source = inputs[0]
        default_width, default_height = default_size(source.width, source.height)
        width = args.get("width", default_width)
        height = args.get("height", default_height)
        seed = secrets.randbelow(SEED_LIMIT) if seed is None else seed
        request = ImageRequest(prompt, width, height, steps, seed)
        model = self.generator.model
        lines = [
            f"prompt: {escape_display(prompt)}",
            f"source: {_describe_input(source)}",
            *(f"reference: {_describe_input(item)}" for item in inputs[1:]),
            f"size: {width}\u00d7{height}, {steps} steps, seed {seed}",
            f"model: {escape_display(model)}",
        ]
        if path.exists():
            lines.append("[overwrites the existing file]")
        if not model_cached(model):
            lines.append(f"[model {escape_display(model)} is not downloaded; this call may download several GB]")
        preview = Preview("content", f"edit image {escape_display(raw)}", "\n".join(lines))

        async def execute() -> ToolResult:
            return await _run_mflux(self.generator, request, path, raw, ctx, "edited", lambda folder: stage(inputs, folder))

        return Prepared(preview, execute)


async def _generate(generator: MfluxGenerator, request: ImageRequest, out_dir: Path, ctx: ToolContext) -> Path:
    """Run mflux; with a progress sink, a ticker sends the display line when it changes (at most 4 a second)."""
    from src.engine.events import ToolProgress  # not at module level: events imports the tools package

    sink = ctx.progress
    if sink is None:
        return await generator.generate(request, out_dir)
    return await run_with_ticker(generator, request, out_dir, lambda text: sink(ToolProgress(text)))


async def _run_mflux(
    generator: MfluxGenerator,
    request: ImageRequest,
    path: Path,
    raw: str,
    ctx: ToolContext,
    verb: str,
    stager: Callable[[Path], tuple[Path, ...]] | None = None,
) -> ToolResult:
    """Run mflux in a private folder, check the PNG it wrote and place it at `path`. `stager` (edit_image) copies the
    inputs into that folder first; the folder is always removed."""
    out_dir = Path(tempfile.mkdtemp(prefix="agent-image-"))
    try:
        if stager is not None:
            try:
                request = dataclasses.replace(request, inputs=await _finish_placement(asyncio.to_thread(stager, out_dir)))
            except OSError as exc:
                return ToolResult(f"Cannot stage the input images: {exc.strerror or exc}", is_error=True, summary="failed")
        try:
            generated = await _generate(generator, request, out_dir, ctx)
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
            refused = await _finish_placement(asyncio.to_thread(_place, path, data, ctx))
        except OSError as exc:
            return ToolResult(f"Cannot write {raw}: {exc.strerror or exc}", is_error=True, summary="write failed")
        if refused is not None:
            return refused
        described = f"PNG, {size[0]}\u00d7{size[1]}, {size_text(len(data))}"
        return ToolResult(f"{verb} {raw} ({described}, seed {request.seed})", summary=described)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)



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

