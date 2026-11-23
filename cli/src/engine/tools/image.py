"""The generate_image tool: make a PNG in the workspace with mflux, after approval."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import errno
import importlib.util
import json
import os
import re
import secrets
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from src.engine.image_inputs import (
    MAX_REFERENCES,
    InputImage,
    default_size,
    oriented_size,
    resolve_input,
    stage,
    stage_one,
)
from src.engine.image_mask import DEFAULT_FEATHER, MAX_FEATHER, MAX_SHAPES, Shape, check_region, describe, parse_region
from src.engine.image_pull import REFUSED, ImagePuller
from src.engine.imagegen import (
    DEFAULT_EDIT_STEPS,
    DEFAULT_SIDE,
    DEFAULT_STEPS,
    MAX_PROMPT,
    MAX_SIDE,
    MAX_STEPS,
    MAX_UPSCALE_SIDE,
    MIN_SIDE,
    SIDE_STEP,
    UPSCALE_SCALES,
    UPSCALE_STEP,
    ImageGenError,
    ImageRequest,
    MfluxGenerator,
    _tail,
    edit_upscale_scale,
    model_cached,
    png_size,
    run_helper,
    run_with_ticker,
    upscale_size,
)
from src.engine.provider.images import size_text, sniff
from src.engine.text import escape_display

from .base import Prepared, Preview, Tool, ToolContext, ToolResult, check_args
from .outputs import TooBig, read_nofollow
from .outputs import check_output_path as _check_output_path
from .outputs import finish_placement as _finish_placement
from .outputs import place_bytes as _place

SEED_LIMIT = 2**32
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_IEND = b"IEND\xaeB`\x82"
_SPEC = importlib.util.find_spec("src.engine.composite_helper")  # found, never imported: it needs Pillow
COMPOSITE_HELPER: Path | None = Path(_SPEC.origin) if _SPEC is not None and _SPEC.origin else None
OUTPUT_COMPOSITE = "out.png"
MASK_NAME = "mask.png"
MAX_COMPOSITE_PIXELS = 50_000_000  # the helper decodes the source and the mask: its memory grows with the pixels
_SIZES = re.compile(rb"source is (\d+)x(\d+), expected (\d+)x(\d+)")
_FORMAT_NAMES = {"jpeg": "a JPEG", "gif": "a GIF", "webp": "a WebP image"}


def _invalid(message: str) -> ToolResult:
    return ToolResult(message, is_error=True, summary="invalid")


def _side_error(name: str, value: int) -> ToolResult | None:
    if MIN_SIDE <= value <= MAX_SIDE and value % SIDE_STEP == 0:
        return None
    return _invalid(f"{name} must be {MIN_SIDE}-{MAX_SIDE} and a multiple of {SIDE_STEP}")


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
        "To change only part of the image, give `region` (shapes in source pixel coordinates) or `mask` (a mask image: "
        "white edits, black keeps), optionally with `feather`; the output then keeps the source's exact size and every "
        "pixel outside the selection is the source's own. With a region or mask, `upscale` sharpens the edit with the "
        "local upscaler (SeedVR2) before it is blended in; it needs a source larger than the edit and adds minutes. "
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
            "region": {
                "type": "array",
                "maxItems": MAX_SHAPES,
                "items": {
                    "type": "object",
                    "properties": {
                        "shape": {"type": "string", "enum": ["rect", "ellipse"]},
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                        "w": {"type": "integer", "minimum": 1},
                        "h": {"type": "integer", "minimum": 1},
                    },
                    "required": ["shape", "x", "y", "w", "h"],
                },
                "description": (
                    f"1-{MAX_SHAPES} shapes to edit, in source pixel coordinates (x, y is the top-left corner of the "
                    "box, w and h its size; an ellipse is inscribed in its box). Shapes may overhang the source and are "
                    "clipped. The output keeps the source size. Not with `mask`, `width` or `height`."
                ),
            },
            "mask": {
                "type": "string",
                "description": (
                    "A mask image to edit through: a workspace path or clipboard:N. White edits, black keeps, grey "
                    "blends; any size (scaled to the source). Counts as one of the 4 input images. Not with `region`, "
                    "`width` or `height`."
                ),
            },
            "feather": {
                "type": "integer",
                "description": f"Edge softness of the region or mask in pixels, 0-{MAX_FEATHER}. Default {DEFAULT_FEATHER}.",
            },
            "upscale": {
                "type": "boolean",
                "description": "With region or mask only: sharpen the edit with the local upscaler (SeedVR2) before blending it in; adds minutes.",
            },
        },
        "required": ["prompt", "image", "path"],
    }
    mutating: ClassVar[bool] = True  # hidden and refused in plan mode
    needs_approval: ClassVar[bool] = True  # every call asks; target stays None so accept-edits never skips it
    uncancellable: ClassVar[bool | None] = False  # Esc kills the process group
    grant_arg: ClassVar[str | None] = "prompt"

    def __init__(
        self, generator: MfluxGenerator, pull: ImagePuller | None = None, upscaler: MfluxGenerator | None = None
    ) -> None:
        self.generator = generator
        self.pull = pull
        self.upscaler = upscaler

    def label(self, args: dict[str, Any]) -> str:
        """The output file, not the prompt."""
        path = args.get("path")
        return f"{self.name} {path}" if isinstance(path, str) and path else self.name

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        if self.pull is not None and self.pull.busy:
            return ToolResult(REFUSED, is_error=True, summary="refused")
        if (problem := check_args(self.parameters, args)) is not None:
            return _invalid(problem)
        upscale = bool(args.get("upscale"))
        if upscale and self.upscaler is None:
            return _invalid("upscale model not available")
        if upscale and "region" not in args and "mask" not in args:
            return _invalid("upscale needs a region or mask")
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
        selection = _check_selection(args)
        if isinstance(selection, ToolResult):
            return selection
        shapes, mask_ref, feather = selection
        if isinstance(checked := _check_output_path(raw, ctx), ToolResult):
            return checked
        path = checked
        if mask_ref is not None and len(references) + 2 > 1 + MAX_REFERENCES:
            return _invalid(f"too many input images (at most {1 + MAX_REFERENCES} with the mask)")
        found_all: list[InputImage] = []
        for ref in (args["image"], *references, *(() if mask_ref is None else (mask_ref,))):
            found = await asyncio.to_thread(resolve_input, ref, ctx)
            if isinstance(found, ToolResult):
                return found
            if any(found.identity == other.identity for other in found_all):
                return _invalid(f"{escape_display(ref)} is given twice (the same image as {found.label})")
            found_all.append(found)
        mask_item = found_all.pop() if mask_ref is not None else None  # the mask never goes to mflux
        if shapes is not None or mask_item is not None:
            for item in (found_all[0], *(() if mask_item is None else (mask_item,))):
                if item.width * item.height > MAX_COMPOSITE_PIXELS:
                    return _invalid(
                        f"{item.label} is {item.width}\u00d7{item.height}: too large to composite "
                        f"(limit {MAX_COMPOSITE_PIXELS // 1_000_000} megapixels)"
                    )
        inputs = found_all
        source = inputs[0]
        composite: Composite | None = None
        if shapes is not None or mask_item is not None:
            size = oriented_size(source)
            if shapes is not None and (outside := check_region(shapes, *size)) is not None:
                return _invalid(outside)
            composite = Composite(shapes, mask_item, feather, size)
            default_width, default_height = default_size(*size)
        else:
            default_width, default_height = default_size(source.width, source.height)
        width = args.get("width", default_width)
        height = args.get("height", default_height)
        if upscale and composite is not None:
            if max(composite.size) <= max(width, height):
                return _invalid(f"upscale is only useful when the source is larger than the edit ({width}\u00d7{height})")
            factor = edit_upscale_scale((width, height), composite.size)
            if factor is None:
                return _invalid(f"upscale is not possible for an edit of {width}\u00d7{height}")
            composite = dataclasses.replace(composite, upscale=factor, upscaled_size=upscale_size(width, height, factor))
        seed = secrets.randbelow(SEED_LIMIT) if seed is None else seed
        request = ImageRequest(prompt, width, height, steps, seed)
        model = self.generator.model
        lines = [
            f"prompt: {escape_display(prompt)}",
            f"source: {_describe_input(source)}",
            *(f"reference: {_describe_input(item)}" for item in inputs[1:]),
            f"size: {width}\u00d7{height}, {steps} steps, seed {seed}",
            *(
                [
                    describe(shapes, mask_item.label if mask_item else None, feather),
                    f"output: {composite.size[0]}\u00d7{composite.size[1]} (source size)",
                ]
                if composite is not None
                else []
            ),
            *(
                [f"upscale: {composite.upscale}\u00d7 ({escape_display(self.upscaler.model)})"]
                if composite is not None and composite.upscale is not None and self.upscaler is not None
                else []
            ),
            f"model: {escape_display(model)}",
        ]
        if path.exists():
            lines.append("[overwrites the existing file]")
        if not model_cached(model):
            lines.append(f"[model {escape_display(model)} is not downloaded; this call may download several GB]")
        if (
            composite is not None
            and composite.upscale is not None
            and self.upscaler is not None
            and not model_cached(self.upscaler.model)
        ):
            lines.append(f"[model {escape_display(self.upscaler.model)} is not downloaded; this call may download several GB]")
        preview = Preview("content", f"edit image {escape_display(raw)}", "\n".join(lines))

        async def execute() -> ToolResult:
            return await _run_mflux(
                self.generator,
                request,
                path,
                raw,
                ctx,
                "edited",
                lambda folder: stage(inputs, folder),
                composite,
                self.upscaler,
            )

        return Prepared(preview, execute)


@dataclass(frozen=True)
class Composite:
    """A masked edit: what to edit (shapes or a mask image), the feather and the source's oriented size."""

    shapes: list[Shape] | None
    mask: InputImage | None
    feather: int
    size: tuple[int, int]
    upscale: int | None = None  # edit_image(upscale=true): the SeedVR2 factor applied to the edit before the blend
    upscaled_size: tuple[int, int] | None = None  # what that step must write


def _check_selection(args: dict[str, Any]) -> tuple[list[Shape] | None, str | None, int] | ToolResult:
    """(shapes, mask reference, feather) from the arguments, or the `invalid` refusal."""
    region, mask, feather = args.get("region"), args.get("mask"), args.get("feather")
    if region is not None and mask is not None:
        return _invalid("give either region or mask, not both")
    if region is None and mask is None:
        if feather is not None:
            return _invalid("feather needs a region or a mask")
        return None, None, DEFAULT_FEATHER
    if feather is not None and not 0 <= feather <= MAX_FEATHER:
        return _invalid(f"feather must be 0-{MAX_FEATHER}")
    if "width" in args or "height" in args:
        return _invalid(
            "width and height cannot be combined with a mask or region (the output keeps the source size)"
        )
    shapes: list[Shape] | None = None
    if region is not None:
        parsed = parse_region(region)
        if isinstance(parsed, str):
            return _invalid(parsed)
        shapes = parsed
    return shapes, mask, DEFAULT_FEATHER if feather is None else feather


class UpscaleImageTool(Tool):
    name: ClassVar[str] = "upscale_image"
    description: ClassVar[str] = (
        "Enlarge an image 2x, 3x or 4x with a local model (mflux, SeedVR2) and save the result as a PNG at a path "
        "inside the workspace. `image` is the source (a workspace path, or clipboard:N for an image pasted into the "
        "latest user message). The output's longer side may be at most "
        f"{MAX_UPSCALE_SIDE} pixels. It takes minutes, replaces a file at the path (the source included), and every "
        "call needs user approval."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The source image: a workspace path or clipboard:N."},
            "path": {"type": "string", "description": "Where to save the result: a .png path inside the workspace."},
            "scale": {"type": "integer", "enum": list(UPSCALE_SCALES), "description": "Enlargement factor. Default 2."},
            "seed": {"type": "integer", "description": "Integer 0-4294967295 to make the same result again. Default random."},
            "softness": {"type": "number", "minimum": 0, "maximum": 1, "description": "0.0-1.0; higher is softer. Default 0."},
        },
        "required": ["image", "path"],
    }
    mutating: ClassVar[bool] = True  # hidden and refused in plan mode
    needs_approval: ClassVar[bool] = True  # every call asks; target stays None so accept-edits never skips it
    uncancellable: ClassVar[bool | None] = False  # Esc kills the process group
    grant_arg: ClassVar[str | None] = "path"

    def __init__(self, generator: MfluxGenerator, pull: ImagePuller | None = None) -> None:
        self.generator = generator
        self.pull = pull

    def label(self, args: dict[str, Any]) -> str:
        path = args.get("path")
        return f"{self.name} {path}" if isinstance(path, str) and path else self.name

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        if self.pull is not None and self.pull.busy:
            return ToolResult(REFUSED, is_error=True, summary="refused")
        if (problem := check_args(self.parameters, args)) is not None:
            return _invalid(problem)
        raw: str = args["path"]
        scale = args.get("scale", UPSCALE_SCALES[0])
        softness = args.get("softness", 0.0)
        seed = args.get("seed")
        if scale not in UPSCALE_SCALES:
            return _invalid(f"scale must be one of {', '.join(map(str, UPSCALE_SCALES))}")
        if isinstance(softness, bool) or not isinstance(softness, int | float) or not 0.0 <= softness <= 1.0:
            return _invalid("softness must be a number from 0.0 to 1.0")
        if seed is not None and not 0 <= seed < SEED_LIMIT:
            return _invalid(f"seed must be 0-{SEED_LIMIT - 1}")
        if isinstance(checked := _check_output_path(raw, ctx), ToolResult):
            return checked
        path = checked
        source = await asyncio.to_thread(resolve_input, args["image"], ctx)
        if isinstance(source, ToolResult):
            return source
        if source.width * source.height > MAX_COMPOSITE_PIXELS:
            return _invalid(
                f"{source.label} is {source.width}\u00d7{source.height}: too large to upscale "
                f"(limit {MAX_COMPOSITE_PIXELS // 1_000_000} megapixels)"
            )
        width, height = upscale_size(*oriented_size(source), scale)
        if max(width, height) > MAX_UPSCALE_SIDE:
            return _invalid(
                f"the output would be {width}\u00d7{height}; the longer side may be at most {MAX_UPSCALE_SIDE} "
                "(use a smaller scale)"
            )
        if min(width, height) < UPSCALE_STEP:  # mflux cuts the shorter side to a multiple of 16: 0 for a tiny source
            return _invalid(
                f"the output would be {width}\u00d7{height}; the source is too small for {scale}\u00d7 "
                f"(the shorter side must be at least {-(-UPSCALE_STEP // scale)} px)"
            )
        seed = secrets.randbelow(SEED_LIMIT) if seed is None else seed
        request = ImageRequest("", width, height, 1, seed, scale=scale, softness=float(softness))
        model = self.generator.model
        lines = [
            f"source: {_describe_input(source)}",
            f"output: {width}\u00d7{height} ({scale}\u00d7)",
            *([f"softness: {softness:g}"] if softness else []),
            f"seed: {seed}",
            f"model: {escape_display(model)}",
        ]
        if path.exists():
            lines.append("[overwrites the existing file]")
        if not model_cached(model):
            lines.append(f"[model {escape_display(model)} is not downloaded; this call may download several GB]")
        preview = Preview("content", f"upscale image {escape_display(raw)}", "\n".join(lines))

        async def execute() -> ToolResult:
            return await _run_mflux(
                self.generator, request, path, raw, ctx, "upscaled", lambda folder: stage([source], folder)
            )

        return Prepared(preview, execute)


async def _generate(generator: MfluxGenerator, request: ImageRequest, out_dir: Path, ctx: ToolContext) -> Path:
    """Run mflux; with a progress sink, a ticker sends the display line when it changes (at most 4 a second)."""
    from src.engine.events import ToolProgress  # not at module level: events imports the tools package

    sink = ctx.progress
    if sink is None:
        return await generator.generate(request, out_dir)
    return await run_with_ticker(generator, request, out_dir, lambda text: sink(ToolProgress(text)))


_TooBig = TooBig


def _read_nofollow(path: Path) -> bytes:
    """A regular file's bytes, never through a symlink, at most `MAX_OUTPUT_BYTES`."""
    return read_nofollow(path, MAX_OUTPUT_BYTES)


async def _read_png(path: Path, expected: tuple[int, int], who: str) -> tuple[bytes, tuple[int, int]] | ToolResult:
    """The bytes of a PNG that `who` wrote, checked: size limit, signature, IHDR size equal to `expected`, IEND."""
    try:
        data = await asyncio.to_thread(_read_nofollow, path)
    except _TooBig:
        return ToolResult(f"{who} wrote an image over 64 MB", is_error=True, summary="failed")
    except (FileNotFoundError, NotADirectoryError):
        return ToolResult(f"{who} wrote no image", is_error=True, summary="failed")
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):  # a symlink where the image should be
            return ToolResult(f"{who} wrote no image", is_error=True, summary="failed")
        return ToolResult(f"Cannot read the image: {exc.strerror or exc}", is_error=True, summary="failed")
    size = png_size(data)
    if size is None:
        kind = _FORMAT_NAMES.get(sniff(data[:16]) or "", "an unknown file")
        return ToolResult(f"{who} wrote {kind} instead of a PNG", is_error=True, summary="failed")
    if size != expected:
        return ToolResult(
            f"{who} wrote {size[0]}\u00d7{size[1]}, expected {expected[0]}\u00d7{expected[1]}", is_error=True, summary="failed"
        )
    if not data.endswith(_IEND):
        return ToolResult(f"{who} wrote an incomplete PNG", is_error=True, summary="failed")
    return data, size


def _timeout_message(message: str, generator: MfluxGenerator) -> str:
    """A timeout names the whole deadline, not the part left for the step that hit it."""
    if message.startswith("timed out") and generator.timeout is not None:
        return f"timed out after {generator.timeout:g} s"
    return message


def _helper_failure(code: int, stderr: bytes) -> ToolResult:
    if code == 3:
        message = "the mask selects nothing"
    elif code == 4:
        found = _SIZES.search(stderr)
        message = "the source's orientation could not be matched"
        if found is not None:
            w, h, ew, eh = (int(g) for g in found.groups())
            message = f"the source decoded as {w}\u00d7{h}, expected {ew}\u00d7{eh}"
    elif code == 5:
        message = "image too large to composite (limit 50 megapixels)"
    else:
        message = f"compositing failed (exit {code}): {_tail(stderr)}"
    return ToolResult(message, is_error=True, summary="failed")


async def _composite(
    step: str, flags: list[str], out_dir: Path, generator: MfluxGenerator, deadline: float | None
) -> ToolResult | None:
    """Run one helper step in the private folder under the shared deadline; the refusal, or None when it succeeded."""
    if COMPOSITE_HELPER is None or not COMPOSITE_HELPER.is_file():
        return ToolResult("the compositing helper is missing", is_error=True, summary="failed")
    left = None
    if deadline is not None:
        left = deadline - time.monotonic()
        if left <= 0:
            message = _timeout_message("timed out", generator)
            return ToolResult(message, is_error=True, summary="timed out")
    with contextlib.suppress(OSError):  # never write through something another process left in the way
        (out_dir / (MASK_NAME if step == "prepare" else OUTPUT_COMPOSITE)).unlink()
    argv = [sys.executable, "-I", str(COMPOSITE_HELPER), step, *flags]
    try:
        code, stderr = await run_helper(argv, out_dir, generator.guard, left)
    except ImageGenError as exc:
        message = _timeout_message(str(exc), generator)
        return ToolResult(message, is_error=True, summary="timed out" if message.startswith("timed out") else "failed")
    return None if code == 0 else _helper_failure(code, stderr)


async def _run_mflux(
    generator: MfluxGenerator,
    request: ImageRequest,
    path: Path,
    raw: str,
    ctx: ToolContext,
    verb: str,
    stager: Callable[[Path], tuple[Path, ...]] | None = None,
    composite: Composite | None = None,
    upscaler: MfluxGenerator | None = None,
) -> ToolResult:
    """Run mflux in a private folder, check the PNG it wrote and place it at `path`. `stager` (edit_image) copies the
    inputs into that folder first. With a `composite` (a region or mask), a helper process first builds the feathered
    mask and, after mflux, blends the edit over the source at the source's size; one deadline covers all three runs.
    The folder is always removed."""
    out_dir = Path(tempfile.mkdtemp(prefix="agent-image-"))
    deadline = None if generator.timeout is None else time.monotonic() + generator.timeout
    try:
        if stager is not None:
            try:
                request = dataclasses.replace(request, inputs=await _finish_placement(asyncio.to_thread(stager, out_dir)))
            except OSError as exc:
                return ToolResult(f"Cannot stage the input images: {exc.strerror or exc}", is_error=True, summary="failed")
        mask_file = out_dir / MASK_NAME
        if composite is not None:
            source = request.inputs[0]
            flags = ["--source", str(source), "--expect", str(composite.size[0]), str(composite.size[1])]
            flags += ["--feather", str(composite.feather), "--out-mask", str(mask_file)]
            try:
                if composite.mask is not None:
                    staged = await _finish_placement(asyncio.to_thread(stage_one, composite.mask, out_dir, "in-m"))
                    flags += ["--mask", str(staged)]
                else:
                    region = [{"shape": s.kind, "x": s.x, "y": s.y, "w": s.w, "h": s.h} for s in composite.shapes or []]
                    flags += ["--region", json.dumps(region)]
            except OSError as exc:
                return ToolResult(f"Cannot stage the input images: {exc.strerror or exc}", is_error=True, summary="failed")
            if (failed := await _composite("prepare", flags, out_dir, generator, deadline)) is not None:
                return failed
        run_generator = generator
        if composite is not None and deadline is not None:  # the preparation took part of the one deadline
            run_generator = copy.copy(generator)
            run_generator.timeout = max(0.1, deadline - time.monotonic())
        try:
            generated = await _generate(run_generator, request, out_dir, ctx)
        except ImageGenError as exc:
            message = _timeout_message(str(exc), generator)
            return ToolResult(message, is_error=True, summary="timed out" if message.startswith("timed out") else "failed")
        checked = await _read_png(generated, (request.width, request.height), "mflux")
        if isinstance(checked, ToolResult):
            return checked
        data, size = checked
        suffix = ""
        if composite is not None and composite.upscale is not None and composite.upscaled_size is not None and upscaler is not None:
            upscaled = await _upscale_edit(upscaler, generator, request, composite, data, out_dir, deadline, ctx)
            if isinstance(upscaled, ToolResult):
                return upscaled
            generated = upscaled
        if composite is not None:
            if ctx.progress is not None:
                from src.engine.events import ToolProgress  # not at module level: events imports the tools package

                ctx.progress(ToolProgress("compositing…"))
            blend = ["--source", str(request.inputs[0]), "--edit", str(generated), "--mask", str(mask_file)]
            blend += ["--out", str(out_dir / OUTPUT_COMPOSITE)]
            if (failed := await _composite("blend", blend, out_dir, generator, deadline)) is not None:
                return failed
            checked = await _read_png(out_dir / OUTPUT_COMPOSITE, composite.size, "compositing")
            if isinstance(checked, ToolResult):
                return checked
            data, size = checked
            suffix = ", masked" if composite.upscale is None else f", masked, upscaled {composite.upscale}\u00d7"
        try:
            refused = await _finish_placement(asyncio.to_thread(_place, path, data, ctx))
        except OSError as exc:
            return ToolResult(f"Cannot write {raw}: {exc.strerror or exc}", is_error=True, summary="write failed")
        if refused is not None:
            return refused
        described = f"PNG, {size[0]}\u00d7{size[1]}, {size_text(len(data))}"
        return ToolResult(f"{verb} {raw} ({described}, seed {request.seed}{suffix})", summary=described)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


async def _upscale_edit(
    upscaler: MfluxGenerator,
    generator: MfluxGenerator,
    request: ImageRequest,
    composite: Composite,
    data: bytes,
    out_dir: Path,
    deadline: float | None,
    ctx: ToolContext,
) -> Path | ToolResult:
    """Upscale the checked edit (written as `edit.png` from `data`) on what is left of the one deadline; the path of
    the checked upscaled PNG, or the error."""
    assert composite.upscale is not None and composite.upscaled_size is not None
    edit_file = out_dir / "edit.png"

    def write() -> None:
        fd = os.open(edit_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)

    try:
        await asyncio.to_thread(write)
    except OSError as exc:
        return ToolResult(f"Cannot stage the input images: {exc.strerror or exc}", is_error=True, summary="failed")
    with contextlib.suppress(OSError):
        (out_dir / "image.png").unlink()  # the edit's output; the upscaler writes its own
    step = copy.copy(upscaler)
    if deadline is None:
        step.timeout = None
    else:
        left = deadline - time.monotonic()
        if left <= 0:
            return ToolResult(_timeout_message("timed out", generator), is_error=True, summary="timed out")
        step.timeout = max(0.1, left)
    if ctx.progress is not None:
        from src.engine.events import ToolProgress  # not at module level: events imports the tools package

        ctx.progress(ToolProgress("upscaling…"))
    up_request = ImageRequest(
        "", *composite.upscaled_size, 1, request.seed, inputs=(edit_file,), scale=composite.upscale
    )
    try:
        result = await _generate(step, up_request, out_dir, ctx)
    except ImageGenError as exc:
        message = _timeout_message(str(exc), generator)
        return ToolResult(message, is_error=True, summary="timed out" if message.startswith("timed out") else "failed")
    checked = await _read_png(result, composite.upscaled_size, "mflux")
    if isinstance(checked, ToolResult):
        return checked
    return result
