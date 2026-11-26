"""Input images for edit_image: a workspace file or a pasted image, checked once and copied into a private folder.

UI-free. The bytes the checks read are the bytes mflux gets: it never opens the user's file or the paste folder.
"""

from __future__ import annotations

import os
import stat
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.engine.guardrails.paths import fd_path
from src.engine.guardrails.setup import guard_of
from src.engine.imagegen import MAX_SIDE, MIN_SIDE, SIDE_STEP
from src.engine.provider.base import ImageFormat
from src.engine.provider.images import MAX_IMAGE_BYTES, ImageError, read_image_fd
from src.engine.text import escape_display
from src.engine.tools.base import ToolContext, ToolResult
from src.engine.tools.paths import InvalidPath, inside_workspace, invalid_path_result, resolve_path
from src.engine.tools.protected import refuse_protected

MAX_REFERENCES = 3
PASTE_PREFIX = "clipboard:"
_EXTENSIONS: dict[str, str] = {"png": "png", "jpeg": "jpg", "gif": "gif", "webp": "webp"}


@dataclass(frozen=True)
class InputImage:
    label: str  # escaped for display: "docs/hero.png" or "clipboard:1"
    fmt: ImageFormat
    data: bytes = field(repr=False)
    width: int
    height: int
    identity: tuple[int, int] | tuple[str, int]  # (st_dev, st_ino) for a file, ("clipboard", n) for a paste


DEFAULT_LONG_SIDE = 1024


def _round_side(value: float) -> int:
    return max(MIN_SIDE, min(MAX_SIDE, int(value / SIDE_STEP + 0.5) * SIDE_STEP))


def default_size(width: int, height: int) -> tuple[int, int]:
    """The default output size: the source scaled down (never up) so its longer side is at most 1024, each side
    rounded to the nearest multiple of 16 (half up) and clamped to the allowed range."""
    scale = min(1.0, DEFAULT_LONG_SIDE / max(width, height))
    return _round_side(width * scale), _round_side(height * scale)


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    pos, end = 2, len(data)
    while pos + 4 <= end:
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker == 0xFF:
            pos += 1
            continue
        if marker in (0xD9, 0xDA):  # EOI or SOS: no frame header can follow
            return None
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if pos + 9 > end:
                return None
            height, width = struct.unpack(">HH", data[pos + 5 : pos + 9])
            return (width, height) if width and height else None
        (length,) = struct.unpack(">H", data[pos + 2 : pos + 4])
        pos += 2 + max(length, 2)
    return None


def _tiff_orientation(tiff: bytes) -> int:
    """Orientation (tag 0x0112) from the first IFD of a TIFF block, 1 when absent or malformed."""
    if len(tiff) < 8 or tiff[:2] not in (b"II", b"MM"):
        return 1
    e = "<" if tiff[:2] == b"II" else ">"
    if struct.unpack(e + "H", tiff[2:4])[0] != 42:
        return 1
    (offset,) = struct.unpack(e + "I", tiff[4:8])
    if offset + 2 > len(tiff):
        return 1
    (count,) = struct.unpack(e + "H", tiff[offset : offset + 2])
    for index in range(min(count, 256)):
        entry = offset + 2 + index * 12
        if entry + 12 > len(tiff):
            return 1
        tag, kind, number = struct.unpack(e + "HHI", tiff[entry : entry + 8])
        if tag == 0x0112:
            if kind != 3 or number != 1:
                return 1
            (value,) = struct.unpack(e + "H", tiff[entry + 8 : entry + 10])
            return value if 1 <= value <= 8 else 1
    return 1


def jpeg_orientation(data: bytes) -> int:
    """The EXIF orientation (1-8) of a JPEG from its APP1 segment, 1 when absent or malformed. Stops at SOS/EOI."""
    pos, end = 2, len(data)
    while pos + 4 <= end:
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker == 0xFF:
            pos += 1
            continue
        if marker in (0xD9, 0xDA):
            return 1
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        (length,) = struct.unpack(">H", data[pos + 2 : pos + 4])
        if marker == 0xE1 and data[pos + 4 : pos + 10] == b"Exif\0\0":
            return _tiff_orientation(data[pos + 10 : pos + 2 + length])
        pos += 2 + max(length, 2)
    return 1


def _tiff_block(block: bytes) -> bytes:
    return block[6:] if block[:6] == b"Exif\0\0" else block


def png_orientation(data: bytes) -> int:
    """The orientation from a PNG `eXIf` chunk, 1 when absent or malformed. Stops at IDAT or IEND."""
    pos, end = 8, len(data)
    while pos + 8 <= end:
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        if kind in (b"IDAT", b"IEND"):
            return 1
        if kind == b"eXIf":
            return _tiff_orientation(_tiff_block(data[pos + 8 : pos + 8 + length]))
        pos += 12 + length
    return 1


def webp_orientation(data: bytes) -> int:
    """The orientation from a WebP RIFF `EXIF` chunk (VP8X files), 1 when absent or malformed."""
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return 1
    pos, end = 12, min(len(data), 8 + int.from_bytes(data[4:8], "little"))
    while pos + 8 <= end:
        kind = data[pos : pos + 4]
        length = int.from_bytes(data[pos + 4 : pos + 8], "little")
        if kind == b"EXIF":
            return _tiff_orientation(_tiff_block(data[pos + 8 : pos + 8 + length]))
        pos += 8 + length + (length & 1)
    return 1


def oriented_size(item: InputImage) -> tuple[int, int]:
    """(width, height) as displayed: an EXIF orientation 5-8 (JPEG, PNG or WebP) turns the image a quarter."""
    reader = {"jpeg": jpeg_orientation, "png": png_orientation, "webp": webp_orientation}.get(item.fmt)
    if reader is not None and reader(item.data) >= 5:
        return item.height, item.width
    return item.width, item.height


def _webp_size(data: bytes) -> tuple[int, int] | None:
    kind = data[12:16]
    if kind == b"VP8 " and len(data) >= 30 and data[23:26] == b"\x9d\x01\x2a":
        width, height = struct.unpack("<HH", data[26:30])
        width, height = width & 0x3FFF, height & 0x3FFF
    elif kind == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        (bits,) = struct.unpack("<I", data[21:25])
        width, height = (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    elif kind == b"VP8X" and len(data) >= 30:
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
    else:
        return None
    return (width, height) if width and height else None


def image_dimensions(fmt: ImageFormat, data: bytes) -> tuple[int, int] | None:
    """(width, height) from the header of a PNG, JPEG, GIF or WebP, or None when the header is cut short or odd."""
    size: tuple[int, int] | None = None
    if fmt == "png":
        if len(data) >= 24 and data[12:16] == b"IHDR":
            size = struct.unpack(">II", data[16:24])
    elif fmt == "jpeg":
        size = _jpeg_size(data)
    elif fmt == "gif":
        if len(data) >= 10:
            size = struct.unpack("<HH", data[6:10])
    elif fmt == "webp":
        size = _webp_size(data)
    return size if size is not None and size[0] > 0 and size[1] > 0 else None


def _error(message: str, summary: str) -> ToolResult:
    return ToolResult(message, is_error=True, summary=summary)


def _paste(ref: str, ctx: ToolContext) -> InputImage | ToolResult:
    shown = escape_display(ref)
    number = ref[len(PASTE_PREFIX) :]
    index = int(number) if number.isascii() and number.isdigit() else 0
    if not 1 <= index <= len(ctx.pasted):
        if ctx.pasted:
            labels = ", ".join(f"{PASTE_PREFIX}{n}" for n in range(1, len(ctx.pasted) + 1))
            return _error(f"{shown}: no such paste; the pasted images are {labels}", "no such paste")
        return _error(f"{shown}: no such paste; there are no pasted images in this message", "no such paste")
    image = ctx.pasted[index - 1]
    size = image_dimensions(image.format, image.data)
    if size is None:
        return _error(f"{shown} is not a readable image", "not an image")
    return InputImage(f"{PASTE_PREFIX}{index}", image.format, image.data, size[0], size[1], ("clipboard", index))


def resolve_input(ref: str, ctx: ToolContext, max_bytes: int = MAX_IMAGE_BYTES) -> InputImage | ToolResult:
    """Check one input reference (a workspace path or `clipboard:N`) and read its bytes once. An error is a
    `ToolResult`; a protected file is refused without a dialog, because its bytes go to another process. A file
    over `max_bytes` (5 MB unless the caller allows more) is refused."""
    if ref.startswith(PASTE_PREFIX):
        return _paste(ref, ctx)
    shown = escape_display(ref)
    try:
        path = resolve_path(ctx, ref)
    except InvalidPath as exc:
        return invalid_path_result(exc)
    from src.engine.tools.fs import _refuse_outside  # not at module level: fs imports the tools package

    if not inside_workspace(path, ctx):
        return _refuse_outside()
    guard = guard_of(ctx)
    if guard.paths.is_protected(path):
        return refuse_protected()
    if path.is_dir():
        return _error(f"{shown} is a directory", "directory")
    if not path.exists():
        return _error(f"No such file: {shown}", "not found")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)  # NONBLOCK: a FIFO must not hang the open
    except OSError as exc:
        return _error(f"Cannot read {shown}: {escape_display(exc.strerror or str(exc))}", "unreadable")
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return _error(f"{shown} is not a regular file", "not a regular file")
        where = fd_path(fd)  # what was opened, not what was checked: a folder above may have been swapped
        if where is None or not inside_workspace(Path(where), ctx):
            return _refuse_outside()
        if guard.paths.is_protected_fd(fd):
            return refuse_protected()
        try:
            image = read_image_fd(fd, ref, max_bytes)
        except ImageError as exc:
            return _error(f"{shown}: {exc.reason}", "image too large")
        if image is None:
            return _error(f"{shown} is not an image (PNG, JPEG, GIF or WebP)", "not an image")
        size = image_dimensions(image.format, image.data)
        if size is None:
            return _error(f"{shown} is not a readable image", "not an image")
        return InputImage(shown, image.format, image.data, size[0], size[1], (info.st_dev, info.st_ino))
    except OSError as exc:
        return _error(f"Cannot read {shown}: {escape_display(exc.strerror or str(exc))}", "unreadable")
    finally:
        os.close(fd)


def stage_one(item: InputImage, folder: Path, name: str) -> Path:
    """Write one input as `<name>.<ext>` (mode 0600) into `folder`."""
    path = folder / f"{name}.{_EXTENSIONS[item.fmt]}"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(item.data)
    return path


def stage(inputs: Sequence[InputImage], folder: Path) -> tuple[Path, ...]:
    """Write each input as `in-<i>.<ext>` (mode 0600) into `folder`, in order. The source goes first."""
    return tuple(stage_one(item, folder, f"in-{index}") for index, item in enumerate(inputs))
