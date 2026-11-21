"""Image files for vision models: the format is read from the first bytes, never from the name (UI-free)."""

from __future__ import annotations

import os
from collections.abc import Sequence

from src.engine.text import escape

from .base import Image, ImageFormat
from .models import shown_name

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGES = 10  # per user message
IMAGE_TOKENS = 800  # what the context estimate counts for one image
SNIFF = 12  # bytes `sniff` needs


class ImageError(Exception):
    """An image that is refused; `reason` is meant for a `not attached` note."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def sniff(head: bytes) -> ImageFormat | None:
    """The image format the first bytes of a file show (`png`, `jpeg`, `gif`, `webp`), else None."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return None


def size_text(n: int) -> str:
    """`214 KB` or `4.8 MB` (1 KB = 1024 bytes; at least 1 KB)."""
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{max(round(n / 1024), 1)} KB"


def describe(image: Image) -> str:
    return f"{image.format.upper()}, {size_text(len(image.data))}"


def read_image_fd(fd: int, name: str) -> Image | None:
    """The image in the open file `fd`, or None when its first bytes are no image (the caller goes on with text;
    the offset of `fd` is not moved). Raises `ImageError` for one over `MAX_IMAGE_BYTES`. The bytes are read once,
    by size. The caller opened `fd` and decided what it may be (links, protected paths)."""
    head = os.pread(fd, SNIFF, 0)
    fmt = sniff(head)
    if fmt is None:
        return None
    size = os.fstat(fd).st_size
    if size > MAX_IMAGE_BYTES:
        raise ImageError(f"image too large ({size_text(size)}; limit 5 MB)")
    chunks = [head[:size]]
    have = len(chunks[0])
    while have < size:
        chunk = os.pread(fd, size - have, have)
        if not chunk:
            break
        chunks.append(chunk)
        have += len(chunk)
    return Image(fmt, b"".join(chunks), name)


def filter_images(images: Sequence[Image], has_vision: bool, model: str) -> tuple[tuple[Image, ...], list[str]]:
    """`images` as they are with vision; without it none, and one note per image."""
    if has_vision or not images:
        return tuple(images), []
    return (), [f"not attached {escape(i.name)} (model '{shown_name(model)}' has no vision)" for i in images]


def cap_images(images: Sequence[Image]) -> tuple[tuple[Image, ...], list[str]]:
    """The first `MAX_IMAGES` of `images`, and a note for each one past them."""
    kept, rest = tuple(images[:MAX_IMAGES]), images[MAX_IMAGES:]
    return kept, [f"not attached {escape(i.name)} (image limit ({MAX_IMAGES} per message))" for i in rest]
