"""Image files: format by magic bytes, size limits, read once, filtering and capping."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import pytest
from core.engine.provider.base import Image, Message
from core.engine.provider.images import (
    IMAGE_TOKENS,
    MAX_IMAGE_BYTES,
    MAX_IMAGES,
    ImageError,
    cap_images,
    describe,
    filter_images,
    read_image_fd,
    size_text,
    sniff,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 8
JPEG = b"\xff\xd8\xff\xe0" + b"\0" * 8
GIF87 = b"GIF87a" + b"\0" * 8
GIF89 = b"GIF89a" + b"\0" * 8
WEBP = b"RIFF\x01\x02\x03\x04WEBPVP8 "


@pytest.mark.parametrize(
    ("head", "expected"),
    [(PNG, "png"), (JPEG, "jpeg"), (GIF87, "gif"), (GIF89, "gif"), (WEBP, "webp")],
)
def test_sniff_knows_each_format(head: bytes, expected: str) -> None:
    assert sniff(head) == expected


@pytest.mark.parametrize(
    "head",
    [
        b"",
        b"\x89PNG\r\n\x00\x00data",  # PNG start with a wrong tail
        b"\xff\xd8",  # truncated JPEG
        b"GIF88a" + b"\0" * 8,
        b"RIFF\x01\x02\x03\x04WAVEfmt ",
        b"RIFF\x01\x02\x03\x04WEB",  # truncated WebP
        b"plain text, not an image",
    ],
)
def test_sniff_rejects_near_misses(head: bytes) -> None:
    assert sniff(head) is None


@pytest.mark.parametrize(
    ("size", "text"),
    [(0, "1 KB"), (1, "1 KB"), (1024, "1 KB"), (214 * 1024, "214 KB"), (int(4.8 * 1024 * 1024), "4.8 MB")],
)
def test_size_text(size: int, text: str) -> None:
    assert size_text(size) == text


def test_describe_names_the_format_and_size() -> None:
    assert describe(Image("png", b"x" * 2048, "a.png")) == "PNG, 2 KB"
    assert describe(Image("jpeg", b"x", "a")) == "JPEG, 1 KB"


def test_image_repr_never_shows_the_bytes() -> None:
    assert "secretbytes" not in repr(Image("png", b"secretbytes", "a.png"))


def test_message_images_default_to_empty() -> None:
    assert Message("user", "x").images == ()
    assert Message("user", "x") == Message("user", "x", images=())


def _fd(path: Path) -> int:
    return os.open(path, os.O_RDONLY)


def test_read_image_fd_returns_the_image(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    path.write_bytes(PNG + b"body")
    fd = _fd(path)
    try:
        image = read_image_fd(fd, "a.png")
    finally:
        os.close(fd)
    assert image == Image("png", PNG + b"body", "a.png")


def test_read_image_fd_ignores_the_file_name(tmp_path: Path) -> None:
    path = tmp_path / "img.dat"
    path.write_bytes(JPEG)
    fd = _fd(path)
    try:
        assert read_image_fd(fd, "img.dat") is not None
    finally:
        os.close(fd)


def test_read_image_fd_returns_none_for_text_named_png(tmp_path: Path) -> None:
    path = tmp_path / "fake.png"
    path.write_text("hello")
    fd = _fd(path)
    try:
        assert read_image_fd(fd, "fake.png") is None
    finally:
        os.close(fd)


def test_read_image_fd_refuses_over_the_cap(tmp_path: Path) -> None:
    path = tmp_path / "big.png"
    path.write_bytes(PNG + b"\0" * (6 * 1024 * 1024))
    fd = _fd(path)
    try:
        with pytest.raises(ImageError) as caught:
            read_image_fd(fd, "big.png")
    finally:
        os.close(fd)
    assert caught.value.reason == "image too large (6.0 MB; limit 5 MB)"


def test_read_image_fd_accepts_exactly_the_cap(tmp_path: Path) -> None:
    path = tmp_path / "edge.png"
    path.write_bytes(PNG + b"\0" * (MAX_IMAGE_BYTES - len(PNG)))
    fd = _fd(path)
    try:
        assert read_image_fd(fd, "edge.png") is not None
    finally:
        os.close(fd)


def test_read_image_fd_reads_the_bytes_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "a.png"
    path.write_bytes(PNG + b"x" * 100)
    reads: list[int] = []
    real = os.pread

    def counting(fd: int, n: int, offset: int) -> bytes:
        data = real(fd, n, offset)
        reads.append(len(data))
        return data

    monkeypatch.setattr(os, "pread", counting)
    fd = _fd(path)
    try:
        image = read_image_fd(fd, "a.png")
    finally:
        os.close(fd)
    assert image is not None
    assert sum(reads) == len(PNG) + 100  # no byte is read twice


def test_attaching_a_5mb_png_costs_under_100ms(tmp_path: Path) -> None:
    path = tmp_path / "big.png"
    path.write_bytes(PNG + os.urandom(MAX_IMAGE_BYTES - len(PNG)))
    fd = _fd(path)
    try:
        start = time.perf_counter()
        image = read_image_fd(fd, "big.png")
        assert image is not None
        base64.b64encode(image.data)
        elapsed = time.perf_counter() - start
    finally:
        os.close(fd)
    assert elapsed < 0.1


def _images(n: int) -> tuple[Image, ...]:
    return tuple(Image("png", PNG, f"i{k}.png") for k in range(n))


def test_filter_images_keeps_all_with_vision() -> None:
    images = _images(2)
    assert filter_images(images, True, "m") == (images, [])


def test_filter_images_drops_all_without_vision_with_escaped_notes() -> None:
    images = (Image("png", PNG, "a\x1b[2J.png"),)
    kept, notes = filter_images(images, False, "llama3.1:8b")
    assert kept == ()
    assert notes == ["not attached a\\x1b[2J.png (model 'llama3.1:8b' has no vision)"]


def test_cap_images_keeps_the_first_ten() -> None:
    kept, notes = cap_images(_images(MAX_IMAGES + 1))
    assert len(kept) == MAX_IMAGES
    assert notes == ["not attached i10.png (image limit (10 per message))"]


def test_cap_images_leaves_ten_alone() -> None:
    images = _images(MAX_IMAGES)
    assert cap_images(images) == (images, [])


def test_image_tokens_constant() -> None:
    assert IMAGE_TOKENS == 800
