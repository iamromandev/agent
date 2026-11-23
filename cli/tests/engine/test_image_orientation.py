"""JPEG EXIF orientation, the oriented size and the mask staging name (AC10 parse)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from src.engine.image_inputs import InputImage, jpeg_orientation, oriented_size, stage_one
from src.engine.provider.base import ImageFormat

from tests.engine.test_image_inputs import jpeg
from tests.image_fakes import png


def exif_jpeg(width: int, height: int, orientation: int, *, big: bool = False) -> bytes:
    """A JPEG whose APP1 Exif segment holds one orientation entry (II or MM)."""
    order, pack = (b"MM", ">") if big else (b"II", "<")
    tiff = order + struct.pack(pack + "H", 42) + struct.pack(pack + "I", 8)
    tiff += struct.pack(pack + "H", 1) + struct.pack(pack + "HHI", 0x0112, 3, 1) + struct.pack(pack + "HH", orientation, 0)
    tiff += struct.pack(pack + "I", 0)
    body = b"Exif\0\0" + tiff
    app1 = b"\xff\xe1" + struct.pack(">H", len(body) + 2) + body
    base = jpeg(width, height)
    return base[:2] + app1 + base[2:]


@pytest.mark.parametrize("big", [False, True])
@pytest.mark.parametrize("value", [1, 2, 3, 6, 8])
def test_orientation_is_read_from_the_exif_segment(value: int, big: bool) -> None:
    assert jpeg_orientation(exif_jpeg(30, 20, value, big=big)) == value


@pytest.mark.parametrize(
    "data",
    [
        jpeg(30, 20),
        b"\xff\xd8\xff\xe1\x00\x08Exif\0\0",
        b"\xff\xd8\xff\xe1\x00\x20Exif\0\0II*\0\xff\xff\xff\xff",
        exif_jpeg(30, 20, 6)[:30],
        exif_jpeg(30, 20, 9),
        b"",
    ],
)
def test_orientation_is_1_when_absent_malformed_or_out_of_range(data: bytes) -> None:
    assert jpeg_orientation(data) == 1


def _item(fmt: ImageFormat, data: bytes, width: int, height: int) -> InputImage:
    return InputImage("x", fmt, data, width, height, ("x", 1))


def test_oriented_size_swaps_sides_for_orientations_5_to_8_of_a_jpeg_only() -> None:
    assert oriented_size(_item("jpeg", exif_jpeg(30, 20, 6), 30, 20)) == (20, 30)
    assert oriented_size(_item("jpeg", exif_jpeg(30, 20, 3), 30, 20)) == (30, 20)
    assert oriented_size(_item("png", png(30, 20), 30, 20)) == (30, 20)


def test_stage_one_writes_the_named_private_copy(tmp_path: Path) -> None:
    path = stage_one(_item("png", png(4, 4), 4, 4), tmp_path, "in-m")
    assert path == tmp_path / "in-m.png"
    assert path.read_bytes() == png(4, 4)
    assert path.stat().st_mode & 0o777 == 0o600


def _pillow_image(fmt: str, width: int, height: int, orientation: int) -> bytes:
    import io

    pil = pytest.importorskip("PIL.Image")
    image = pil.new("RGB", (width, height), (10, 20, 30))
    exif = pil.Exif()
    exif[0x0112] = orientation
    out = io.BytesIO()
    image.save(out, fmt, exif=exif)
    return out.getvalue()


@pytest.mark.parametrize(("fmt", "name"), [("PNG", "png"), ("WEBP", "webp")])
def test_oriented_size_reads_png_and_webp_exif_orientation(fmt: str, name: ImageFormat) -> None:
    from src.engine.image_inputs import image_dimensions

    data = _pillow_image(fmt, 30, 20, 6)
    assert image_dimensions(name, data) == (30, 20)
    assert oriented_size(_item(name, data, 30, 20)) == (20, 30)
    plain = _pillow_image(fmt, 30, 20, 1)
    assert oriented_size(_item(name, plain, 30, 20)) == (30, 20)


@pytest.mark.parametrize("fmt", ["PNG", "WEBP"])
def test_malformed_png_and_webp_exif_keeps_the_size(fmt: str) -> None:
    name: ImageFormat = "png" if fmt == "PNG" else "webp"
    data = _pillow_image(fmt, 30, 20, 6)
    cut = data.replace(b"II*\x00", b"XX*\x00").replace(b"MM\x00*", b"XX\x00*")
    assert oriented_size(_item(name, cut, 30, 20)) == (30, 20)
    assert oriented_size(_item(name, data[:40], 30, 20)) == (30, 20)
