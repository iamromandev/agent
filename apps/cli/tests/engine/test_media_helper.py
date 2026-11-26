"""The media helper (frames and captions for assemble_video), run for real as a separate process. Needs Pillow."""

from __future__ import annotations

import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

# Loaded through importorskip, not `import`: CI has no media extra, and ty must not need Pillow to check this file.
Image = pytest.importorskip("PIL.Image")

HELPER = Path(__file__).resolve().parents[2] / "src" / "engine" / "media_helper.py"


def run(*args: object) -> subprocess.CompletedProcess[str]:
    argv = [sys.executable, "-I", str(HELPER), *map(str, args)]
    return subprocess.run(argv, capture_output=True, text=True, timeout=60)


def frame(tmp: Path, source: Path, width: int, height: int) -> Image.Image:
    out = tmp / "frame-0.png"
    done = run("frame", "--in", source, "--out", out, "--width", width, "--height", height)
    assert done.returncode == 0, done.stderr
    image = Image.open(out)
    image.load()
    return image


def near(pixel: tuple[int, ...], expected: tuple[int, int, int], slack: int = 3) -> bool:
    return all(abs(a - b) <= slack for a, b in zip(pixel[:3], expected, strict=True))


def test_exif_orientation_6_turns_the_picture_a_quarter(tmp_path: Path) -> None:
    image = Image.new("RGB", (40, 20), (255, 0, 0))  # left half red, right half blue
    image.paste((0, 0, 255), (20, 0, 40, 20))
    exif = Image.Exif()
    exif[0x0112] = 6
    image.save(tmp_path / "in.png", exif=exif)
    out = frame(tmp_path, tmp_path / "in.png", 20, 40)
    assert out.size == (20, 40)
    assert near(out.getpixel((10, 2)), (255, 0, 0)) and near(out.getpixel((10, 37)), (0, 0, 255))


def test_transparency_becomes_black(tmp_path: Path) -> None:
    Image.new("RGBA", (30, 30), (255, 0, 0, 0)).save(tmp_path / "in.png")
    out = frame(tmp_path, tmp_path / "in.png", 30, 30)
    assert out.mode == "RGB" and out.getpixel((5, 5)) == (0, 0, 0)


def test_half_transparent_white_becomes_mid_grey(tmp_path: Path) -> None:
    Image.new("RGBA", (30, 30), (255, 255, 255, 128)).save(tmp_path / "in.png")
    assert near(frame(tmp_path, tmp_path / "in.png", 30, 30).getpixel((5, 5)), (128, 128, 128))


@pytest.mark.parametrize("size", [(300, 100), (100, 300), (50, 50), (1000, 400)])
def test_cover_crop_gives_exactly_the_requested_size(tmp_path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size, (10, 20, 30)).save(tmp_path / "in.png")
    assert frame(tmp_path, tmp_path / "in.png", 60, 100).size == (60, 100)


def test_cover_crop_keeps_the_centre(tmp_path: Path) -> None:
    image = Image.new("RGB", (300, 100), (255, 0, 0))
    image.paste((0, 255, 0), (100, 0, 200, 100))
    image.paste((0, 0, 255), (200, 0, 300, 100))
    image.save(tmp_path / "in.png")
    out = frame(tmp_path, tmp_path / "in.png", 100, 100)
    assert all(near(out.getpixel((x, 50)), (0, 255, 0)) for x in (5, 50, 95))


def test_a_gif_gives_its_first_frame(tmp_path: Path) -> None:
    first, second = Image.new("RGB", (20, 20), (255, 0, 0)), Image.new("RGB", (20, 20), (0, 0, 255))
    first.save(tmp_path / "in.gif", save_all=True, append_images=[second], duration=100, loop=0)
    assert near(frame(tmp_path, tmp_path / "in.gif", 20, 20).getpixel((10, 10)), (255, 0, 0))


def test_a_jpeg_is_decoded(tmp_path: Path) -> None:
    Image.new("RGB", (64, 32), (200, 100, 50)).save(tmp_path / "in.jpg")
    assert near(frame(tmp_path, tmp_path / "in.jpg", 64, 32).getpixel((5, 5)), (200, 100, 50), 6)


def caption(tmp: Path, text: str) -> Image.Image:
    (tmp / "cap-0.txt").write_text(text, encoding="utf-8")
    done = run("caption", "--text-file", tmp / "cap-0.txt", "--out", tmp / "cap-0.png")
    assert done.returncode == 0, done.stderr
    image = Image.open(tmp / "cap-0.png")
    image.load()
    return image


def test_a_caption_is_a_transparent_frame_with_white_text_in_the_lower_third(tmp_path: Path) -> None:
    image = caption(tmp_path, "Every morning, Mia\nclimbed the hill.")
    assert image.size == (1080, 1920) and image.mode == "RGBA"
    box = image.getchannel("A").getbbox()
    assert box is not None
    assert box[1] >= 1280 and box[3] <= 1920, box  # drawn pixels inside the lower third
    assert box[0] > 0 and box[2] < 1080
    assert abs(box[0] - (1080 - box[2])) <= 40  # centred
    assert image.getpixel((540, 100))[3] == 0
    colors = [color for _, color in image.getcolors(maxcolors=1_000_000)]
    assert (255, 255, 255, 255) in colors  # white text
    assert (0, 0, 0, 255) in colors  # dark outline


def test_the_widest_caption_still_fits_the_frame(tmp_path: Path) -> None:
    text = "\n".join(["W" * 32] * 4)
    box = caption(tmp_path, text).getchannel("A").getbbox()
    assert box is not None and box[0] >= 0 and box[2] <= 1080 and box[1] >= 1280 and box[3] <= 1920


def test_a_caption_with_markup_characters_is_drawn_as_text(tmp_path: Path) -> None:
    box = caption(tmp_path, "it's 100% \\n done; drawtext=x %{pts}").getchannel("A").getbbox()
    assert box is not None


def _bomb_png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")


@pytest.mark.parametrize("size", [(10000, 6000), (20000, 20000)])
def test_an_image_over_50_megapixels_exits_5(tmp_path: Path, size: tuple[int, int]) -> None:
    (tmp_path / "big.png").write_bytes(_bomb_png(*size))
    done = run("frame", "--in", tmp_path / "big.png", "--out", tmp_path / "o.png", "--width", 10, "--height", 10)
    assert done.returncode == 5
    assert "50 megapixels" in done.stderr
    assert not (tmp_path / "o.png").exists()


def test_garbage_exits_3(tmp_path: Path) -> None:
    (tmp_path / "bad.png").write_bytes(b"not an image" * 20)
    done = run("frame", "--in", tmp_path / "bad.png", "--out", tmp_path / "o.png", "--width", 10, "--height", 10)
    assert done.returncode == 3 and not (tmp_path / "o.png").exists()


def test_a_missing_input_exits_3(tmp_path: Path) -> None:
    done = run("frame", "--in", tmp_path / "none.png", "--out", tmp_path / "o.png", "--width", 10, "--height", 10)
    assert done.returncode == 3


def test_usage_errors_exit_2(tmp_path: Path) -> None:
    assert run().returncode == 2
    assert run("frame", "--in", tmp_path / "x.png").returncode == 2
    assert run("caption").returncode == 2


def test_a_text_file_that_is_not_utf8_exits_3(tmp_path: Path) -> None:
    (tmp_path / "c.txt").write_bytes(b"\xff\xfe\x00bad")
    done = run("caption", "--text-file", tmp_path / "c.txt", "--out", tmp_path / "o.png")
    assert done.returncode == 3
