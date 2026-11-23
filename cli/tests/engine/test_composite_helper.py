"""The compositing helper, run for real as a separate process (AC1-AC4, AC8, AC10, NFR2, NFR3). Needs Pillow."""

from __future__ import annotations

import io
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Loaded through importorskip, not `import`: CI has no image extra, and ty must not need Pillow to check this file.
Image = pytest.importorskip("PIL.Image")
ImageOps = pytest.importorskip("PIL.ImageOps")
np = pytest.importorskip("numpy")

HELPER = Path(__file__).resolve().parents[2] / "src" / "engine" / "composite_helper.py"


def run(*args: object) -> subprocess.CompletedProcess[str]:
    argv = [sys.executable, "-I", str(HELPER), *map(str, args)]
    return subprocess.run(argv, capture_output=True, text=True, timeout=60)


def gradient(width: int, height: int) -> Image.Image:
    x = np.arange(width, dtype=np.uint8)[None, :].repeat(height, 0)
    y = np.arange(height, dtype=np.uint8)[:, None].repeat(width, 1)
    return Image.fromarray(np.stack([x, y, (x // 2 + y // 2).astype(np.uint8)], axis=2), "RGB")


def solid(width: int, height: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def rect(x: int, y: int, w: int, h: int, shape: str = "rect") -> dict[str, object]:
    return {"shape": shape, "x": x, "y": y, "w": w, "h": h}


def composite(
    tmp: Path,
    source: Path,
    edit: Image.Image,
    size: tuple[int, int],
    *,
    region: list[dict[str, object]] | None = None,
    mask: Path | None = None,
    feather: int = 0,
) -> np.ndarray:
    out_mask = tmp / "mask.png"
    selector = ["--region", json.dumps(region)] if region is not None else ["--mask", mask]
    done = run("prepare", "--source", source, "--expect", *size, "--feather", feather, *selector, "--out-mask", out_mask)
    assert done.returncode == 0, done.stderr
    edit_path = tmp / "edit.png"
    edit.save(edit_path)
    done = run("blend", "--source", source, "--edit", edit_path, "--mask", out_mask, "--out", tmp / "out.png")
    assert done.returncode == 0, done.stderr
    with Image.open(tmp / "out.png") as image:
        assert image.mode == "RGB"
        return np.asarray(image).copy()


RED = (255, 0, 0)


def test_a_rect_region_edits_only_the_box_and_keeps_the_rest_exactly(tmp_path: Path) -> None:
    src = gradient(200, 150)
    src.save(tmp_path / "s.png")
    out = composite(tmp_path, tmp_path / "s.png", solid(200, 150, RED), (200, 150), region=[rect(0, 0, 100, 150)])
    base = np.asarray(src)
    assert (out[:, :100] == RED).all()
    assert (out[:, 100:] == base[:, 100:]).all()


def test_an_ellipse_region_leaves_the_box_corner_alone(tmp_path: Path) -> None:
    src = gradient(200, 150)
    src.save(tmp_path / "s.png")
    out = composite(
        tmp_path, tmp_path / "s.png", solid(200, 150, RED), (200, 150), region=[rect(50, 25, 100, 100, "ellipse")]
    )
    base = np.asarray(src)
    assert tuple(out[75, 100]) == RED
    assert (out[25, 50] == base[25, 50]).all()
    assert (out[0, 0] == base[0, 0]).all()


def test_a_region_larger_than_the_source_is_clipped(tmp_path: Path) -> None:
    src = gradient(40, 30)
    src.save(tmp_path / "s.png")
    out = composite(tmp_path, tmp_path / "s.png", solid(40, 30, RED), (40, 30), region=[rect(-50, -50, 70, 80)])
    assert (out[:, :20] == RED).all()
    assert (out[:, 20:] == np.asarray(src)[:, 20:]).all()


def test_feather_blends_the_edge_and_leaves_pixels_beyond_it_exact(tmp_path: Path) -> None:
    src = solid(200, 100, (0, 0, 0))
    src.save(tmp_path / "s.png")
    out = composite(tmp_path, tmp_path / "s.png", solid(200, 100, (240, 0, 0)), (200, 100), region=[rect(0, 0, 100, 100)], feather=8)
    row = out[50, :, 0].astype(int)
    assert (row[:90] == 240).all()
    assert (row[109:] == 0).all()
    assert any(0 < v < 240 for v in row[92:109])
    assert all(a >= b for a, b in itertools.pairwise(row))


def test_a_mask_image_is_scaled_to_the_source_and_white_edits(tmp_path: Path) -> None:
    src = gradient(160, 120)
    src.save(tmp_path / "s.png")
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[:30] = 255
    Image.fromarray(mask, "L").save(tmp_path / "m.png")
    out = composite(tmp_path, tmp_path / "s.png", solid(160, 120, RED), (160, 120), mask=tmp_path / "m.png")
    base = np.asarray(src)
    assert (out[:58] == RED).all()
    assert (out[62:] == base[62:]).all()


def test_a_grey_mask_blends_proportionally_and_alpha_is_ignored(tmp_path: Path) -> None:
    src = solid(20, 20, (0, 0, 0))
    src.save(tmp_path / "s.png")
    Image.new("RGBA", (20, 20), (128, 128, 128, 0)).save(tmp_path / "m.png")
    out = composite(tmp_path, tmp_path / "s.png", solid(20, 20, (200, 100, 50)), (20, 20), mask=tmp_path / "m.png")
    assert abs(int(out[5, 5, 0]) - 100) <= 2
    assert abs(int(out[5, 5, 1]) - 50) <= 2


def test_an_all_black_mask_exits_3_and_writes_nothing(tmp_path: Path) -> None:
    solid(20, 20, RED).save(tmp_path / "s.png")
    Image.new("L", (20, 20), 0).save(tmp_path / "m.png")
    done = run("prepare", "--source", tmp_path / "s.png", "--expect", 20, 20, "--feather", 4, "--mask", tmp_path / "m.png", "--out-mask", tmp_path / "o.png")
    assert done.returncode == 3
    assert not (tmp_path / "o.png").exists()


def test_a_wrong_expected_size_exits_4(tmp_path: Path) -> None:
    solid(20, 10, RED).save(tmp_path / "s.png")
    done = run("prepare", "--source", tmp_path / "s.png", "--expect", 10, 20, "--feather", 0, "--region", json.dumps([rect(0, 0, 5, 5)]), "--out-mask", tmp_path / "o.png")
    assert done.returncode == 4


def test_bad_usage_exits_2(tmp_path: Path) -> None:
    assert run("prepare").returncode == 2


def test_a_missing_source_exits_1_with_a_short_message(tmp_path: Path) -> None:
    done = run("prepare", "--source", tmp_path / "nope.png", "--expect", 1, 1, "--feather", 0, "--region", json.dumps([rect(0, 0, 1, 1)]), "--out-mask", tmp_path / "o.png")
    assert done.returncode == 1
    assert 0 < len(done.stderr) < 500


def test_a_jpeg_with_orientation_6_is_composited_as_displayed(tmp_path: Path) -> None:
    stored = gradient(60, 40)  # stored 60x40, shown 40x60 after orientation 6
    exif = Image.Exif()
    exif[0x0112] = 6
    stored.save(tmp_path / "s.jpg", "JPEG", quality=95, exif=exif)
    with Image.open(tmp_path / "s.jpg") as opened:
        shown = np.asarray(ImageOps.exif_transpose(opened).convert("RGB")).copy()
    assert shown.shape[:2] == (60, 40)
    out = composite(tmp_path, tmp_path / "s.jpg", solid(40, 60, RED), (40, 60), region=[rect(0, 0, 40, 30)])
    assert out.shape[:2] == (60, 40)
    assert (out[:30] == RED).all()
    assert (out[30:] == shown[30:]).all()


@pytest.mark.parametrize("fmt", ["JPEG", "WEBP", "GIF"])
def test_outside_pixels_are_exact_for_other_formats(tmp_path: Path, fmt: str) -> None:
    path = tmp_path / f"s.{fmt.lower()}"
    gradient(64, 48).save(path, fmt)
    with Image.open(path) as opened:
        decoded = np.asarray(opened.convert("RGB")).copy()
    out = composite(tmp_path, path, solid(64, 48, RED), (64, 48), region=[rect(0, 0, 32, 48)], feather=4)
    assert (out[:, 40:] == decoded[:, 40:]).all()
    assert (out[:, :28] == RED).all()


def test_a_transparent_source_is_flattened_over_white(tmp_path: Path) -> None:
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(tmp_path / "s.png")
    out = composite(tmp_path, tmp_path / "s.png", solid(10, 10, RED), (10, 10), region=[rect(0, 0, 5, 10)])
    assert (out[:, 5:] == 255).all()


def _mflux_load(path: Path) -> np.ndarray:
    """`ImageUtil.load_image` of mflux: oriented, alpha over white, convert("RGB")."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.has_transparency_data:
            rgba = (image.convert("LA") if image.mode == "La" else image).convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return np.asarray(Image.alpha_composite(white, rgba).convert("RGB")).copy()
        return np.asarray(image.convert("RGB")).copy()


@pytest.mark.parametrize("fill", [0x8000, 0x0080, 0xFFFF, 200])
def test_a_16_bit_grayscale_source_is_decoded_exactly_like_mflux(tmp_path: Path, fill: int) -> None:
    array = np.full((10, 10), fill, dtype=np.uint16)
    array[0, 0] = 0x4000
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, "PNG")
    (tmp_path / "s.png").write_bytes(buffer.getvalue())
    out = composite(tmp_path, tmp_path / "s.png", solid(10, 10, RED), (10, 10), region=[rect(0, 0, 5, 10)])
    assert (out[:, 5:] == _mflux_load(tmp_path / "s.png")[:, 5:]).all()


def test_a_wrong_sized_edit_is_resized_to_the_source(tmp_path: Path) -> None:
    gradient(80, 60).save(tmp_path / "s.png")
    out = composite(tmp_path, tmp_path / "s.png", solid(40, 30, RED), (80, 60), region=[rect(0, 0, 80, 60)])
    assert out.shape[:2] == (60, 80)
    assert (out == RED).all()


def test_a_1024x768_composite_with_feather_is_fast(tmp_path: Path) -> None:
    gradient(1024, 768).save(tmp_path / "s.png")
    started = time.monotonic()
    composite(tmp_path, tmp_path / "s.png", solid(1024, 768, RED), (1024, 768), region=[rect(100, 100, 600, 400, "ellipse")], feather=64)
    assert time.monotonic() - started < 3  # two process starts and the work; the plan's budget is 1 s for the work itself


def test_a_tiny_selection_with_a_big_feather_that_blurs_to_nothing_exits_3(tmp_path: Path) -> None:
    solid(100, 100, RED).save(tmp_path / "s.png")
    done = run("prepare", "--source", tmp_path / "s.png", "--expect", 100, 100, "--feather", 64, "--region", json.dumps([rect(50, 50, 1, 1)]), "--out-mask", tmp_path / "o.png")
    assert done.returncode == 3, done.stderr
    assert "selects nothing" in done.stderr
    assert not (tmp_path / "o.png").exists()


def test_an_image_over_the_pixel_cap_exits_5_fast(tmp_path: Path) -> None:
    bomb = Image.new("1", (12000, 12000), 0)
    bomb.save(tmp_path / "bomb.png")
    assert (tmp_path / "bomb.png").stat().st_size < 1_000_000
    started = time.monotonic()
    done = run("prepare", "--source", tmp_path / "bomb.png", "--expect", 12000, 12000, "--feather", 0, "--region", json.dumps([rect(0, 0, 5, 5)]), "--out-mask", tmp_path / "o.png")
    assert done.returncode == 5, done.stderr
    assert "50 megapixels" in done.stderr
    assert time.monotonic() - started < 5


def test_a_bomb_mask_is_refused_too(tmp_path: Path) -> None:
    solid(20, 20, RED).save(tmp_path / "s.png")
    Image.new("1", (12000, 12000), 0).save(tmp_path / "m.png")
    done = run("prepare", "--source", tmp_path / "s.png", "--expect", 20, 20, "--feather", 0, "--mask", tmp_path / "m.png", "--out-mask", tmp_path / "o.png")
    assert done.returncode == 5, done.stderr


@pytest.mark.parametrize("fmt", ["PNG", "WEBP"])
def test_a_png_or_webp_with_orientation_6_is_composited_as_displayed(tmp_path: Path, fmt: str) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6
    path = tmp_path / f"s.{fmt.lower()}"
    gradient(60, 40).save(path, fmt, exif=exif, **({"lossless": True} if fmt == "WEBP" else {}))
    shown = _mflux_load(path)
    assert shown.shape[:2] == (60, 40)
    out = composite(tmp_path, path, solid(40, 60, RED), (40, 60), region=[rect(0, 0, 40, 30)])
    assert (out[:30] == RED).all()
    assert (out[30:] == shown[30:]).all()


def test_exit_4_names_both_sizes(tmp_path: Path) -> None:
    solid(20, 10, RED).save(tmp_path / "s.png")
    done = run("prepare", "--source", tmp_path / "s.png", "--expect", 10, 20, "--feather", 0, "--region", json.dumps([rect(0, 0, 5, 5)]), "--out-mask", tmp_path / "o.png")
    assert done.stderr.strip() == "source is 20x10, expected 10x20"
