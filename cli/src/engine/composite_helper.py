"""Compositing helper for a masked edit_image. A script, run as `python -I composite_helper.py ...`; never imported.

It runs in its own process (sandboxed, filtered environment) because it decodes untrusted images with Pillow, which the
agent never imports. Two steps:

    prepare --source S --expect W H --feather F (--region JSON | --mask M) --out-mask mask.png
    blend   --source S --edit E --mask mask.png --out out.png

Exit codes: 0 ok, 1 other failure (one short stderr line), 2 usage, 3 the mask selects nothing, 4 the source's
(oriented) size is not the expected one, 5 an image over the pixel cap (50 megapixels).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import warnings
from typing import Any


def _load(name: str) -> Any:
    return importlib.import_module(name)  # by a variable name: nothing here needs Pillow to type-check


Image = _load("PIL.Image")
ImageOps = _load("PIL.ImageOps")
ImageDraw = _load("PIL.ImageDraw")
ImageFilter = _load("PIL.ImageFilter")
np = _load("numpy")

EXIT_NOTHING, EXIT_SIZE, EXIT_TOO_LARGE = 3, 4, 5
MAX_PIXELS = 50_000_000  # Pillow only warns below twice its own cap; a decode of more is a memory bomb
FORMATS = ["PNG", "JPEG", "GIF", "WEBP"]
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
warnings.simplefilter("error", Image.DecompressionBombWarning)
ROWS = 256  # rows per blend chunk: bounds the memory of the 16-bit arithmetic
SIGMA_PER_FEATHER = 3  # the blur reaches 3 sigma, so its tail is gone at `feather` pixels


class Exit(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _open(path):
    try:
        image = Image.open(path, formats=FORMATS)
        image.load()  # first frame of a GIF
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise Exit(EXIT_TOO_LARGE, "image too large to composite (limit 50 megapixels)") from exc
    return ImageOps.exif_transpose(image)  # the mode is left alone, as mflux does


def load_source(path):
    """RGB as mflux reads it: EXIF orientation applied, transparency composited over white."""
    image = _open(path)
    if image.has_transparency_data:
        rgba = (image.convert("LA") if image.mode == "La" else image).convert("RGBA")  # premultiplied La, as mflux
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")


def _dilate(binary, radius):
    """The square neighbourhood of `radius` around every True pixel, by running sums (linear time)."""
    size = 2 * radius + 1
    for axis in (0, 1):
        pad = [(radius, radius) if a == axis else (0, 0) for a in (0, 1)]
        total = np.cumsum(np.pad(binary.astype(np.int32), pad), axis=axis, dtype=np.int32)
        zero_shape = list(total.shape)
        zero_shape[axis] = 1
        total = np.concatenate([np.zeros(zero_shape, dtype=np.int32), total], axis=axis)
        n = binary.shape[axis]
        upper = np.take(total, range(size, size + n), axis=axis)
        lower = np.take(total, range(0, n), axis=axis)
        binary = (upper - lower) > 0
    return binary


def build_mask(size, feather, region, mask_path):
    if region is not None:
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        for shape in region:
            box = [shape["x"], shape["y"], shape["x"] + shape["w"] - 1, shape["y"] + shape["h"] - 1]
            if shape["shape"] == "ellipse":
                draw.ellipse(box, fill=255)
            else:
                draw.rectangle(box, fill=255)
    else:
        mask = _open(mask_path).convert("L")  # luminance; alpha is dropped, not composited
        if mask.size != size:
            mask = mask.resize(size, Image.BILINEAR)
    if mask.getextrema()[1] == 0:
        raise Exit(EXIT_NOTHING, "the mask selects nothing")
    if feather > 0:
        blurred = np.asarray(mask.filter(ImageFilter.GaussianBlur(radius=feather / SIGMA_PER_FEATHER)))
        reach = _dilate(np.asarray(mask) > 0, feather)
        mask = Image.fromarray(np.where(reach, blurred, 0).astype(np.uint8), "L")
        if mask.getextrema()[1] == 0:  # a tiny shape blurred away
            raise Exit(EXIT_NOTHING, "the mask selects nothing")
    return mask


def prepare(args):
    source = load_source(args.source)
    if source.size != (args.expect[0], args.expect[1]):
        raise Exit(EXIT_SIZE, f"source is {source.size[0]}x{source.size[1]}, expected {args.expect[0]}x{args.expect[1]}")
    region = json.loads(args.region) if args.region is not None else None
    build_mask(source.size, args.feather, region, args.mask).save(args.out_mask, "PNG")


def blend(args):
    source = load_source(args.source)
    edit = load_source(args.edit)
    mask = _open(args.mask).convert("L")
    if mask.size != source.size:
        raise Exit(1, "mask and source differ in size")
    if edit.size != source.size:
        edit = edit.resize(source.size, Image.LANCZOS)
    src, new, weight = np.asarray(source), np.asarray(edit), np.asarray(mask)
    out = np.empty_like(src)
    for top in range(0, src.shape[0], ROWS):
        rows = slice(top, top + ROWS)
        m = weight[rows, :, None].astype(np.uint16)
        mixed = (src[rows].astype(np.uint16) * (255 - m) + new[rows].astype(np.uint16) * m + 127) // 255
        out[rows] = mixed.astype(np.uint8)
    Image.fromarray(out, "RGB").save(args.out, "PNG")


def parser():
    top = argparse.ArgumentParser(prog="composite_helper")
    sub = top.add_subparsers(dest="step", required=True)
    one = sub.add_parser("prepare")
    one.add_argument("--source", required=True)
    one.add_argument("--expect", required=True, nargs=2, type=int)
    one.add_argument("--feather", required=True, type=int)
    pick = one.add_mutually_exclusive_group(required=True)
    pick.add_argument("--region")
    pick.add_argument("--mask")
    one.add_argument("--out-mask", required=True)
    two = sub.add_parser("blend")
    two.add_argument("--source", required=True)
    two.add_argument("--edit", required=True)
    two.add_argument("--mask", required=True)
    two.add_argument("--out", required=True)
    return top


def main(argv):
    args = parser().parse_args(argv)  # a usage error exits 2
    try:
        prepare(args) if args.step == "prepare" else blend(args)
    except Exit as exc:
        print(str(exc), file=sys.stderr)
        return exc.code
    except Exception as exc:  # untrusted input: any decode failure is one short line
        print(f"{type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
