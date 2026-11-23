"""Frame and caption helper for assemble_video. A script, run as `python -I media_helper.py ...`; never imported.

It runs in its own process (sandboxed, filtered environment) because it decodes untrusted images with Pillow, which the
agent never imports, and it is the only place caption text is drawn: ffmpeg reads only the PNGs written here, so no
text ever reaches a filter string. Two steps:

    frame   --in IMAGE --out frame.png --width W --height H
    caption --text-file cap.txt --out cap.png

`frame` applies the EXIF orientation, composites transparency over black, takes the first frame of a GIF and
cover-scales and centre-crops to exactly W x H. `caption` writes a transparent 1080x1920 PNG with the already wrapped
text in white with a dark outline, centred in the lower third.

Exit codes: 0 ok, 1 other failure (one short stderr line), 2 usage, 3 an image or text file that cannot be read, 5 an
image over the pixel cap (50 megapixels).
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import warnings
from typing import Any


def _load(name: str) -> Any:
    return importlib.import_module(name)  # by a variable name: nothing here needs Pillow to type-check


Image = _load("PIL.Image")
ImageOps = _load("PIL.ImageOps")
ImageDraw = _load("PIL.ImageDraw")
ImageFont = _load("PIL.ImageFont")

EXIT_UNREADABLE, EXIT_TOO_LARGE = 3, 5
MAX_PIXELS = 50_000_000  # Pillow only warns below twice its own cap; a decode of more is a memory bomb
FORMATS = ["PNG", "JPEG", "GIF", "WEBP"]
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
warnings.simplefilter("error", Image.DecompressionBombWarning)
FRAME = (1080, 1920)
FONTS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"]
FONT_SIZE, MIN_FONT_SIZE = 64, 24
STROKE = 4
LINE_GAP = 12
MARGIN = 40  # the text block stays this far from the sides
MAX_TEXT_BYTES = 4096


class Exit(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _open(path):
    try:
        image = Image.open(path, formats=FORMATS)
        image.load()  # first frame of a GIF
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise Exit(EXIT_TOO_LARGE, "image too large for a video (limit 50 megapixels)") from exc
    except (OSError, SyntaxError, ValueError) as exc:  # includes UnidentifiedImageError and a missing file
        raise Exit(EXIT_UNREADABLE, f"cannot decode the image: {type(exc).__name__}") from exc
    return ImageOps.exif_transpose(image)


def frame(args):
    image = _open(args.input)
    if image.has_transparency_data:
        rgba = image.convert("RGBA")
        black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        image = Image.alpha_composite(black, rgba)
    image = ImageOps.fit(image.convert("RGB"), (args.width, args.height), Image.LANCZOS)
    image.save(args.out, "PNG")


def _font(size):
    for path in FONTS:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def caption(args):
    try:
        with open(args.text_file, "rb") as handle:
            text = handle.read(MAX_TEXT_BYTES + 1).decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise Exit(EXIT_UNREADABLE, f"cannot read the caption: {type(exc).__name__}") from exc
    text = text.strip("\n")
    canvas = Image.new("RGBA", FRAME, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    size = FONT_SIZE
    while True:
        font = _font(size)
        left, top, right, bottom = draw.multiline_textbbox(
            (0, 0), text, font=font, stroke_width=STROKE, spacing=LINE_GAP, align="center"
        )
        if right - left <= FRAME[0] - 2 * MARGIN or size <= MIN_FONT_SIZE:
            break
        size -= 4
    third = FRAME[1] // 3
    x = (FRAME[0] - (right - left)) / 2 - left
    y = FRAME[1] - third + (third - (bottom - top)) / 2 - top
    draw.multiline_text(
        (x, y), text, font=font, fill=(255, 255, 255, 255), stroke_width=STROKE, stroke_fill=(0, 0, 0, 255),
        spacing=LINE_GAP, align="center",
    )  # fmt: skip
    canvas.save(args.out, "PNG")


def parser():
    p = argparse.ArgumentParser(prog="media_helper")
    steps = p.add_subparsers(dest="step", required=True)
    f = steps.add_parser("frame")
    f.add_argument("--in", dest="input", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--width", type=int, required=True)
    f.add_argument("--height", type=int, required=True)
    c = steps.add_parser("caption")
    c.add_argument("--text-file", required=True)
    c.add_argument("--out", required=True)
    return p


def main(argv):
    args = parser().parse_args(argv)  # a usage error exits 2
    try:
        frame(args) if args.step == "frame" else caption(args)
    except Exit as exc:
        print(str(exc), file=sys.stderr)
        return exc.code
    except Exception as exc:  # untrusted input: any decode failure is one short line
        print(f"{type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
