"""Pixel checks for the image eval, each in its own process so that no eval process imports Pillow.

Run as `python -I pixel_compare.py <mode> ...`:

- `same <source> <output> <x0>`: every pixel from column x0 on is equal (the sizes must match);
- `changed <source> <output> <x1> <min>`: at least `min` pixels in the columns left of x1 differ;
- `varied <image>`: the image is not one flat colour.

Exit 0 when the check holds, 1 when it fails (a one-line reason on stderr), 2 on bad usage, 3 when the check could not
run (Pillow or numpy missing, an unreadable file): never 1, so a broken compare is not reported as a failed check.
Images are decoded like the compositing helper does (EXIF orientation applied, converted to RGB).
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

USAGE = "usage: pixel_compare.py same <src> <out> <x0> | changed <src> <out> <x1> <min> | varied <img>"


def _load(path: str) -> Any:
    image = importlib.import_module("PIL.Image")
    ops = importlib.import_module("PIL.ImageOps")
    numpy = importlib.import_module("numpy")
    with image.open(path) as raw:
        return numpy.asarray(ops.exif_transpose(raw).convert("RGB"))


def _differ(source: Any, out: Any, start: int, stop: int | None) -> int:
    return int((source[:, start:stop] != out[:, start:stop]).any(axis=2).sum())


def _size(array: Any) -> str:
    return f"{array.shape[1]}x{array.shape[0]}"


def check(argv: list[str]) -> int | None:
    """0 or 1 for a check that ran, None for bad usage."""
    mode, rest = (argv[0], argv[1:]) if argv else ("", [])
    if mode == "varied" and len(rest) == 1:
        image = _load(rest[0])
        if (image == image[0, 0]).all():
            print("the image is one flat colour", file=sys.stderr)
            return 1
        return 0
    numbers = rest[2:]
    if mode not in ("same", "changed") or len(rest) != 2 + (1 if mode == "same" else 2):
        return None
    if not all(value.isdigit() for value in numbers):
        return None
    source, out = _load(rest[0]), _load(rest[1])
    if source.shape != out.shape:
        print(f"size differs: {_size(source)} vs {_size(out)}", file=sys.stderr)
        return 1
    if mode == "same":
        differ = _differ(source, out, int(numbers[0]), None)
        if differ:
            print(f"{differ} pixels differ from column {numbers[0]} on", file=sys.stderr)
            return 1
        return 0
    differ = _differ(source, out, 0, int(numbers[0]))
    if differ < int(numbers[1]):
        print(f"{differ} pixels differ left of column {numbers[0]}, expected at least {numbers[1]}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str]) -> int:
    try:
        code = check(argv)
    except Exception as exc:
        print(f"pixel compare could not run: {type(exc).__name__}: {' '.join(str(exc).split())}", file=sys.stderr)
        return 3
    if code is None:
        print(USAGE, file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
