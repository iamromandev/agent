"""Region shapes for a masked edit_image: validation and the dialog line. UI-free, pure Python."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MAX_SHAPES, MAX_FEATHER, DEFAULT_FEATHER = 8, 64, 8
_LIMIT = 1_000_000  # pixels: far above any source, keeps the arguments small and the arithmetic safe
_KEYS = {"shape", "x", "y", "w", "h"}


@dataclass(frozen=True)
class Shape:
    kind: Literal["rect", "ellipse"]
    x: int
    y: int
    w: int
    h: int


def _int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and abs(value) <= _LIMIT


def parse_region(value: object) -> list[Shape] | str:
    """The shapes of a `region` argument, or the message saying what is wrong with it."""
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_SHAPES:
        return f"region must be a list of 1-{MAX_SHAPES} shapes"
    shapes: list[Shape] = []
    for number, item in enumerate(value, 1):
        if not isinstance(item, dict) or set(item) != _KEYS:
            return f"region shape {number} must have exactly shape, x, y, w and h"
        kind = item["shape"]
        if kind not in ("rect", "ellipse"):
            return f"region shape {number}: shape must be rect or ellipse"
        if not all(_int(item[key]) for key in ("x", "y", "w", "h")):
            return f"region shape {number}: x, y, w and h must be integers"
        if item["w"] < 1 or item["h"] < 1:
            return f"region shape {number}: w and h must be at least 1"
        shapes.append(Shape(kind, item["x"], item["y"], item["w"], item["h"]))
    return shapes


def check_region(shapes: list[Shape], width: int, height: int) -> str | None:
    """None when every shape is at least partly inside a width x height source, else the message."""
    for number, s in enumerate(shapes, 1):
        if s.x + s.w <= 0 or s.y + s.h <= 0 or s.x >= width or s.y >= height:
            return f"region shape {number} lies outside the source ({width}\u00d7{height})"
    return None


def _count(shapes: list[Shape], kind: str, noun: str) -> str | None:
    n = sum(1 for s in shapes if s.kind == kind)
    if n == 0:
        return None
    return f"{n} {noun}" + ("" if n == 1 else "s")


def describe(shapes: list[Shape] | None, mask_label: str | None, feather: int) -> str:
    """The dialog line for the selection."""
    if shapes is not None:
        parts = [p for p in (_count(shapes, "rect", "rect"), _count(shapes, "ellipse", "ellipse")) if p]
        return f"region: {', '.join(parts)}, feather {feather} px"
    return f"mask: {mask_label}, feather {feather} px"
