"""Seconds a workflow's calls are expected to take, for the plan's total: the median of what this machine recorded,
else a default measured on an M1."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from typing import Any

from core.engine.workflows.model import PlannedCall
from core.engine.workflows.timings import TimingStore

DEFAULT_RATE = {
    "generate_image": 4.24e-5,  # seconds per pixel x step: 225 s for 576x1024 at 9 steps
    "edit_image": 7.0e-5,  # 294 s for 1024x1024 at 4 steps
    "speak": 4.0,  # per call
    "assemble_video": 6.5,  # per scene: about 19 s for 3 scenes
    "upscale_image": 6.1e-5,  # seconds per output pixel: 257 s for 1024 -> 2048 (512 -> 1024 took 60 s)
}
OTHER_RATE = 30.0
GUESS = "a rough guess"


def _number(args: dict[str, Any], key: str, default: int) -> int:
    value = args.get(key)
    return value if type(value) is int and value > 0 else default


Pixels = Callable[[dict[str, Any]], int | None]
ASSUMED_SOURCE = 1024  # side of the source assumed for an `upscale_image` call whose source is not known


def unit(tool: str, args: dict[str, Any], pixels: Pixels | None = None) -> tuple[str, float]:
    """`(key, amount)`: a call takes about `rate(key) * amount` seconds. `pixels(args)` gives the output pixels of an
    `upscale_image` call when they are known (else a 1024x1024 source is assumed)."""
    if tool == "upscale_image":
        known = pixels(args) if pixels is not None else None
        if known is not None and known > 0:
            return tool, float(known)
        scale = args.get("scale")
        factor = scale if type(scale) is int and scale in (2, 3, 4) else 2
        return tool, float((ASSUMED_SOURCE * factor) ** 2)
    if tool in ("generate_image", "edit_image"):
        steps = _number(args, "steps", 9 if tool == "generate_image" else 4)
        return tool, float(_number(args, "width", 1024) * _number(args, "height", 1024) * steps)
    if tool == "assemble_video":
        scenes = args.get("scenes")
        return tool, float(len(scenes) if isinstance(scenes, list) and scenes else 1)
    return tool, 1.0


def estimate_plan(
    calls: Sequence[PlannedCall], timings: TimingStore, pixels: Pixels | None = None
) -> tuple[int, str]:
    """`(seconds, source)`: the source says whether the rates were learned, partly learned or a rough guess."""
    medians: dict[str, tuple[float, int] | None] = {}
    total = 0.0
    learned = 0
    for call in calls:
        key, amount = unit(call.tool, call.args, pixels)
        if key not in medians:
            values = timings.values(key)
            medians[key] = (statistics.median(values), len(values)) if values else None
        found = medians[key]
        if found is not None:
            learned += 1
            total += found[0] * amount
        else:
            total += DEFAULT_RATE.get(key, OTHER_RATE) * amount
    seconds = round(total)
    used = sum(found[1] for found in medians.values() if found is not None)
    if calls and learned == len(calls):
        return seconds, f"learned from {used} calls"
    if learned:
        return seconds, f"partly learned ({learned} of {len(calls)} calls)"
    return seconds, GUESS
