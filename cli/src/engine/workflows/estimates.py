"""Rough seconds a tool call takes, for the plan's total. A guess, shown as one."""

from __future__ import annotations

from typing import Any

FIXED = {"speak": 10, "assemble_video": 60, "upscale_image": 60, "edit_image": 300}
OTHER = 30


def _number(args: dict[str, Any], key: str, default: int) -> int:
    value = args.get(key)
    return value if type(value) is int and value > 0 else default


def estimate(tool: str, args: dict[str, Any]) -> int:
    """Seconds the call is expected to take: image generation scales with the pixels and the steps, the rest is fixed."""
    if tool == "generate_image":
        pixels = _number(args, "width", 1024) * _number(args, "height", 1024)
        return round(pixels * _number(args, "steps", 9) * 4e-7)
    return FIXED.get(tool, OTHER)
