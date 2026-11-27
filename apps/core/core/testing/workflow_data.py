"""Workflow data shared by the runner tests here and the app's /workflow tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.engine.provider.base import Done, TextDelta
from core.engine.workflows.loader import parse_workflow
from core.engine.workflows.model import Workflow

SCHEMA = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {"image_prompt": {"type": "string"}, "narration": {"type": "string"}},
                "required": ["image_prompt", "narration"],
            },
        }
    },
    "required": ["scenes"],
}


def reel_data(pic_path: str = "{folder}/s{index}.png", folder: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "description": "make a reel",
        "steps": [
            {"id": "story", "kind": "model", "prompt": "Write scenes about {input}", "schema": SCHEMA},
            {
                "id": "pic",
                "kind": "each",
                "over": "story.scenes",
                "tool": "generate_image",
                "args": {"prompt": "{item.image_prompt}", "path": pic_path},
            },
            {
                "id": "voice",
                "kind": "each",
                "over": "story.scenes",
                "tool": "speak",
                "args": {"prompt": "{item.narration}", "path": "{folder}/v{index}.wav"},
            },
            {
                "id": "video",
                "kind": "tool",
                "tool": "assemble_video",
                "args": {
                    "path": "{folder}/reel.mp4",
                    "scenes": {"$map": "story.scenes", "to": {"image": "{folder}/s{index}.png"}},
                },
            },
        ],
    }
    if folder is not None:
        data["folder"] = folder
    return data


def workflow(data: dict[str, Any] | None = None, tmp: Path = Path("/w/reel.json")) -> Workflow:
    return parse_workflow("reel", "project", tmp, json.dumps(data or reel_data()).encode())


def scenes(n: int) -> str:
    return json.dumps({"scenes": [{"image_prompt": f"p{i}", "narration": f"n{i}"} for i in range(1, n + 1)]})


def answer(text: str) -> list[Any]:
    return [TextDelta(text), Done(20, 10)]


def scenes_with(narrations: list[str]) -> str:
    return json.dumps(
        {"scenes": [{"image_prompt": f"p{i}", "narration": n} for i, n in enumerate(narrations, 1)]}
    )
