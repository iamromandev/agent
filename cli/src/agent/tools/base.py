"""Tool contract: what a tool is, what it returns, and how its arguments are checked."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from src.agent.provider.base import ToolSpec


@dataclass(frozen=True)
class ToolContext:
    workspace: Path  # resolved absolute path
    bash_timeout: float


@dataclass(frozen=True)
class ToolResult:
    output: str  # what the model sees
    is_error: bool = False
    summary: str = ""  # short transcript text: "31 lines", "exit 0", "refused: outside workspace"


@dataclass(frozen=True)
class Preview:
    kind: Literal["diff", "content", "command"]
    title: str  # "edit src/main.py", "bash uv run pytest -q"
    body: str  # unified diff / full new content / the command


@dataclass(frozen=True)
class Prepared:
    preview: Preview | None
    execute: Callable[[], Awaitable[ToolResult]]


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]
    mutating: ClassVar[bool] = False  # write/edit: hidden + refused in plan mode
    needs_approval: ClassVar[bool] = False  # write/edit/bash

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)

    def label(self, args: dict[str, Any]) -> str:
        """Short transcript label: the tool name plus its primary (first) argument."""
        properties = self.parameters.get("properties", {})
        primary = next(iter(properties), None)
        value = args.get(primary) if primary is not None else None
        return f"{self.name} {value}" if value not in (None, "") else self.name

    @abstractmethod
    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        """Validate and plan a call.

        A ToolResult means the call finished without approval (a read-only result, a
        refusal or an error). A Prepared carries the preview and the deferred execute.
        """


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "boolean": (bool,),
}
_ARTICLES = {"string": "a string", "integer": "an integer", "boolean": "a boolean"}


def check_args(parameters: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Return a short error for missing keys or wrong types; unknown keys are ignored."""
    if "__invalid__" in args:
        return "arguments were not valid JSON"
    for key in parameters.get("required", []):
        if key not in args:
            return f'missing "{key}"'
    properties = parameters.get("properties", {})
    for key, value in args.items():
        expected = properties.get(key, {}).get("type")
        types = _JSON_TYPES.get(expected) if isinstance(expected, str) else None
        if types is None:
            continue
        # bool is a subclass of int, but True is not a valid integer argument.
        if not isinstance(value, types) or (expected == "integer" and isinstance(value, bool)):
            return f'"{key}" must be {_ARTICLES[expected]}'
    return None
