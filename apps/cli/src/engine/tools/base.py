"""Tool contract: what a tool is, what it returns, and how its arguments are checked."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from src.engine.provider.base import Image, ToolSpec

if TYPE_CHECKING:
    from src.engine.events import AgentEvent
    from src.engine.guardrails.setup import Guardrails


class FileJournal(Protocol):
    def before(self, path: Path) -> None:
        """Called in the worker thread, before the first byte of `path` changes."""
        ...

    def after(self, path: Path) -> None:
        """Called in the worker thread after the write, also when it failed."""
        ...


@dataclass(frozen=True)
class ToolContext:
    workspace: Path  # resolved absolute path
    bash_timeout: float
    journal: FileJournal | None = None  # told about each file write/edit changes
    progress: Callable[[AgentEvent], None] | None = None  # set per call by the loop; the task and generate_image tools use it
    guard: Guardrails | None = None  # None: the built-in defaults (see `guard_of`)
    protected_ok: bool = False  # set per call by the loop: the user allowed this call to read the protected file it names
    has_vision: Callable[[], Awaitable[bool]] | None = None  # set per call by the loop; asked only once `read` found an image; None: no
    model: str = ""  # set per call by the loop with `has_vision`: the model that gets the result
    pasted: tuple[Image, ...] = ()  # set per call by the loop for edit_image: the images the turn's message carried (clipboard:N)


@dataclass(frozen=True)
class ToolResult:
    output: str  # what the model sees
    is_error: bool = False
    summary: str = ""  # short transcript text: "31 lines", "exit 0", "refused: outside workspace"
    display: str | None = None  # display-only copy (may contain ANSI SGR); never sent to the model or saved
    images: tuple[Image, ...] = ()  # sent to the model with the result and saved in the session


@dataclass(frozen=True)
class Preview:
    kind: Literal["diff", "content", "command"]
    title: str  # "edit src/main.py", "bash uv run pytest -q"
    body: str  # unified diff / full new content / the command


@dataclass(frozen=True)
class Prepared:
    preview: Preview | None
    execute: Callable[[], Awaitable[ToolResult]]
    # write/edit: the file `prepare` resolved and checked (workspace, protected paths) and `execute` writes; the
    # approval step decides from it instead of resolving the argument again. None for every other tool.
    target: Path | None = None


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]
    mutating: ClassVar[bool] = False  # write/edit: hidden + refused in plan mode
    needs_approval: ClassVar[bool] = False  # write/edit/bash
    # A cancel waits for the call to finish (it changes files in a thread that cannot be stopped). None: as `mutating`.
    uncancellable: ClassVar[bool | None] = None
    grant_arg: ClassVar[str | None] = None  # "Always" covers only calls with the same value of this argument

    def runs_to_completion(self) -> bool:
        """True when a cancel must wait for a running call instead of interrupting it."""
        return self.mutating if self.uncancellable is None else self.uncancellable

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
