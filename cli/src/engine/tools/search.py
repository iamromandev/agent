"""Search tools: glob (file names) and grep (file contents)."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from .base import Prepared, Tool, ToolContext, ToolResult
from .listing import list_files
from .paths import InvalidPath, inside_workspace, invalid_path_result, resolve_path
from .text import is_binary

MAX_RESULTS = 200


def _display(path: Path, ctx: ToolContext) -> str:
    """Workspace-relative when inside the workspace, else absolute."""
    if inside_workspace(path, ctx):
        return path.relative_to(ctx.workspace.resolve()).as_posix()
    return str(path)


def _capped(lines: list[str], total: int) -> str:
    out = lines[:MAX_RESULTS]
    if total > MAX_RESULTS:
        out.append(f"... {total - MAX_RESULTS} more")
    return "\n".join(out)


class GlobTool(Tool):
    name: ClassVar[str] = "glob"
    description: ClassVar[str] = (
        "List files matching a glob pattern such as **/*.py. Patterns use **, *, ? and [...] "
        "but not brace expansion like *.{py,js}: call the tool once per extension instead. "
        "Respects .gitignore in git repositories."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. src/**/*.py."},
            "path": {"type": "string", "description": "Directory to search. Default: the workspace."},
        },
        "required": ["pattern"],
    }

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        pattern: str = args["pattern"]
        raw_base: str = args.get("path") or "."
        try:
            base = resolve_path(ctx, raw_base)
        except InvalidPath as exc:
            return invalid_path_result(exc)
        if not base.is_dir():
            return ToolResult(f"No such directory: {raw_base}", is_error=True, summary="not found")
        return await asyncio.to_thread(_glob, base, pattern, ctx)


def _glob(base: Path, pattern: str, ctx: ToolContext) -> ToolResult:
    matches = [
        _display(path, ctx)
        for path in list_files(base)
        if PurePosixPath(path.relative_to(base).as_posix()).full_match(pattern)
    ]
    if not matches:
        return ToolResult("No files matched", summary="0 files")
    return ToolResult(_capped(matches, len(matches)), summary=f"{len(matches)} files")


class GrepTool(Tool):
    name: ClassVar[str] = "grep"
    description: ClassVar[str] = (
        "Search file contents with a Python regular expression. Returns file:line: text. "
        "Respects .gitignore in git repositories and skips binary files. "
        "The include glob uses **, *, ? and [...] but not brace expansion like *.{py,js}: "
        "call the tool once per extension instead."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression (Python re syntax)."},
            "path": {"type": "string", "description": "File or directory to search. Default: the workspace."},
            "include": {"type": "string", "description": (
                    "Only search files matching this glob, e.g. *.py. Supports **, *, ? and [...]; "
                    "no brace expansion like *.{py,js}: call grep once per extension."
                ),
            },
        },
        "required": ["pattern"],
    }

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        try:
            regex = re.compile(args["pattern"])
        except re.error as exc:
            return ToolResult(f"Invalid regex: {exc}", is_error=True, summary="invalid regex")
        raw_base: str = args.get("path") or "."
        try:
            base = resolve_path(ctx, raw_base)
        except InvalidPath as exc:
            return invalid_path_result(exc)
        if not base.exists():
            return ToolResult(
                f"No such file or directory: {raw_base}", is_error=True, summary="not found"
            )
        return await asyncio.to_thread(_grep, regex, base, args.get("include"), ctx)


def _included(path: Path, base: Path, include: str | None) -> bool:
    if not include:
        return True
    rel = PurePosixPath(path.relative_to(base).as_posix())
    return rel.full_match(include) or PurePosixPath(path.name).full_match(include)


def _grep(regex: re.Pattern[str], base: Path, include: str | None, ctx: ToolContext) -> ToolResult:
    if base.is_file():
        files = [base]
        include = None
        root = base.parent
    else:
        files = list_files(base)
        root = base
    shown: list[str] = []
    total = 0
    for path in files:
        if not _included(path, root, include):
            continue
        matches = _grep_file(regex, path, _display(path, ctx))
        total += len(matches)
        if len(shown) < MAX_RESULTS:
            shown.extend(matches[: MAX_RESULTS - len(shown)])
    if total == 0:
        return ToolResult("No matches", summary="0 matches")
    return ToolResult(_capped(shown, total), summary=f"{total} matches")


def _grep_file(regex: re.Pattern[str], path: Path, shown_path: str) -> list[str]:
    try:
        if is_binary(path):
            return []
        found: list[str] = []
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                if regex.search(line):
                    found.append(f"{shown_path}:{lineno}: {line.rstrip()}")
        return found
    except (OSError, UnicodeDecodeError):  # unreadable or binary past the sniffed prefix
        return []
