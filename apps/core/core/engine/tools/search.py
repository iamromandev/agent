"""Search tools: glob (file names) and grep (file contents)."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from core.engine.env import child_environment
from core.engine.guardrails.setup import guard_of

from .base import Prepared, Tool, ToolContext, ToolResult
from .listing import list_files
from .paths import InvalidPath, inside_workspace, invalid_path_result, resolve_path
from .protected import ProtectedFile, open_text_checked, refuse_protected

MAX_RESULTS = 200


def _display(path: Path, ctx: ToolContext) -> str:
    """Workspace-relative when inside the workspace, else absolute."""
    if inside_workspace(path, ctx):
        return path.relative_to(ctx.workspace.resolve()).as_posix()
    return str(path)


_WILDCARDS = frozenset("*?[")


def _skipped_note(count: int) -> str:
    return f"{count} protected files skipped"


def _with_note(text: str, skipped: int) -> str:
    return f"{text}\n{_skipped_note(skipped)}" if skipped else text


class _Guard:
    """Tells protected files from the others in a listing. A file counts by where it really is: a symlink, or a
    folder that is one, must not hide a credentials file."""

    def __init__(self, ctx: ToolContext) -> None:
        guard = guard_of(ctx)
        self._paths = guard.paths
        self.env = guard.env.filter(child_environment())  # what git runs with: no secrets from the agent's own
        self._folders: dict[Path, bool] = {}  # per-call cache of folder identity lookups
        self._parents: dict[Path, Path] = {}  # folder -> its resolved path: one lookup per folder, not per file

    def protected(self, path: Path) -> bool:
        if self._paths.is_protected(path, self._folders):
            return True
        parent = self._parents.get(path.parent)
        if parent is None:
            parent = self._parents[path.parent] = path.parent.resolve()
        real = path.resolve() if path.is_symlink() else parent / path.name
        return real != path and self._paths.is_protected(real, self._folders)


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
    # A pattern without wildcards names one file: a targeted access, which the loop asked the user about already.
    targeted = not _WILDCARDS.intersection(pattern)
    guard = _Guard(ctx)
    matches: list[str] = []
    skipped = 0
    for path in list_files(base, guard.env):
        if not PurePosixPath(path.relative_to(base).as_posix()).full_match(pattern):
            continue
        if not targeted and guard.protected(path):
            skipped += 1
            continue
        matches.append(_display(path, ctx))
    if not matches:
        return ToolResult(_with_note("No files matched", skipped), summary="0 files")
    return ToolResult(_with_note(_capped(matches, len(matches)), skipped), summary=f"{len(matches)} files")


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
    targeted = base.is_file()  # one file named by the caller: the loop asked the user about it already
    guard = _Guard(ctx)
    if targeted:
        files = [base]
        include = None
        root = base.parent
    else:
        files = list_files(base, guard.env)
        root = base
    shown: list[str] = []
    total = 0
    skipped = 0
    for path in files:
        if not _included(path, root, include):
            continue
        if not targeted and guard.protected(path):
            skipped += 1
            continue
        found = _grep_file(regex, path, _display(path, ctx), ctx)
        if found is None:  # protected by the time it was opened: a symlink changed after the listing was checked
            if targeted:
                return refuse_protected()
            skipped += 1
            continue
        matches = found
        total += len(matches)
        if len(shown) < MAX_RESULTS:
            shown.extend(matches[: MAX_RESULTS - len(shown)])
    if total == 0:
        return ToolResult(_with_note("No matches", skipped), summary="0 matches")
    return ToolResult(_with_note(_capped(shown, total), skipped), summary=f"{total} matches")


def _grep_file(regex: re.Pattern[str], path: Path, shown_path: str, ctx: ToolContext) -> list[str] | None:
    """The matching lines, or None when the file turned out to be a protected one (checked on the open file)."""
    try:
        handle, binary = open_text_checked(path, ctx)
        with handle:
            if binary:
                return []
            found: list[str] = []
            for lineno, line in enumerate(handle, start=1):
                if regex.search(line):
                    found.append(f"{shown_path}:{lineno}: {line.rstrip()}")
            return found
    except ProtectedFile:
        return None
    except (OSError, UnicodeDecodeError):  # unreadable, not regular or binary past the sniffed prefix
        return []
