"""Filesystem tools: read, write and edit."""

from __future__ import annotations

import asyncio
import itertools
from pathlib import Path
from typing import Any, ClassVar

from core.engine.guardrails.redact import PLACEHOLDER, adds_placeholder
from core.engine.guardrails.setup import guard_of
from core.engine.provider.base import Image
from core.engine.provider.images import ImageError, describe, read_image_fd
from core.engine.provider.models import shown_name

from .base import Prepared, Preview, Tool, ToolContext, ToolResult
from .paths import InvalidPath, inside_workspace, invalid_path_result, resolve_path
from .protected import NotRegular, ProtectedFile, as_text, open_checked, refuse_protected
from .text import is_binary, unified_diff

READ_MAX_LINES = 2000


class ReadTool(Tool):
    name: ClassVar[str] = "read"
    description: ClassVar[str] = (
        "Read a text file and return its lines with line numbers, or show an image (PNG, JPEG, GIF, WebP) to a model with vision. "
        "Use offset (1-based first line) and limit (max 2000) for large files."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative to the workspace or absolute."},
            "offset": {"type": "integer", "description": "First line to read, 1-based. Default 1."},
            "limit": {"type": "integer", "description": "Number of lines to read. Default and maximum 2000."},
        },
        "required": ["path"],
    }

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        raw: str = args["path"]
        offset = max(1, args.get("offset", 1))
        limit = max(1, min(args.get("limit", READ_MAX_LINES), READ_MAX_LINES))
        try:
            path = resolve_path(ctx, raw)
        except InvalidPath as exc:
            return invalid_path_result(exc)
        result = await asyncio.to_thread(_read, path, raw, offset, limit, ctx)
        if not isinstance(result, Image):
            return result
        if ctx.has_vision is None or not await ctx.has_vision():  # asked only now: a text read never waits on it
            message = f"{raw} is an image and the model '{shown_name(ctx.model)}' has no vision"
            return ToolResult(message, is_error=True, summary="no vision")
        described = describe(result)
        return ToolResult(f"image {raw} ({described})", summary=described, images=(result,))


def _read(path: Path, raw: str, offset: int, limit: int, ctx: ToolContext) -> ToolResult | Image:
    if path.is_dir():
        return ToolResult(f"{raw} is a directory", is_error=True, summary="directory")
    if not path.exists():
        return ToolResult(f"No such file: {raw}", is_error=True, summary="not found")
    if not path.is_file():  # a FIFO or device would block the read forever
        return ToolResult(f"{raw} is not a regular file", is_error=True, summary="not a regular file")
    try:
        # One open, and the open file is what gets checked: the path check in the loop is not enough on its own.
        raw_handle = open_checked(path, ctx)
        try:
            try:
                image = read_image_fd(raw_handle.fileno(), raw)
            except ImageError as exc:
                raw_handle.close()
                return ToolResult(f"{raw}: {exc.reason}", is_error=True, summary="image too large")
            if image is not None:
                raw_handle.close()
                return image
            handle, binary = as_text(raw_handle)  # the same open file: no second open to race
        except BaseException:
            raw_handle.close()
            raise
        with handle:
            if binary:
                return ToolResult(f"{raw} is binary or not UTF-8", is_error=True, summary="binary")
            window = itertools.islice(handle, offset - 1, offset - 1 + limit)
            lines = [f"{n:>6}\t{line.rstrip('\r\n')}" for n, line in enumerate(window, start=offset)]
            if not lines and offset > 1:
                handle.seek(0)
                total = sum(1 for _ in handle)
                message = f"offset {offset} is past the end of the file ({total} lines)"
                return ToolResult(message, summary="0 lines")
    except ProtectedFile:
        return refuse_protected()
    except NotRegular:
        return ToolResult(f"{raw} is not a regular file", is_error=True, summary="not a regular file")
    except UnicodeDecodeError:  # invalid UTF-8 past the first 8 KB
        return ToolResult(f"{raw} is binary or not UTF-8", is_error=True, summary="binary")
    except OSError as exc:
        return ToolResult(f"Cannot read {raw}: {exc.strerror or exc}", is_error=True, summary="unreadable")
    return ToolResult("\n".join(lines), summary=f"{len(lines)} lines")


REFUSED_OUTSIDE = "refused: outside workspace"


def _refuse_outside() -> ToolResult:
    return ToolResult(REFUSED_OUTSIDE, is_error=True, summary=REFUSED_OUTSIDE)


# Confinement caveat (TOCTOU): write and edit resolve the path once, in `prepare`, and
# `execute` writes to that resolved path after the user approves. A symlink swapped in
# between prepare and execute is not defended against. That is acceptable for a
# single-user local tool whose model cannot race the user; bash can do anything anyway.


class WriteTool(Tool):
    name: ClassVar[str] = "write"
    description: ClassVar[str] = (
        "Create a file or replace its whole content. Creates missing folders. "
        "Only paths inside the workspace are allowed. Prefer edit for changes to existing files."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path inside the workspace."},
            "content": {"type": "string", "description": "The full new content of the file."},
        },
        "required": ["path", "content"],
    }
    mutating: ClassVar[bool] = True
    needs_approval: ClassVar[bool] = True

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        raw: str = args["path"]
        content: str = args["content"]
        try:
            path = resolve_path(ctx, raw)
        except InvalidPath as exc:
            return invalid_path_result(exc)
        if not inside_workspace(path, ctx):
            return _refuse_outside()
        if guard_of(ctx).paths.is_protected(path):
            return refuse_protected()
        if path.is_dir():
            return ToolResult(f"{raw} is a directory", is_error=True, summary="directory")
        if path.exists() and not path.is_file():
            return _not_regular(raw)
        try:
            data = content.encode("utf-8")  # before any mkdir: a lone surrogate must leave no trace
        except UnicodeEncodeError:
            return ToolResult("content is not valid UTF-8 text", is_error=True, summary="invalid text")
        title = f"write {raw}"
        old = await asyncio.to_thread(_read_lenient, path) if path.exists() else None
        if adds_placeholder(old or "", content):
            return _refuse_placeholder("content")
        if old is not None:
            preview = Preview("diff", title, unified_diff(old, content, raw))
        else:
            preview = Preview("content", title, content)

        async def execute() -> ToolResult:
            def write() -> None:
                if ctx.journal is not None:
                    ctx.journal.before(path)
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                finally:
                    if ctx.journal is not None:
                        ctx.journal.after(path)

            try:
                await asyncio.to_thread(write)
            except OSError as exc:
                return ToolResult(f"Cannot write {raw}: {exc.strerror or exc}", is_error=True, summary="write failed")
            lines = len(content.splitlines())
            return ToolResult(f"wrote {lines} lines to {raw}", summary=f"{lines} lines")

        return Prepared(preview, execute, target=path)


def _refuse_placeholder(argument: str) -> ToolResult:
    return ToolResult(
        f"{argument} would add the text '{PLACEHOLDER}...]': that is a placeholder the agent puts where a tool result "
        "held a secret, not the secret's value. Writing it would replace the real value with the placeholder. Leave "
        "that part out, or take the value from where it is kept (an environment variable, a secrets manager) in the "
        "code. Placeholders that the file already has can stay.",
        is_error=True,
        summary="redaction placeholder",
    )


def _not_regular(raw: str) -> ToolResult:
    return ToolResult(f"{raw} is not a regular file", is_error=True, summary="not a regular file")


def _read_lenient(path: Path) -> str:
    """Existing content for a diff preview; undecodable bytes are replaced, not fatal."""
    return path.read_bytes().decode("utf-8", errors="replace")


class EditTool(Tool):
    name: ClassVar[str] = "edit"
    description: ClassVar[str] = (
        "Replace old_string with new_string in an existing file. old_string must match exactly "
        "once (include surrounding lines for context) unless replace_all is true. "
        "Read the file first. Only paths inside the workspace are allowed."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path inside the workspace."},
            "old_string": {"type": "string", "description": "Exact text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence. Default false."},
        },
        "required": ["path", "old_string", "new_string"],
    }
    mutating: ClassVar[bool] = True
    needs_approval: ClassVar[bool] = True

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        raw: str = args["path"]
        old_string: str = args["old_string"]
        new_string: str = args["new_string"]
        replace_all: bool = args.get("replace_all", False)
        try:
            path = resolve_path(ctx, raw)
        except InvalidPath as exc:
            return invalid_path_result(exc)
        if not inside_workspace(path, ctx):
            return _refuse_outside()
        if guard_of(ctx).paths.is_protected(path):
            return refuse_protected()
        if path.is_dir():
            return ToolResult(f"{raw} is a directory", is_error=True, summary="directory")
        if not path.exists():
            return ToolResult(f"No such file: {raw}", is_error=True, summary="not found")
        if not path.is_file():
            return _not_regular(raw)
        try:
            new_string.encode("utf-8")
        except UnicodeEncodeError:
            return ToolResult("new_string is not valid UTF-8 text", is_error=True, summary="invalid text")
        if adds_placeholder(old_string, new_string):
            return _refuse_placeholder("new_string")
        if not old_string:
            return ToolResult("old_string must not be empty", is_error=True, summary="empty old_string")
        try:
            original = await asyncio.to_thread(_read_strict, path)
        except UnicodeDecodeError:
            return ToolResult(f"{raw} is binary or not UTF-8", is_error=True, summary="binary")
        except OSError as exc:
            return ToolResult(f"Cannot read {raw}: {exc.strerror or exc}", is_error=True, summary="unreadable")

        count = original.count(old_string)
        if count == 0:
            return ToolResult(f"old_string not found in {raw}", is_error=True, summary="not found")
        if count > 1 and not replace_all:
            return ToolResult(
                f"old_string occurs {count} times in {raw}; add context or set replace_all",
                is_error=True,
                summary="ambiguous",
            )
        replaced = count if replace_all else 1
        updated = original.replace(old_string, new_string, -1 if replace_all else 1)
        preview = Preview("diff", f"edit {raw}", unified_diff(original, updated, raw))

        async def execute() -> ToolResult:
            def apply() -> bool:
                if _read_strict(path) != original:
                    return False
                if ctx.journal is not None:
                    ctx.journal.before(path)
                try:
                    path.write_bytes(updated.encode("utf-8"))
                finally:
                    if ctx.journal is not None:
                        ctx.journal.after(path)
                return True

            try:
                applied = await asyncio.to_thread(apply)
            except (OSError, UnicodeDecodeError) as exc:
                return ToolResult(f"Cannot edit {raw}: {exc}", is_error=True, summary="edit failed")
            if not applied:
                return ToolResult(
                    f"{raw} changed since the edit was prepared", is_error=True, summary="file changed"
                )
            return ToolResult(f"replaced {replaced} occurrence(s) in {raw}", summary=f"{replaced} replaced")

        return Prepared(preview, execute, target=path)


def _read_strict(path: Path) -> str:
    """Decode without newline translation, so CRLF files are written back unchanged."""
    if is_binary(path):
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "binary file")
    return path.read_bytes().decode("utf-8")
