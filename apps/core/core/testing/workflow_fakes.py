"""Fake tools and helpers for the workflow tests: nothing here runs a model, a process or the network."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

from core.engine.events import ToolProgress
from core.engine.tools.base import Prepared, Tool, ToolContext, ToolResult
from core.engine.tools.paths import resolve_path
from core.testing.fakes import Gate


class WriteFileTool(Tool):
    """Writes `content` (default: the prompt, else the tool name) to `path`; fails when `fail` is true."""

    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = "writes a file"
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "path": {"type": "string"},
            "fail": {"type": "boolean"},
        },
        "required": ["path"],
    }
    needs_approval: ClassVar[bool] = True
    mutating: ClassVar[bool] = True

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_on: set[str] = set()  # paths whose call fails
        self.invalid_on: dict[str, str] = {}  # path -> error text of an `invalid` refusal
        self.timeout_on: set[str] = set()  # paths whose call fails with a timeout
        self.log: list[str] | None = None  # shared by several tools: "<tool>:<file name>" per call, in call order

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        async def execute() -> ToolResult:
            self.calls.append(dict(args))
            if self.log is not None:
                self.log.append(f"{self.name}:{Path(args['path']).name}")
            if ctx.progress is not None:
                ctx.progress(ToolProgress("halfway"))
            if args["path"] in self.invalid_on:
                return ToolResult(self.invalid_on[args["path"]], True, "invalid")
            if args["path"] in self.timeout_on:
                return ToolResult("timed out after 900 s", True, "failed")
            if args.get("fail") or args["path"] in self.fail_on:
                return ToolResult("it broke", True, "failed")
            path = resolve_path(ctx, args["path"])
            if ctx.journal is not None:
                ctx.journal.before(path)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(args.get("prompt") or self.name))
            finally:
                if ctx.journal is not None:
                    ctx.journal.after(path)
            return ToolResult(f"wrote {args['path']}", False, "1 file")

        return Prepared(None, execute)


class GenerateImage(WriteFileTool):
    name: ClassVar[str] = "generate_image"


class Speak(WriteFileTool):
    name: ClassVar[str] = "speak"


class AssembleVideo(WriteFileTool):
    name: ClassVar[str] = "assemble_video"
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "scenes": {"type": "array"}},
        "required": ["path"],
    }


class SlowTool(WriteFileTool):
    """Waits at a `Gate` before writing, so a test can cancel it mid-call."""

    name: ClassVar[str] = "slow"
    mutating: ClassVar[bool] = False
    uncancellable: ClassVar[bool | None] = False  # a cancel interrupts the wait at once

    def __init__(self, gate: Gate) -> None:
        super().__init__()
        self.gate = gate

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        inner = await super().prepare(args, ctx)
        assert isinstance(inner, Prepared)

        async def execute() -> ToolResult:
            self.gate.reached.set()
            await asyncio.wait_for(self.gate.release.wait(), 5)
            return await inner.execute()

        return Prepared(None, execute)


class ReadOnlyTool(Tool):
    name: ClassVar[str] = "peek"
    description: ClassVar[str] = "reads"
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        return ToolResult("seen")


def tools_by_name(*tools: Tool) -> dict[str, Tool]:
    return {t.name: t for t in tools}


def make_workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root.resolve()
