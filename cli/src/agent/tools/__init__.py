from __future__ import annotations

from .base import Tool
from .fs import EditTool, ReadTool, WriteTool
from .search import GlobTool, GrepTool
from .shell import BashTool


def default_tools() -> list[Tool]:
    return [ReadTool(), GlobTool(), GrepTool(), WriteTool(), EditTool(), BashTool()]
