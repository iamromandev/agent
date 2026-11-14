from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Tool
from .fs import EditTool, ReadTool, WriteTool
from .search import GlobTool, GrepTool
from .shell import BashTool
from .terminal import TermRead, TermStart, TermStop

if TYPE_CHECKING:
    from src.agent.terminal.manager import TerminalManager


def default_tools(terminals: TerminalManager | None = None) -> list[Tool]:
    """The built-in tools; the terminal tools only when there is a manager for them to use."""
    tools: list[Tool] = [ReadTool(), GlobTool(), GrepTool(), WriteTool(), EditTool(), BashTool()]
    if terminals is not None:
        tools += [TermStart(terminals), TermRead(terminals), TermStop(terminals)]
    return tools
