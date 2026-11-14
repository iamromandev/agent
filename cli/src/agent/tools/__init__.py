from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Tool
from .fs import EditTool, ReadTool, WriteTool
from .search import GlobTool, GrepTool
from .shell import BashTool
from .terminal import TermRead, TermStart, TermStop
from .web import WebFetch

if TYPE_CHECKING:
    from src.agent.terminal.manager import TerminalManager
    from src.agent.web import Fetcher


def default_tools(terminals: TerminalManager | None = None, fetcher: Fetcher | None = None) -> list[Tool]:
    """The built-in tools; the terminal and web tools only when there is a manager or fetcher to use."""
    tools: list[Tool] = [ReadTool(), GlobTool(), GrepTool(), WriteTool(), EditTool(), BashTool()]
    if terminals is not None:
        tools += [TermStart(terminals), TermRead(terminals), TermStop(terminals)]
    if fetcher is not None:
        tools.append(WebFetch(fetcher))
    return tools
