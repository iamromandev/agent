from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Tool
from .fs import EditTool, ReadTool, WriteTool
from .image import GenerateImageTool
from .search import GlobTool, GrepTool
from .shell import BashTool
from .skill import SkillTool
from .terminal import TermRead, TermStart, TermStop
from .todo import TodoTool
from .web import WebFetch
from .web_search import WebSearch

if TYPE_CHECKING:
    from src.engine.imagegen import MfluxGenerator
    from src.engine.search_web import Searcher
    from src.engine.skills.loader import Skills
    from src.engine.terminal.manager import TerminalManager
    from src.engine.web import Fetcher


def read_only_tools(
    fetcher: Fetcher | None = None, searcher: Searcher | None = None, skills: Skills | None = None
) -> list[Tool]:
    """What a `task` child may use: reading and searching, and web_fetch and web_search when there is a fetcher or searcher."""
    tools: list[Tool] = [ReadTool(), GlobTool(), GrepTool()]
    if fetcher is not None:
        tools.append(WebFetch(fetcher))
    if searcher is not None:
        tools.append(WebSearch(searcher))
    if skills is not None and skills.by_name:
        tools.append(SkillTool(skills))
    return tools


def default_tools(
    terminals: TerminalManager | None = None,
    fetcher: Fetcher | None = None,
    searcher: Searcher | None = None,
    skills: Skills | None = None,
    imager: MfluxGenerator | None = None,
) -> list[Tool]:
    """The built-in tools; the terminal, web and image tools only when there is a manager, fetcher or generator to use."""
    tools: list[Tool] = [ReadTool(), GlobTool(), GrepTool(), WriteTool(), EditTool(), BashTool(), TodoTool()]
    if terminals is not None:
        tools += [TermStart(terminals), TermRead(terminals), TermStop(terminals)]
    if fetcher is not None:
        tools.append(WebFetch(fetcher))
    if searcher is not None:
        tools.append(WebSearch(searcher))
    if skills is not None and skills.by_name:
        tools.append(SkillTool(skills))
    if imager is not None:
        tools.append(GenerateImageTool(imager))
    return tools
