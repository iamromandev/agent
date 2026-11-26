"""Model-facing names of MCP tools."""

from __future__ import annotations

import re

ALLOWED = re.compile(r"[^A-Za-z0-9_-]")


def clean(text: str) -> str:
    return ALLOWED.sub("_", text)


def tool_name(server: str, tool: str) -> str:
    """`mcp__<server>__<tool>`, with every character a tool name may not have replaced by `_`."""
    return f"mcp__{clean(server)}__{clean(tool)}"
