"""A stdio MCP server for tests. FAKE_MCP_MODE picks the behavior:

normal     tools echo, env, image, mixed, fail, big, hang, exit, pid
slow_init  sleeps 5 s before it starts serving
hang_init  never answers `initialize`
garbage    writes text that is not JSON to stdout, then exits
dup        tools "do/thing" and "do_thing" (the same name once cleaned)
"""

from __future__ import annotations

import asyncio
import base64
import os
import sys
import time

import mcp_types as types
from mcp.server.mcpserver import MCPServer

mode = os.environ.get("FAKE_MCP_MODE", "normal")
server = MCPServer("fake")


@server.tool()
def echo(text: str) -> str:
    """Return the text it is given."""
    return text


@server.tool()
def env(name: str) -> str:
    """Return one environment variable of the server process."""
    return os.environ.get(name, "")


@server.tool()
def pid() -> str:
    """Return the process id and the working folder of the server."""
    return f"{os.getpid()} {os.getcwd()}"


@server.tool()
def image() -> list[types.TextContent | types.ImageContent]:
    """Two text parts and an image."""
    png = base64.b64encode(b"\x89PNG").decode()
    return [
        types.TextContent(text="first"),
        types.ImageContent(data=png, mime_type="image/png"),
        types.TextContent(text="second"),
    ]


@server.tool()
def mixed() -> list[types.AudioContent | types.ResourceLink | types.EmbeddedResource]:
    """Audio, a resource link and an embedded resource."""
    return [
        types.AudioContent(data=base64.b64encode(b"RIFF").decode(), mime_type="audio/wav"),
        types.ResourceLink(name="doc", uri="file:///doc.txt"),
        types.EmbeddedResource(resource=types.TextResourceContents(uri="file:///emb.txt", text="inline")),
    ]


@server.tool()
def fail() -> str:
    """Always fails."""
    raise ValueError("boom")


@server.tool()
def big(n: int) -> str:
    """Return n characters."""
    return "x" * n


@server.tool()
async def hang() -> str:
    """Never answers."""
    await asyncio.sleep(3600)
    return "late"


@server.tool(name="exit")
def die() -> str:
    """Kill the server process without answering."""
    os._exit(3)


if mode == "dup":
    server = MCPServer("fake")

    @server.tool(name="do/thing")
    def do_slash() -> str:
        """First."""
        return "slash"

    @server.tool(name="do_thing")
    def do_underscore() -> str:
        """Second."""
        return "underscore"


if __name__ == "__main__":
    if mode == "garbage":
        sys.stdout.write("this is not json\n")
        sys.stdout.flush()
        sys.exit(0)
    if mode == "slow_init":
        time.sleep(5)
    if mode == "hang_init":
        time.sleep(3600)
    server.run("stdio")
