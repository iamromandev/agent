"""What the agent needs from an MCP client (no MCP library here): a connector that opens one server connection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.agent.mcp.config import ServerConfig


@dataclass(frozen=True)
class ToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class CallResult:
    parts: list[str]  # text parts; non-text content is already a placeholder such as "[image omitted]"
    is_error: bool


class McpError(Exception):
    """An MCP operation failed; str(exc) is a one-line reason such as 'timeout after 60s'."""

    def __init__(self, message: str, summary: str | None = None) -> None:
        super().__init__(message)
        self.summary = summary or "error"


class McpTimeout(McpError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "timeout")


class McpExited(McpError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "server exited")


class McpConnection(Protocol):
    async def list_tools(self) -> list[ToolInfo]: ...

    async def call_tool(self, name: str, args: dict[str, Any], timeout: float) -> CallResult:
        """Raises McpTimeout (the server keeps running), McpExited, or McpError."""
        ...

    def on_exit(self, fn: Callable[[str], None]) -> None:
        """Call `fn(reason)` once if the server stops without `aclose()` having been called."""
        ...

    def kill(self) -> None:
        """Synchronous last resort: SIGKILL the server process if it is still alive. Never raises."""
        ...

    async def aclose(self) -> None:
        """Stop the server. Idempotent."""
        ...


class McpConnector(Protocol):
    async def connect(self, config: ServerConfig, cwd: Path) -> McpConnection:
        """Start the server and initialize the session. Raises McpError with a one-line reason."""
        ...

    def kill_all(self) -> None:
        """Synchronous last resort: SIGKILL every server process this connector started that is still alive,
        also those still connecting. Never raises."""
        ...
