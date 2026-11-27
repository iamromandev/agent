"""Which MCP servers a launch has and which of them may start: both config files, merged, and the trust check."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path

from core.engine.configwatch import TrustRevoked
from core.engine.mcp.config import ServerConfig, load_file, merge, project_config_path
from core.engine.mcp.trust import TrustStore


@dataclass(frozen=True)
class McpSetup:
    servers: list[ServerConfig]
    warnings: list[str]
    untrusted: frozenset[str]  # project servers that must not start until the user trusts the project file
    remember: Callable[[], None]  # trust the project file as it was read here; raises OSError if it cannot be saved


def prepare(
    workspace: Path,
    global_path: Path,
    store: TrustStore,
    own_write: Callable[[Sequence[Path]], AbstractContextManager[None]] = lambda paths: nullcontext(),
    *,
    revoked: Callable[[], bool] = lambda: False,
) -> McpSetup:
    """Read the global and the project file once. The project file's bytes are hashed as they were parsed.
    `own_write([store])` is entered around the trust store write, so the config watch does not take it for a child's
    change. Once `revoked()` (the config watch revoked project trust in this run) `remember` raises `TrustRevoked`."""
    project = load_file(project_config_path(workspace), "project")
    merged = merge(load_file(global_path, "global"), project)
    raw = project.raw
    needs_trust = bool(project.servers) and raw is not None and not store.is_trusted(workspace, raw)
    untrusted = frozenset(s.name for s in project.servers) if needs_trust else frozenset()
    warnings = list(merged.warnings)
    if needs_trust and store.inside(workspace):  # a repository's own "trust": never honoured, never written
        warnings.append(f"mcp: the trust store {store.path} is inside the workspace, so project servers stay untrusted")

    def remember() -> None:
        if revoked():
            raise TrustRevoked("trusting the project's MCP servers")
        if raw is not None:
            with own_write([store.path]):
                if revoked():  # the check that opens the write saw a change
                    raise TrustRevoked("trusting the project's MCP servers")
                store.trust(workspace, raw)
        if revoked():  # the check after it did: the guard already forgot the entry
            raise TrustRevoked("trusting the project's MCP servers")

    return McpSetup(merged.servers, warnings, untrusted, remember)
