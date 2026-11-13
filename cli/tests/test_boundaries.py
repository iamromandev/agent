"""The agent core stays free of UI and HTTP libraries: only the client and the TUI may use them."""

from __future__ import annotations

import ast
from pathlib import Path

AGENT_SRC = Path(__file__).parent.parent / "src" / "agent"
FORBIDDEN = {"textual", "httpx", "rich"}


def _imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_the_scan_sees_the_agent_modules() -> None:
    assert len(list(AGENT_SRC.rglob("*.py"))) > 10


def test_no_agent_module_imports_textual_httpx_or_rich() -> None:
    offenders = {
        str(path.relative_to(AGENT_SRC)): sorted(_imported_roots(path) & FORBIDDEN)
        for path in sorted(AGENT_SRC.rglob("*.py"))
        if _imported_roots(path) & FORBIDDEN
    }
    assert offenders == {}


SRC = AGENT_SRC.parent


def test_httpx_is_only_imported_by_the_client_package() -> None:
    offenders = sorted(
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if "httpx" in _imported_roots(path) and path.relative_to(SRC).parts[0] != "client"
    )
    assert offenders == []


def test_textual_is_only_imported_by_the_tui_package() -> None:
    offenders = sorted(
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if "textual" in _imported_roots(path) and path.relative_to(SRC).parts[0] != "tui"
    )
    assert offenders == []


async def test_the_no_network_fixture_blocks_real_requests() -> None:
    import httpx
    import pytest

    from tests.conftest import NetworkAccessError

    async with httpx.AsyncClient() as client:
        with pytest.raises(NetworkAccessError):
            await client.get("http://example.invalid")


def test_the_boundary_scan_catches_a_rich_import(tmp_path: Path) -> None:
    module = tmp_path / "m.py"
    module.write_text("from rich.text import Text\nimport rich\n")
    assert _imported_roots(module) & FORBIDDEN == {"rich"}
