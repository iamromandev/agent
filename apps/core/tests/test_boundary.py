"""core must never import from an app that depends on it (apps/cli ships the top-level `src` package)."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "core"


def _imports_src(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name == "src" or a.name.startswith("src.")]
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and (node.module == "src" or node.module.startswith("src."))
        ):
            found.append(node.module)
    return found


def test_no_module_in_core_imports_src() -> None:
    offenders = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        bad = _imports_src(ast.parse(path.read_text(), filename=str(path)))
        if bad:
            offenders[str(path.relative_to(PACKAGE))] = bad
    assert offenders == {}
