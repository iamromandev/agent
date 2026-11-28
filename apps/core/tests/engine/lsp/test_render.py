"""Rendering diagnostics: bounded pieces."""

from __future__ import annotations

from core.engine.lsp.render import render
from core.engine.lsp.server import Diagnostic


def test_a_long_code_is_cut_before_it_is_shown() -> None:
    found = [Diagnostic(1, 1, 1, "m", "E" * 500)]
    block = render("ty", "a.py", found, warnings=False, max_lines=20)
    assert block is not None and len(block.rows[0]) < 150
