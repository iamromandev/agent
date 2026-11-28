"""`/lsp`: the text of `info.lsp_summary`."""

from __future__ import annotations

import os
from pathlib import Path

from core.engine.lsp.config import LspSetup, ServerSpec
from core.engine.lsp.hub import LspHub
from core.info import lsp_summary


def hub(*specs: ServerSpec, warnings: tuple[str, ...] = ()) -> LspHub:
    return LspHub(LspSetup(True, False, specs, warnings), Path("."), dict(os.environ), wait=1, start_wait=1, max_lines=20)


def test_without_a_hub_it_says_how_to_turn_lsp_on() -> None:
    assert lsp_summary(None) == 'LSP is off (set "lsp": true in settings)'


def test_a_hub_with_no_server_says_so() -> None:
    assert lsp_summary(hub()) == "No language servers found."


def test_each_server_is_one_escaped_line_idle_until_a_file_matches_and_warnings_follow() -> None:
    spec = ServerSpec("ty\x1b", "/bin/ty", (), frozenset({".py"}), {}, "global")
    other = ServerSpec("ruff", "/bin/ruff", (), frozenset({".py"}), {}, "project")
    text = lsp_summary(hub(spec, other, warnings=("lsp: something odd",)))
    assert text.splitlines() == ["ty\\x1b (global) idle", "ruff (project) idle", "lsp: something odd"]
