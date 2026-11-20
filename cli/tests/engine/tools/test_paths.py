from __future__ import annotations

from pathlib import Path

import pytest
from src.engine.tools.base import ToolContext
from src.engine.tools.paths import inside_workspace, resolve_path


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return ToolContext(workspace=workspace.resolve(), bash_timeout=5.0)


def test_tilde_with_an_unknown_user_is_a_literal_relative_path(ctx: ToolContext) -> None:
    path = resolve_path(ctx, "~nouser-xyz/x")
    assert path == ctx.workspace / "~nouser-xyz" / "x"
    assert inside_workspace(path, ctx)
