from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
from src.engine.configwatch import TrustRevoked
from src.engine.mcp.setup import prepare
from src.engine.mcp.trust import TrustStore


def write(path: Path, servers: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"servers": servers}))
    return path


def test_an_invalid_project_file_only_warns(tmp_path: Path) -> None:
    project = tmp_path / "ws" / ".agent" / "mcp.json"
    project.parent.mkdir(parents=True)
    project.write_text("{")

    setup = prepare(tmp_path / "ws", tmp_path / "none.json", TrustStore(tmp_path / "trust.json"))

    assert setup.servers == []
    assert setup.untrusted == frozenset()
    assert setup.warnings and setup.warnings[0].startswith(f"{project}: invalid JSON")


def test_a_store_inside_the_workspace_leaves_the_servers_untrusted_with_a_warning(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    write(workspace / ".agent" / "mcp.json", {"db": {"command": "psql"}})
    store = TrustStore(workspace / "data" / "trust.json")

    setup = prepare(workspace, tmp_path / "none.json", store)

    assert setup.untrusted == {"db"}
    assert any("trust store" in w and "inside the workspace" in w for w in setup.warnings)
    with pytest.raises(OSError, match="inside the workspace"):
        setup.remember()


def test_remember_after_a_revoke_is_refused_and_writes_nothing(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    write(workspace / ".agent" / "mcp.json", {"db": {"command": "psql"}})
    store = TrustStore(tmp_path / "data" / "trust.json")
    revoked = [False]
    paths: list[tuple[Path, ...]] = []

    @contextmanager
    def own_write(named: Sequence[Path]) -> Generator[None]:
        paths.append(tuple(named))
        revoked[0] = True  # the check that opens the write sees a change and revokes
        yield

    setup = prepare(workspace, tmp_path / "none.json", store, own_write, revoked=lambda: revoked[0])

    with pytest.raises(TrustRevoked):
        setup.remember()
    assert paths == [(store.path,)]
    assert not store.path.exists()
    with pytest.raises(TrustRevoked):
        setup.remember()  # and refused outright afterwards
    assert len(paths) == 1
