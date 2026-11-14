from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.agent.mcp.setup import prepare
from src.agent.mcp.trust import TrustStore


def write(path: Path, servers: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"servers": servers}))
    return path


def test_without_a_project_file_only_the_global_servers_load_and_nothing_is_untrusted(tmp_path: Path) -> None:
    global_ = write(tmp_path / "g" / "mcp.json", {"a": {"command": "x"}})

    setup = prepare(tmp_path / "ws", global_, TrustStore(tmp_path / "trust.json"))

    assert [(s.name, s.source) for s in setup.servers] == [("a", "global")]
    assert setup.untrusted == frozenset()


def test_project_servers_of_an_untrusted_file_are_untrusted_and_global_ones_are_not(tmp_path: Path) -> None:
    global_ = write(tmp_path / "g" / "mcp.json", {"a": {"command": "x"}})
    write(tmp_path / "ws" / ".agent" / "mcp.json", {"db": {"command": "psql"}})

    setup = prepare(tmp_path / "ws", global_, TrustStore(tmp_path / "trust.json"))

    assert [(s.name, s.source) for s in setup.servers] == [("a", "global"), ("db", "project")]
    assert setup.untrusted == {"db"}


def test_a_project_server_replaces_the_global_one_and_is_untrusted(tmp_path: Path) -> None:
    global_ = write(tmp_path / "g" / "mcp.json", {"db": {"command": "global-db"}})
    write(tmp_path / "ws" / ".agent" / "mcp.json", {"db": {"command": "project-db"}})

    setup = prepare(tmp_path / "ws", global_, TrustStore(tmp_path / "trust.json"))

    assert [(s.name, s.command, s.source) for s in setup.servers] == [("db", "project-db", "project")]
    assert setup.untrusted == {"db"}


def test_remembering_trusts_the_file_as_it_was_read_even_if_it_changed_since(tmp_path: Path) -> None:
    store = TrustStore(tmp_path / "trust.json")
    project = write(tmp_path / "ws" / ".agent" / "mcp.json", {"db": {"command": "psql"}})
    setup = prepare(tmp_path / "ws", tmp_path / "none.json", store)

    write(project, {"db": {"command": "rm -rf"}})  # edited after launch, before the user answered
    setup.remember()

    assert prepare(tmp_path / "ws", tmp_path / "none.json", store).untrusted == frozenset({"db"})  # the new bytes ask again
    project.write_text(json.dumps({"servers": {"db": {"command": "psql"}}}))
    assert prepare(tmp_path / "ws", tmp_path / "none.json", store).untrusted == frozenset()


def test_a_trusted_file_has_no_untrusted_servers(tmp_path: Path) -> None:
    store = TrustStore(tmp_path / "trust.json")
    write(tmp_path / "ws" / ".agent" / "mcp.json", {"db": {"command": "psql"}})
    prepare(tmp_path / "ws", tmp_path / "none.json", store).remember()

    setup = prepare(tmp_path / "ws", tmp_path / "none.json", store)

    assert setup.untrusted == frozenset()
    assert [s.source for s in setup.servers] == ["project"]


def test_an_edited_project_file_is_untrusted_again(tmp_path: Path) -> None:
    store = TrustStore(tmp_path / "trust.json")
    project = write(tmp_path / "ws" / ".agent" / "mcp.json", {"db": {"command": "psql"}})
    prepare(tmp_path / "ws", tmp_path / "none.json", store).remember()
    write(project, {"db": {"command": "psql", "args": ["--evil"]}})

    assert prepare(tmp_path / "ws", tmp_path / "none.json", store).untrusted == {"db"}


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


def test_a_symlinked_project_file_is_read_once_and_trusted_as_read(tmp_path: Path) -> None:
    store = TrustStore(tmp_path / "trust.json")
    target = write(tmp_path / "shared" / "mcp.json", {"db": {"command": "psql"}})
    link = tmp_path / "ws" / ".agent" / "mcp.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    setup = prepare(tmp_path / "ws", tmp_path / "none.json", store)

    write(target, {"db": {"command": "evil"}})  # the link now leads to other bytes
    setup.remember()

    assert [s.command for s in setup.servers] == ["psql"]
    assert prepare(tmp_path / "ws", tmp_path / "none.json", store).untrusted == {"db"}  # the new bytes ask again
    write(target, {"db": {"command": "psql"}})
    assert prepare(tmp_path / "ws", tmp_path / "none.json", store).untrusted == frozenset()
