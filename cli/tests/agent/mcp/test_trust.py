from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from src.agent.mcp.trust import TrustStore, trust_store_path
from src.config import get_settings

CONTENT = b'{"servers": {"db": {"command": "x"}}}'


def test_nothing_is_trusted_before_anything_was_trusted(tmp_path: Path) -> None:
    assert TrustStore(tmp_path / "trust.json").is_trusted(tmp_path / "ws", CONTENT) is False


def test_trusted_content_in_a_folder_is_trusted_again(tmp_path: Path) -> None:
    store = TrustStore(tmp_path / "data" / "trust.json")
    store.trust(tmp_path / "ws", CONTENT)

    assert store.is_trusted(tmp_path / "ws", CONTENT) is True
    assert TrustStore(tmp_path / "data" / "trust.json").is_trusted(tmp_path / "ws", CONTENT) is True  # persisted


def test_changed_content_is_not_trusted(tmp_path: Path) -> None:
    store = TrustStore(tmp_path / "trust.json")
    store.trust(tmp_path / "ws", CONTENT)

    assert store.is_trusted(tmp_path / "ws", CONTENT + b" ") is False


def test_trust_is_per_folder(tmp_path: Path) -> None:
    store = TrustStore(tmp_path / "trust.json")
    store.trust(tmp_path / "one", CONTENT)

    assert store.is_trusted(tmp_path / "two", CONTENT) is False


def test_trusting_a_second_folder_keeps_the_first(tmp_path: Path) -> None:
    store = TrustStore(tmp_path / "trust.json")
    store.trust(tmp_path / "one", CONTENT)
    store.trust(tmp_path / "two", b"{}")

    assert store.is_trusted(tmp_path / "one", CONTENT)
    assert store.is_trusted(tmp_path / "two", b"{}")


def test_trusting_new_content_replaces_the_old_hash_of_the_folder(tmp_path: Path) -> None:
    store = TrustStore(tmp_path / "trust.json")
    store.trust(tmp_path / "ws", CONTENT)
    store.trust(tmp_path / "ws", b"{}")

    assert not store.is_trusted(tmp_path / "ws", CONTENT)
    assert store.is_trusted(tmp_path / "ws", b"{}")


def test_the_store_holds_the_sha256_of_the_exact_bytes_per_folder(tmp_path: Path) -> None:
    path = tmp_path / "trust.json"
    TrustStore(path).trust(tmp_path / "ws", CONTENT)

    assert json.loads(path.read_text()) == {str(tmp_path / "ws"): hashlib.sha256(CONTENT).hexdigest()}


@pytest.mark.parametrize("junk", [b"{not json", b"[1, 2]", b'{"/ws": 5}', b"\xff\xfe", b""])
def test_a_corrupt_file_means_untrusted_and_the_next_trust_overwrites_it(tmp_path: Path, junk: bytes) -> None:
    path = tmp_path / "trust.json"
    path.write_bytes(junk)
    store = TrustStore(path)

    assert store.is_trusted(tmp_path / "ws", CONTENT) is False
    store.trust(tmp_path / "ws", CONTENT)

    assert store.is_trusted(tmp_path / "ws", CONTENT) is True


def test_a_failed_write_leaves_the_old_file_and_no_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "trust.json"
    store = TrustStore(path)
    store.trust(tmp_path / "ws", CONTENT)
    before = path.read_bytes()

    def broken(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError, match="disk full"):
        store.trust(tmp_path / "other", b"{}")

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["trust.json"]


def test_the_file_is_private_to_the_user(tmp_path: Path) -> None:
    path = tmp_path / "trust.json"
    TrustStore(path).trust(tmp_path / "ws", CONTENT)

    assert path.stat().st_mode & 0o077 == 0


def test_the_store_is_under_xdg_data_home() -> None:
    assert trust_store_path({"XDG_DATA_HOME": "/xdg"}) == Path("/xdg/agent/mcp-trust.json")


def test_the_store_falls_back_to_the_home_data_folder() -> None:
    assert trust_store_path({}) == Path.home() / ".local" / "share" / "agent" / "mcp-trust.json"


def test_a_relative_xdg_data_home_is_ignored() -> None:
    assert trust_store_path({"XDG_DATA_HOME": "rel"}) == Path.home() / ".local" / "share" / "agent" / "mcp-trust.json"


def test_the_sessions_folder_setting_does_not_move_the_store() -> None:
    env = {"XDG_DATA_HOME": "/xdg", "AGENT_SESSIONS_DIR": "/repo/sessions"}
    assert trust_store_path(env) == Path("/xdg/agent/mcp-trust.json")


def test_a_dot_env_in_the_workspace_cannot_move_the_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings reads `.env` from the current folder, which is the workspace: the store must not follow it."""
    (tmp_path / ".env").write_text("AGENT_SESSIONS_DIR=sessions\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_SESSIONS_DIR")
    get_settings.cache_clear()
    assert get_settings().sessions_dir == "sessions"  # the attack: the repo does steer the sessions folder

    assert not trust_store_path().is_relative_to(tmp_path)


def test_a_store_inside_the_workspace_is_never_read(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    path = workspace / "data" / "trust.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({str(workspace): hashlib.sha256(CONTENT).hexdigest()}))  # as a repo could ship it

    store = TrustStore(path)

    assert store.inside(workspace) is True
    assert store.is_trusted(workspace, CONTENT) is False


def test_a_store_inside_the_workspace_is_never_written(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = TrustStore(workspace / "trust.json")

    with pytest.raises(OSError, match="inside the workspace"):
        store.trust(workspace, CONTENT)

    assert list(workspace.iterdir()) == []


def test_a_store_reached_through_a_symlink_into_the_workspace_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (tmp_path / "link").symlink_to(workspace)

    assert TrustStore(tmp_path / "link" / "trust.json").inside(workspace) is True


def test_a_store_next_to_the_workspace_is_fine(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()

    assert TrustStore(tmp_path / "trust.json").inside(workspace) is False
