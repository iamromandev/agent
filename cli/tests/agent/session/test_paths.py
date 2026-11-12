from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from src.agent.session.paths import sessions_root, workspace_dir, workspace_id


def test_configured_root_wins(tmp_path: Path) -> None:
    env = {"XDG_DATA_HOME": "/xdg"}
    assert sessions_root(str(tmp_path / "s"), env) == tmp_path / "s"


def test_configured_root_expands_the_home_folder() -> None:
    assert sessions_root("~/sess", {}) == Path.home() / "sess"


def test_xdg_data_home_is_used_when_nothing_is_configured() -> None:
    assert sessions_root(None, {"XDG_DATA_HOME": "/xdg"}) == Path("/xdg/agent/sessions")


def test_default_root_is_under_the_home_folder() -> None:
    assert sessions_root(None, {}) == Path.home() / ".local" / "share" / "agent" / "sessions"


def test_empty_xdg_data_home_is_ignored() -> None:
    assert sessions_root(None, {"XDG_DATA_HOME": ""}) == Path.home() / ".local/share/agent/sessions"


def test_workspace_id_is_a_12_character_sha256_prefix(tmp_path: Path) -> None:
    expected = hashlib.sha256(str(tmp_path.resolve()).encode()).hexdigest()[:12]
    assert workspace_id(tmp_path) == expected
    assert len(workspace_id(tmp_path)) == 12


def test_workspace_id_is_stable_and_depends_on_the_path(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert workspace_id(tmp_path / "a") == workspace_id(tmp_path / "a")
    assert workspace_id(tmp_path / "a") != workspace_id(tmp_path / "b")


def test_symlinked_and_real_paths_give_the_same_id(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    assert workspace_id(link) == workspace_id(real)


def test_workspace_dir_is_root_plus_id(tmp_path: Path) -> None:
    assert workspace_dir(tmp_path / "root", tmp_path) == tmp_path / "root" / workspace_id(tmp_path)


def test_a_relative_configured_root_is_made_absolute_against_the_current_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    root = sessions_root("rel/sessions", {})
    assert root == Path.cwd() / "rel" / "sessions"
    assert root.is_absolute()
    monkeypatch.chdir(second)
    assert root == first.resolve() / "rel" / "sessions"  # fixed at the call: a later chdir does not move it


def test_a_relative_xdg_data_home_is_ignored() -> None:
    assert sessions_root(None, {"XDG_DATA_HOME": "relative/share"}) == Path.home() / ".local/share/agent/sessions"


def test_workspace_id_hashes_the_raw_bytes_of_a_non_utf8_path(tmp_path: Path) -> None:
    path = tmp_path / os.fsdecode(b"x\xff")
    expected = hashlib.sha256(os.fsencode(str(path.resolve()))).hexdigest()[:12]
    assert workspace_id(path) == expected
