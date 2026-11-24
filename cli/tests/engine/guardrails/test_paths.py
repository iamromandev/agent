"""Protected paths (R3): built-in patterns, names anywhere, settings patterns, matching after resolution."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from src.engine.guardrails import paths as paths_module
from src.engine.guardrails.paths import BUILTIN_HOME, BUILTIN_NAMES, ProtectedPaths


@pytest.fixture
def home(tmp_path: Path) -> Path:
    folder = tmp_path / "home"
    folder.mkdir()
    return folder.resolve()


def protected(home: Path, *extra: str) -> ProtectedPaths:
    return ProtectedPaths(home, extra)


def test_a_settings_name_pattern_has_no_template_exception(home: Path) -> None:
    assert protected(home, ".env.example").is_protected(Path("/ws/.env.example"))


def test_patterns_list_the_builtin_ones_then_the_settings_ones(home: Path) -> None:
    listed = protected(home, "~/secrets/**", "*.token").patterns()

    assert listed[: len(BUILTIN_HOME) + len(BUILTIN_NAMES)] == [*BUILTIN_HOME, *BUILTIN_NAMES]
    assert listed[-2:] == ["~/secrets/**", "*.token"]


def test_the_counts_split_builtin_from_settings_patterns(home: Path) -> None:
    paths = protected(home, "a", "b/**")

    assert paths.builtin_count == len(BUILTIN_HOME) + len(BUILTIN_NAMES)
    assert paths.extra_count == 2


# --- the same file under another spelling (firmlinks, hard links, Unicode forms) -------


def test_files_deeper_than_the_depth_bound_are_not_tracked_as_hard_links(home: Path, tmp_path: Path) -> None:
    deep = home / ".aws" / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    (deep / "x.txt").write_text("x")
    link = tmp_path / "deep.txt"
    os.link(deep / "x.txt", link)

    paths = protected(home)

    assert paths.tracking_truncated
    assert not paths.is_protected(link)


@pytest.mark.parametrize("prefix", ["/system/volumes/data"])
def test_the_firmlink_prefix_is_removed_whatever_its_case(home: Path, monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
    monkeypatch.setattr(paths_module, "STRIP_FIRMLINK", True)
    paths = protected(home)  # built before the files below exist

    (home / ".ssh").mkdir()
    (home / ".ssh" / "config").write_text("Host x\n")  # created after launch: only the pattern can catch it

    assert paths.is_protected(Path(prefix + str(home / ".ssh" / "config")))
    assert paths.is_protected(Path(prefix + "/ws/.env"))
    assert not paths.is_protected(Path(prefix + "/ws/app.py"))
    assert not paths.is_protected(Path(prefix))


# --- settings patterns that name a folder ----------------------------------------------


def test_a_settings_pattern_with_a_dotdot_segment_is_ignored(home: Path) -> None:
    paths = protected(home, "~/a/../b/**", "../x")

    assert not paths.is_protected(home / "b" / "c")
    assert not paths.is_protected(home / "a" / ".." / "b" / "c")
    assert not paths.is_protected(Path("/ws/../x/y"))


# --- the open file itself (a path can change between a check and the open) --------------


def test_a_descriptor_that_cannot_be_examined_counts_as_protected(home: Path) -> None:
    assert protected(home).is_protected_fd(10_000)


# --- scan bounds, workspace names, trimmed patterns, second home -----------------------


def test_the_scan_of_a_protected_folder_stops_at_a_bound_on_folders_visited(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths_module, "MAX_TRACKED_DIRS", 5)
    for index in range(20):
        (home / ".aws" / f"d{index}").mkdir(parents=True)

    paths = protected(home)

    assert paths.tracking_truncated


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    folder = tmp_path / "ws"
    (folder / "config" / "deep").mkdir(parents=True)
    return folder


def test_a_hard_link_to_a_workspace_file_matching_a_settings_name_pattern_is_protected(
    home: Path, workspace: Path, tmp_path: Path
) -> None:
    (workspace / "a.token").write_text("t\n")
    paths = ProtectedPaths(home, ("*.token",), workspace)
    link = tmp_path / "copy.txt"
    os.link(workspace / "a.token", link)

    assert paths.is_protected(link)


def test_login_home_is_none_when_the_account_has_no_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(uid: int) -> object:
        raise KeyError(uid)

    monkeypatch.setattr(paths_module.pwd, "getpwuid", missing)

    assert paths_module.login_home() is None

