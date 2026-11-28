"""Reading the `lsp` key: the switch, the built-in table, overrides, bad entries, and the `.venv` rule."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from core.engine.lsp.config import LspSetup, find_on_path, read_lsp


def found(*names: str):
    def which(name: str) -> str | None:
        return f"/bin/{name}" if name in names else None

    return which


def names(setup: LspSetup) -> list[str]:
    return [s.name for s in setup.servers]


def test_no_lsp_key_is_off(tmp_path: Path) -> None:
    setup = read_lsp({}, None, tmp_path, which=found("ty", "ruff"))
    assert not setup.enabled and setup.servers == () and setup.warnings_text == ()


@pytest.mark.parametrize("value", [True, {}, {"warnings": True}])
def test_true_or_an_object_turns_it_on(tmp_path: Path, value: object) -> None:
    setup = read_lsp({"lsp": value}, None, tmp_path, which=found("ty"))
    assert setup.enabled and names(setup) == ["ty"] and setup.servers[0].source == "global"


@pytest.mark.parametrize("value", [False, {"enabled": False}, "yes"])
def test_false_or_a_disabled_object_is_off(tmp_path: Path, value: object) -> None:
    assert not read_lsp({"lsp": value}, None, tmp_path, which=found("ty")).enabled


def test_only_the_first_python_checker_found_joins_ruff(tmp_path: Path) -> None:
    setup = read_lsp({"lsp": True}, None, tmp_path, which=found("pyright-langserver", "basedpyright-langserver", "ruff", "gopls"))
    assert names(setup) == ["basedpyright", "ruff", "gopls"]
    assert setup.servers[0].args == ("--stdio",) and setup.servers[0].command == "/bin/basedpyright-langserver"


def test_a_built_in_that_is_not_found_is_left_out(tmp_path: Path) -> None:
    assert read_lsp({"lsp": True}, None, tmp_path, which=found()).servers == ()


def program(folder: Path, name: str) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return str(path)


def test_servers_override_disable_and_add(tmp_path: Path) -> None:
    gopls, dsl = program(tmp_path / "x", "gopls"), program(tmp_path / "x", "dsl")
    data = {"lsp": {"servers": {"ruff": False, "gopls": {"command": gopls, "args": ["-v"]}, "mydsl": {"command": dsl, "extensions": [".dsl"], "env": {"A": "1"}}}}}
    setup = read_lsp(data, None, tmp_path, which=found("ty", "ruff", "gopls"))
    by = {s.name: s for s in setup.servers}
    assert list(by) == ["ty", "gopls", "mydsl"]
    assert by["gopls"].command == gopls and by["gopls"].args == ("-v",) and by["gopls"].extensions == frozenset({".go"})
    assert by["mydsl"].extensions == frozenset({".dsl"}) and by["mydsl"].env == {"A": "1"}
    assert setup.warnings_text == ()


def test_ac8_a_bad_entry_is_one_escaped_warning_and_the_rest_load(tmp_path: Path) -> None:
    data = {"lsp": {"servers": {"my\x1bdsl": {"args": []}, "ruff": False, "bad2": {"command": "x", "extensions": ["py"]}, "bad3": {"command": "x", "extensions": [".a"], "args": [1]}}}}
    setup = read_lsp(data, None, tmp_path, which=found("ty", "ruff"))
    assert names(setup) == ["ty"]
    assert len(setup.warnings_text) == 3
    assert "\x1b" not in "".join(setup.warnings_text) and "my\\x1bdsl" in setup.warnings_text[0]


def test_warnings_flag(tmp_path: Path) -> None:
    assert read_lsp({"lsp": {"warnings": True}}, None, tmp_path, which=found()).warnings
    assert not read_lsp({"lsp": True}, None, tmp_path, which=found()).warnings


def test_phase_1_project_servers_and_project_only_built_ins_are_ignored_with_one_warning(tmp_path: Path) -> None:
    project = {"lsp": {"servers": {"mine": {"command": "/x/m", "extensions": [".m"]}}}}
    setup = read_lsp({"lsp": True}, project, tmp_path, which=found("ty"))
    assert names(setup) == ["ty"]
    assert setup.warnings_text == ("project lsp servers are not supported yet; ignored: mine",)

    only_project = read_lsp({}, {"lsp": True}, tmp_path, which=found("ty"))
    assert only_project.enabled and only_project.servers == ()
    assert only_project.warnings_text == ("project lsp servers are not supported yet; ignored: ty",)


GIT = shutil.which("git")


def venv_with(tmp_path: Path, tracked: bool) -> Path:
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    exe = tmp_path / ".venv" / "bin" / "ty"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    if tracked is not None:
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        if tracked:
            subprocess.run(["git", "add", "-f", ".venv/bin/ty"], cwd=tmp_path, check=True)
    return tmp_path


@pytest.mark.skipif(GIT is None, reason="needs git")
def test_an_untracked_venv_in_a_git_repo_is_used_but_counts_as_project(tmp_path: Path) -> None:
    ws = venv_with(tmp_path, tracked=False)
    setup = read_lsp({"lsp": True}, None, ws, which=found("basedpyright-langserver"))
    # ty from .venv is project-sourced: skipped in phase 1, and the next checker is used
    assert names(setup) == ["basedpyright"]
    assert setup.warnings_text == ("project lsp servers are not supported yet; ignored: ty",)


@pytest.mark.skipif(GIT is None, reason="needs git")
def test_a_tracked_venv_is_never_used(tmp_path: Path) -> None:
    ws = venv_with(tmp_path, tracked=True)
    setup = read_lsp({"lsp": True}, None, ws, which=found())
    assert setup.servers == () and setup.warnings_text == ()


def test_outside_git_the_venv_is_never_searched(tmp_path: Path) -> None:
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    exe = tmp_path / ".venv" / "bin" / "ty"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    assert read_lsp({"lsp": True}, None, tmp_path, which=found()).servers == ()


@pytest.mark.skipif(GIT is None, reason="needs git")
def test_a_project_venv_program_does_not_hide_the_one_on_path(tmp_path: Path) -> None:
    ws = venv_with(tmp_path, tracked=False)
    setup = read_lsp({"lsp": True}, None, ws, which=found("ty"))
    assert names(setup) == ["ty"] and setup.servers[0].command == "/bin/ty"
    assert setup.warnings_text == ()  # something usable was found, so nothing is reported


@pytest.mark.skipif(GIT is None, reason="needs git")
def test_git_runs_at_most_once_and_only_when_a_venv_program_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine.lsp import config

    runs: list[object] = []
    real = subprocess.run
    monkeypatch.setattr(config.subprocess, "run", lambda *a, **k: runs.append(a) or real(*a, **k))
    read_lsp({"lsp": True}, None, tmp_path, which=found("ty", "ruff"))
    assert runs == []  # no .venv/bin program: no git
    ws = venv_with(tmp_path, tracked=False)
    exe = ws / ".venv" / "bin" / "ruff"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    runs.clear()
    read_lsp({"lsp": True}, None, ws, which=found())
    assert len(runs) == 1  # ty and ruff both have one, git is asked once


def test_a_configured_command_must_exist_and_be_executable_or_be_found_on_an_absolute_path_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = tmp_path / "plain"
    plain.write_text("x")
    good = program(tmp_path / "bin", "good")
    monkeypatch.setenv("PATH", f"relative{os.pathsep}{tmp_path / 'bin'}")
    servers = {
        "gone": {"command": str(tmp_path / "nope"), "extensions": [".a"]},
        "plain": {"command": str(plain), "extensions": [".b"]},
        "bare": {"command": "good", "extensions": [".c"]},
        "missing": {"command": "not-on-path", "extensions": [".d"]},
        "rel": {"command": "bin/good", "extensions": [".e"]},
    }
    setup = read_lsp({"lsp": {"servers": servers}}, None, tmp_path, which=find_on_path)
    assert {s.name: s.command for s in setup.servers} == {"bare": good, "rel": good}
    assert len(setup.warnings_text) == 3
    assert all(f'servers."{n}"' in w for n, w in zip(("gone", "plain", "missing"), setup.warnings_text, strict=True))


def test_extensions_are_lower_cased_when_read(tmp_path: Path) -> None:
    exe = program(tmp_path / "bin", "dsl")
    setup = read_lsp({"lsp": {"servers": {"d": {"command": exe, "extensions": [".DSL", ".Py"]}}}}, None, tmp_path, which=found())
    assert setup.servers[0].extensions == frozenset({".dsl", ".py"})
