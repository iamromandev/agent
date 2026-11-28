"""Reading the `lsp` key: the switch, the built-in table, overrides, bad entries, and the `.venv` rule."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from core.engine.lsp import config as config_module
from core.engine.lsp.config import LspSetup, changed_since_launch, find_on_path, project_needs_trust, read_lsp


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


def test_project_servers_and_project_only_built_ins_are_read_with_the_project_source(tmp_path: Path) -> None:
    project = {"lsp": {"servers": {"mine": {"command": "/bin/sh", "extensions": [".m"]}}}}
    setup = read_lsp({"lsp": True}, project, tmp_path, which=found("ty"))
    assert [(s.name, s.source) for s in setup.servers] == [("ty", "global"), ("mine", "project")]
    assert setup.warnings_text == ()

    only_project = read_lsp({}, {"lsp": True}, tmp_path, which=found("ty"))
    assert [(s.name, s.source) for s in only_project.servers] == [("ty", "project")]


def test_project_needs_trust_only_for_a_project_sourced_server_that_will_run(tmp_path: Path) -> None:
    on = found("ty")
    assert project_needs_trust({}, {"lsp": True}, tmp_path, which=on)  # a built-in turned on by the project file
    assert project_needs_trust({}, {"lsp": {"servers": {"m": {"command": "/bin/sh", "extensions": [".m"]}}}}, tmp_path, which=on)
    assert not project_needs_trust({"lsp": True}, {"lsp": True}, tmp_path, which=on)  # both: the global file turned it on
    assert not project_needs_trust({}, {"lsp": True}, tmp_path, which=found())  # nothing found: nothing to start
    assert not project_needs_trust({}, {"lsp": {"servers": {"m": {"command": "nope"}}}}, tmp_path, which=found())  # invalid entry
    assert not project_needs_trust({}, {"lsp": False}, tmp_path, which=on)
    assert not project_needs_trust({}, {"lsp": {"enabled": False}}, tmp_path, which=on)
    assert not project_needs_trust(None, None, tmp_path, which=on)


def test_a_global_relative_command_that_lands_in_the_workspace_is_project_sourced(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    program(ws / "bin", "evil")
    outside = program(tmp_path / "tools", "fine")
    (ws / "link").symlink_to(tmp_path / "tools")  # a symlink out of the workspace: where it really is counts
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "back").symlink_to(ws / "bin" / "evil")
    servers = {
        "rel": {"command": "bin/evil", "extensions": [".a"]},
        "abs": {"command": str(ws / "bin" / "evil"), "extensions": [".b"]},
        "viaout": {"command": "link/fine", "extensions": [".c"]},
        "away": {"command": outside, "extensions": [".d"]},
        "back": {"command": str(tmp_path / "other" / "back"), "extensions": [".e"]},
    }
    setup = read_lsp({"lsp": {"servers": servers}}, None, ws, which=found())
    assert {s.name: s.source for s in setup.servers} == {
        "rel": "project", "abs": "project", "viaout": "global", "away": "global", "back": "project",
    }


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
def test_an_untracked_venv_takes_the_source_of_the_file_that_turned_lsp_on(tmp_path: Path) -> None:
    ws = venv_with(tmp_path, tracked=False)
    setup = read_lsp({"lsp": True}, None, ws, which=found())
    assert [(s.name, s.source) for s in setup.servers] == [("ty", "global")]  # no trust needed
    assert setup.warnings_text == ()
    only_project = read_lsp({}, {"lsp": True}, ws, which=found())
    assert [(s.name, s.source) for s in only_project.servers] == [("ty", "project")]


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
    assert len(runs) == 2  # ty and ruff both have one: one lookup (rev-parse, then ls-files)


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


@pytest.mark.skipif(GIT is None, reason="needs git")
def test_a_workspace_inside_a_parent_repository_does_not_search_venv(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)  # the parent repo; `.venv` untracked there
    ws = tmp_path / "sub"
    program(ws / ".venv" / "bin", "ty")
    assert read_lsp({"lsp": True}, None, ws, which=found()).servers == ()
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)  # its own repo: now it is the top
    assert names(read_lsp({"lsp": True}, None, ws, which=found())) == ["ty"]


@pytest.mark.skipif(GIT is None, reason="needs git")
def test_git_ignores_the_repositorys_own_fsmonitor_hook(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    venv_with(ws, tracked=False)
    marker = tmp_path / "ran"
    hook = tmp_path / "hook.sh"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)
    subprocess.run(["git", "config", "core.fsmonitor", str(hook)], cwd=ws, check=True)
    assert names(read_lsp({"lsp": True}, None, ws, which=found())) == ["ty"]
    assert not marker.exists()


@pytest.mark.skipif(GIT is None, reason="needs git")
def test_a_venv_program_is_fingerprinted_through_symlinks_and_checked_again(tmp_path: Path) -> None:
    ws = venv_with(tmp_path, tracked=False)
    real = tmp_path / "real-ty"
    real.write_text("#!/bin/sh\n")
    real.chmod(0o755)
    (ws / ".venv" / "bin" / "ty").unlink()
    (ws / ".venv" / "bin" / "ty").symlink_to(real)
    spec = read_lsp({"lsp": True}, None, ws, which=found()).servers[0]
    assert spec.fingerprint is not None and spec.fingerprint.path == str(real.resolve())
    assert not changed_since_launch(spec.command, spec.fingerprint)
    real.write_text("#!/bin/sh\necho changed\n")
    assert changed_since_launch(spec.command, spec.fingerprint)


@pytest.mark.skipif(GIT is None, reason="needs git")
def test_a_venv_program_over_the_size_limit_is_not_used_and_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = venv_with(tmp_path, tracked=False)
    monkeypatch.setattr(config_module, "MAX_PROGRAM_BYTES", 4)
    setup = read_lsp({"lsp": True}, None, ws, which=found())
    assert setup.servers == () and len(setup.warnings_text) == 1
    assert "ty is over 0 MiB; not used" in setup.warnings_text[0]
