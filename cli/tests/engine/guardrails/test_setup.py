"""`prepare` and `Guardrails`: the launch value that tools, the loop and the spawn sites share."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from src.engine.guardrails import setup as guard_setup
from src.engine.guardrails.setup import Guardrails, guard_of, prepare
from src.engine.tools.base import ToolContext, ToolResult


def make(tmp_path: Path, global_data: dict | None = None, project_data: dict | None = None, env: dict | None = None) -> Guardrails:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return prepare(tmp_path / "ws", global_data, project_data, env or {}, "Darwin", home=home)


def test_prepare_builds_paths_from_the_settings_patterns(tmp_path: Path) -> None:
    guard = make(
        tmp_path,
        {"guardrails": {"protected": ["~/secrets/**"]}},
        {"guardrails": {"protected": ["vault.json"]}},
    )

    assert guard.paths.is_protected(tmp_path / "home" / "secrets" / "x")
    assert guard.paths.is_protected(tmp_path / "ws" / "vault.json")
    assert guard.paths.is_protected(tmp_path / "home" / ".ssh" / "id")


def test_prepare_filters_the_environment_with_keep_env_and_lists_removed_names(tmp_path: Path) -> None:
    env = {"GITHUB_TOKEN": "ghp_secretvalue", "NPM_TOKEN": "npm_secretvalue", "PATH": "/bin"}

    guard = make(tmp_path, {"guardrails": {"keepEnv": ["NPM_TOKEN"]}}, None, env)

    assert guard.env.filter(env) == {"NPM_TOKEN": "npm_secretvalue", "PATH": "/bin"}
    assert guard.removed_env == ("GITHUB_TOKEN",)


def test_prepare_passes_the_settings_warnings_on(tmp_path: Path) -> None:
    guard = make(tmp_path, {"guardrails": {"protected": 5}})

    assert len(guard.warnings) == 1
    assert "guardrails.protected" in guard.warnings[0]


def test_the_stub_sandbox_does_not_wrap_and_says_why(tmp_path: Path) -> None:
    guard = make(tmp_path)

    assert guard.sandbox.profile is None
    assert guard.sandbox.wrap(["sh", "-c", "true"]) == ["sh", "-c", "true"]
    assert guard.sandbox.reason == "not built yet"


def test_the_empty_redactor_changes_nothing(tmp_path: Path) -> None:
    guard = make(tmp_path)
    result = ToolResult("ghp_abcdefghijklmnopqrstuvwxyz0123456789", display="shown")

    assert guard.redactor.redact("text") == ("text", 0)
    assert guard.redact_result(result) == result
    assert guard.redactions == 0


def test_off_has_builtin_paths_a_filter_and_no_sandbox(tmp_path: Path) -> None:
    guard = Guardrails.off(tmp_path)

    assert guard.paths.is_protected(tmp_path / ".env")
    assert guard.env.filter({"GITHUB_TOKEN": "x", "PATH": "/bin"}) == {"PATH": "/bin"}
    assert guard.sandbox.profile is None
    assert guard.warnings == ()
    assert guard.removed_env == ()


def test_guard_of_uses_the_context_guard_or_the_default(tmp_path: Path) -> None:
    own = Guardrails.off(tmp_path)

    assert guard_of(ToolContext(tmp_path, 1.0, guard=own)) is own
    assert guard_of(ToolContext(tmp_path, 1.0)).paths.is_protected(tmp_path / "server.pem")


def test_prepare_without_a_home_protects_the_account_home_as_well(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake, real = tmp_path / "fake", tmp_path / "real"
    fake.mkdir()
    real.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake)
    monkeypatch.setattr(guard_setup, "login_home", lambda: real)

    guard = prepare(tmp_path / "ws", None, None, {}, "Darwin")

    assert guard.paths.is_protected(fake / ".ssh" / "id")
    assert guard.paths.is_protected(real / ".ssh" / "id")
    assert guard.paths.home_folders == [fake, real]


def test_prepare_with_a_home_uses_only_that_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard_setup, "login_home", lambda: tmp_path / "real")

    guard = make(tmp_path)

    assert guard.paths.home_folders == [tmp_path / "home"]


def test_prepare_scans_the_workspace_for_files_with_protected_names(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / ".env").write_text("A=1\n")
    guard = make(tmp_path)
    link = tmp_path / "copy.txt"
    os.link(tmp_path / "ws" / ".env", link)

    assert guard.paths.is_protected(link)


def test_off_is_built_once_per_home_and_keeps_its_own_redaction_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    guard_setup._off.cache_clear()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    builds: list[Path] = []
    real_build = guard_setup._build
    monkeypatch.setattr(guard_setup, "_build", lambda *a, **k: (builds.append(a[1]), real_build(*a, **k))[1])

    first = Guardrails.off(tmp_path / "ws")
    second = Guardrails.off(tmp_path / "other")
    first.redactions = 7

    assert builds == [tmp_path]
    assert first.paths is second.paths
    assert second.redactions == 0
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "elsewhere")
    Guardrails.off(tmp_path)
    assert builds == [tmp_path, tmp_path / "elsewhere"]
    guard_setup._off.cache_clear()
