"""The sandbox profile text and `make_sandbox` (R11, R13). Pure text: no `sandbox-exec` needed, so every platform runs it."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.engine.guardrails import sandbox as sandbox_module
from src.engine.guardrails.config import GuardConfig
from src.engine.guardrails.paths import ProtectedPaths
from src.engine.guardrails.sandbox import build_profile, make_sandbox


@pytest.fixture
def home(tmp_path: Path) -> Path:
    folder = tmp_path / "home"
    folder.mkdir()
    return folder.resolve()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    folder = tmp_path / "ws"
    folder.mkdir()
    return folder.resolve()


def profile(workspace: Path, home: Path, *, writable: list[Path] | None = None, deny_write: list[Path] | None = None,
            extra: tuple[str, ...] = (), network: bool = True) -> str:
    return build_profile(workspace, writable if writable is not None else [workspace], deny_write or [],
                         ProtectedPaths(home, extra), network)


def index(text: str, needle: str) -> int:
    assert needle in text, f"{needle!r} not in the profile"
    return text.index(needle)


def test_a_wildcard_path_pattern_becomes_an_anchored_regex(workspace: Path, home: Path) -> None:
    text = profile(workspace, home, extra=("~/logs/*.secret",))

    expected = "".join(f"[{c.lower()}{c.upper()}]" if c.isalpha() else c for c in f"{home}/logs/")
    body = f'{expected.replace(".", chr(92) + ".")}[^/]*\\.[sS][eE][cC][rR][eE][tT]$"'
    assert f'#"^{body}' in text
    assert f'#"^/System/Volumes/Data{body}' in text


def test_a_quote_in_a_regex_is_written_as_a_dot_because_it_cannot_be_escaped(workspace: Path, home: Path) -> None:
    text = profile(workspace, home, extra=('we"ird/**',))

    assert '#"^(.*/)?[wW][eE].[iI][rR][dD]/.*$"' in text
    assert all(line.count('"') % 2 == 0 for line in text.splitlines() if "#" in line)


def test_a_glob_bracket_range_folds_each_endpoint_pair(workspace: Path, home: Path) -> None:
    from src.engine.guardrails.sandbox import glob_regex

    assert glob_regex("[a-c]") == "[a-cA-C]"
    assert glob_regex("[a-z0-9]") == "[a-zA-Z0-9]"
    assert glob_regex("[A-Z]") == "[A-Za-z]"
    assert glob_regex("[!a-c]") == "[^a-cA-C]"
    assert glob_regex("[abc]") == "[aAbBcC]"
    assert glob_regex("[_a-c-]") == "[_a-cA-C-]"


def test_network_off_denies_outgoing_ip_except_loopback_and_allows_unix_sockets(workspace: Path, home: Path) -> None:
    text = profile(workspace, home, network=False)

    assert '(deny network*)(allow network* (remote ip "localhost:*") (remote unix-socket))' in text
    assert "(local ip)" not in text  # it matches every outgoing connection


# make_sandbox (AC13)


def make(cfg: GuardConfig, workspace: Path, home: Path, *, system: str = "Darwin", env: dict | None = None,
         probe=lambda argv: True, sessions: Path | None = None):
    return make_sandbox(cfg, workspace, home, env or {}, system, probe, sessions=sessions)


@pytest.fixture
def fake_exec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "sandbox-exec"
    path.write_text("")
    monkeypatch.setattr(sandbox_module, "SANDBOX_EXEC", path)
    return path


def test_make_sandbox_adds_the_global_write_list_expanding_the_home_folder(workspace: Path, home: Path, fake_exec: Path) -> None:
    cfg = GuardConfig(extra_write=("~/extra/cache", str(workspace.parent / "elsewhere")))

    sandbox, warning = make(cfg, workspace, home)

    assert warning is None
    assert home / "extra" / "cache" in sandbox.writable
    assert workspace.parent / "elsewhere" in sandbox.writable


def test_a_missing_sandbox_exec_runs_unsandboxed_with_one_warning(workspace: Path, home: Path, tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_module, "SANDBOX_EXEC", tmp_path / "missing")

    sandbox, warning = make(GuardConfig(), workspace, home)

    assert sandbox.profile is None
    assert sandbox.reason == "unavailable: sandbox-exec not found"
    assert warning is not None and "sandbox-exec" in warning


def test_a_probe_that_raises_is_a_failed_probe(workspace: Path, home: Path, fake_exec: Path) -> None:
    def probe(argv: list[str]) -> bool:
        raise OSError("boom")

    sandbox, warning = make(GuardConfig(), workspace, home, probe=probe)

    assert sandbox.profile is None
    assert warning is not None


def test_a_profile_that_cannot_be_built_falls_back_with_a_warning(workspace: Path, home: Path, fake_exec: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> str:
        raise ValueError("bad pattern")

    monkeypatch.setattr(sandbox_module, "build_profile", broken)

    sandbox, warning = make(GuardConfig(), workspace, home)

    assert sandbox.profile is None
    assert warning is not None and "sandbox" in warning


def test_the_warnings_escape_control_characters_from_settings(workspace: Path, home: Path, fake_exec: Path) -> None:
    _, warning = make(GuardConfig(extra_write=("bad\x1b[31m",)), workspace, home)

    assert warning is not None and "\x1b" not in warning


def test_a_workspace_that_is_the_home_folder_or_holds_a_protected_folder_is_a_warning(
    home: Path, fake_exec: Path
) -> None:
    (home / ".ssh").mkdir()
    for workspace in (home, home.parent):
        _, warning = make(GuardConfig(), workspace, home)

        assert warning is not None and "workspace" in warning and ".ssh" in warning, workspace
