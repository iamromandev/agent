"""`prepare` and `Guardrails`: the launch value that tools, the loop and the spawn sites share."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from src.engine.guardrails import sandbox as sandbox_module
from src.engine.guardrails import setup as guard_setup
from src.engine.guardrails.setup import Guardrails, guard_of, prepare
from src.engine.tools.base import Preview, ToolContext, ToolResult


def make(tmp_path: Path, global_data: dict | None = None, project_data: dict | None = None, env: dict | None = None) -> Guardrails:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return prepare(tmp_path / "ws", global_data, project_data, env or {}, "Linux", home=home)  # no sandbox: one platform


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

    assert [w for w in guard.warnings if "guardrails.protected" in w] != []
    assert len(guard.warnings) == 2  # the other one: the sandbox is not available on this system


def test_off_has_no_sandbox_and_says_so(tmp_path: Path) -> None:
    guard = Guardrails.off(tmp_path)

    assert guard.sandbox.profile is None
    assert guard.sandbox.reason == "off: not configured"
    assert guard.sandbox.wrap(["sh", "-c", "true"]) == ["sh", "-c", "true"]


def test_prepare_on_another_system_warns_once_and_runs_unsandboxed(tmp_path: Path) -> None:
    guard = make(tmp_path)

    assert guard.sandbox.profile is None
    assert guard.sandbox.reason == "unavailable: not macOS"
    assert len(guard.warnings) == 1 and "not macOS" in guard.warnings[0]
    assert guard.info == ()


def test_prepare_builds_the_sandbox_with_the_settings_and_the_sessions_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "sandbox-exec"
    exe.write_text("")
    monkeypatch.setattr(sandbox_module, "SANDBOX_EXEC", exe)
    home = tmp_path / "home"
    home.mkdir()
    seen: list[list[str]] = []

    guard = prepare(
        tmp_path / "ws",
        {"guardrails": {"sandbox": {"network": False}, "protected": ["~/secrets/**"]}},
        None,
        {},
        "Darwin",
        lambda argv: seen.append(argv) or True,
        home=home,
        sessions=tmp_path / "sessions",
    )

    assert guard.warnings == ()
    assert guard.sandbox.profile is not None
    assert guard.sandbox.reason == "on (macOS sandbox-exec)"
    assert guard.sandbox.network is False
    assert str(home / "secrets") in guard.sandbox.profile  # the same protected paths as the file tools
    assert f'(subpath "{(tmp_path / "sessions").resolve()}")' in guard.sandbox.profile
    assert len(seen) == 1


def test_a_sandbox_disabled_in_the_global_settings_is_an_info_line_not_a_warning(tmp_path: Path) -> None:
    guard = make(tmp_path, {"guardrails": {"sandbox": {"enabled": False}}})

    assert guard.warnings == ()
    assert guard.info == ("sandbox: off: disabled in settings",)
    assert guard.sandbox.profile is None


def test_a_failing_probe_is_a_launch_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "sandbox-exec"
    exe.write_text("")
    monkeypatch.setattr(sandbox_module, "SANDBOX_EXEC", exe)
    home = tmp_path / "home"
    home.mkdir()

    guard = prepare(tmp_path / "ws", None, None, {}, "Darwin", lambda argv: False, home=home)

    assert guard.sandbox.profile is None
    assert len(guard.warnings) == 1 and "test command" in guard.warnings[0]


def test_prepare_builds_the_redactor_from_the_removed_variables_that_are_long_enough(tmp_path: Path) -> None:
    env = {"GITHUB_TOKEN": "a-long-enough-value", "SHORT_TOKEN": "short", "PATH": "/usr/local/bin:/usr/bin"}

    guard = make(tmp_path, env=env)

    assert guard.redactor.redact("x a-long-enough-value short /usr/local/bin:/usr/bin") == (
        "x [redacted:env:GITHUB_TOKEN] short /usr/local/bin:/usr/bin",
        1,
    )


def test_a_kept_variables_value_is_not_redacted_as_an_env_value(tmp_path: Path) -> None:
    guard = make(tmp_path, {"guardrails": {"keepEnv": ["NPM_TOKEN"]}}, None, {"NPM_TOKEN": "npm-value-12345"})

    assert guard.redactor.redact("npm-value-12345") == ("npm-value-12345", 0)


def test_redact_result_replaces_the_output_drops_the_colour_copy_and_counts_the_output_only(tmp_path: Path) -> None:
    guard = make(tmp_path)
    secret = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
    result = ToolResult(f"out {secret}", True, "exit 1", display=f"shown {secret} {secret}")

    redacted = guard.redact_result(result)

    # The display is a copy of the output with colours: with the output redacted it is dropped, so the transcript shows
    # the redacted plain text (a copy cannot be trusted to hold a secret in one piece, see the next tests).
    assert redacted == ToolResult("out [redacted:github-token]", True, "exit 1", display=None)
    assert guard.redactions == 1


def test_redact_result_returns_the_same_result_when_nothing_matches(tmp_path: Path) -> None:
    guard = make(tmp_path)
    result = ToolResult("plain", display="shown")

    assert guard.redact_result(result) is result
    assert guard.redactions == 0


def test_a_redactor_that_raises_withholds_the_result_and_leaks_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    guard = make(tmp_path)

    def broken(self: object, text: str) -> tuple[str, int]:
        raise RuntimeError(f"cannot handle {text}")

    monkeypatch.setattr(type(guard.redactor), "redact", broken)

    withheld = guard.redact_result(ToolResult("secret-text", False, "ok", display="secret-display"))

    assert withheld == ToolResult("[redaction failed: output withheld]", True, "redaction failed")
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


def test_redact_result_counts_a_secret_only_in_the_display_not_at_all(tmp_path: Path) -> None:
    guard = make(tmp_path)
    secret = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"

    redacted = guard.redact_result(ToolResult("clean", display=f"shown {secret}"))

    assert redacted == ToolResult("clean", display=None)
    assert guard.redactions == 0


GITHUB = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"


def test_a_secret_split_by_colour_codes_in_the_display_does_not_reach_the_transcript(tmp_path: Path) -> None:
    guard = make(tmp_path)
    colored = f"\x1b[31mghp_\x1b[0m{GITHUB[4:]}"
    assert GITHUB not in colored  # the display copy alone cannot be matched

    redacted = guard.redact_result(ToolResult(GITHUB, display=colored))

    assert redacted.output == "[redacted:github-token]"
    assert redacted.display is None
    assert guard.redactions == 1


def test_a_jwt_with_a_coloured_segment_in_the_display_does_not_reach_the_transcript(tmp_path: Path) -> None:
    guard = make(tmp_path)
    header, payload, signature = JWT.split(".")
    colored = f"{header}.\x1b[32m{payload}\x1b[0m.{signature}"

    redacted = guard.redact_result(ToolResult(f"token {JWT}", display=colored))

    assert redacted.output == "token [redacted:jwt]"
    assert redacted.display is None


def test_a_display_that_nothing_changed_is_kept(tmp_path: Path) -> None:
    guard = make(tmp_path)
    result = ToolResult("plain text", display="\x1b[32mplain text\x1b[0m")

    assert guard.redact_result(result) is result


def test_redact_preview_replaces_a_secret_in_a_diff_and_does_not_count_it(tmp_path: Path) -> None:
    guard = make(tmp_path)
    secret = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
    preview = Preview("diff", "edit a.txt", f"-token={secret}\n+token=x\n")

    shown = guard.redact_preview(preview)

    assert shown == Preview("diff", "edit a.txt", "-token=[redacted:github-token]\n+token=x\n")
    assert guard.redactions == 0


def test_redact_preview_leaves_what_the_model_wrote_itself(tmp_path: Path) -> None:
    guard = make(tmp_path)
    secret = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
    command = Preview("command", "bash echo", f"echo {secret}")
    content = Preview("content", "write a", secret)

    assert guard.redact_preview(command) is command  # the user approves exactly what runs
    assert guard.redact_preview(content) is content
    assert guard.redact_preview(None) is None


def test_a_redactor_that_raises_withholds_the_diff_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    guard = make(tmp_path)

    def broken(self: object, text: str) -> tuple[str, int]:
        raise RuntimeError(f"cannot handle {text}")

    monkeypatch.setattr(type(guard.redactor), "redact", broken)

    shown = guard.redact_preview(Preview("diff", "edit a", "-secret-text"))

    assert shown == Preview("diff", "edit a", "[redaction failed: output withheld]")


def test_redact_preview_bounds_a_huge_diff_and_says_what_it_did_not_show(tmp_path: Path) -> None:
    guard = make(tmp_path)
    line = "+some added line of the diff\n"
    body = line * 20_000  # 580 KB

    shown = guard.redact_preview(Preview("diff", "edit a", body))

    assert shown is not None
    assert len(shown.body) < guard_setup.PREVIEW_WINDOW + 200
    assert shown.body.startswith(line * 100)
    assert shown.body.endswith("characters not shown]")
    hidden = int(shown.body.rsplit("[", 1)[1].split(" ", 1)[0])
    assert hidden + len(shown.body.rsplit("\n[", 1)[0]) == len(body)


def test_redact_preview_of_a_huge_diff_does_not_leave_the_start_of_a_token_at_the_cut(tmp_path: Path) -> None:
    guard = make(tmp_path)
    body = "+x\n" * 50_000
    body = body[: guard_setup.PREVIEW_WINDOW - 10] + " " + GITHUB + "\n" + body

    shown = guard.redact_preview(Preview("diff", "edit a", body))

    assert shown is not None
    assert "ghp_" not in shown.body


def test_redact_preview_of_a_huge_adversarial_diff_is_fast(tmp_path: Path) -> None:
    import time

    guard = make(tmp_path, None, None, {"A_TOKEN": "value-number-0001-secret"})
    body = "[redacted:" * 200_000

    start = time.perf_counter()
    guard.redact_preview(Preview("diff", "edit a", body))

    assert time.perf_counter() - start < 1.0


def test_a_small_diff_preview_has_no_note(tmp_path: Path) -> None:
    guard = make(tmp_path)
    preview = Preview("diff", "edit a", "-a\n+b\n")

    assert guard.redact_preview(preview) is preview
