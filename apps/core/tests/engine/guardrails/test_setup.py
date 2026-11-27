"""`prepare` and `Guardrails`: the launch value that tools, the loop and the spawn sites share."""

from __future__ import annotations

from pathlib import Path

import pytest
from core.engine.guardrails import setup as guard_setup
from core.engine.guardrails.setup import Guardrails, prepare
from core.engine.tools.base import Preview, ToolResult


def make(tmp_path: Path, global_data: dict | None = None, project_data: dict | None = None, env: dict | None = None) -> Guardrails:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return prepare(tmp_path / "ws", global_data, project_data, env or {}, home=home)


def test_prepare_has_no_sandbox_and_only_the_removed_key_warning(tmp_path: Path) -> None:
    guard = make(tmp_path, {"guardrails": {"sandbox": {"enabled": True}}})

    assert guard.warnings == ("global settings: guardrails.sandbox is no longer supported and is ignored",)
    assert not hasattr(guard, "sandbox") and not hasattr(guard, "info")


def test_redact_result_replaces_the_output_drops_the_colour_copy_and_counts_the_output_only(tmp_path: Path) -> None:
    guard = make(tmp_path)
    secret = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
    result = ToolResult(f"out {secret}", True, "exit 1", display=f"shown {secret} {secret}")

    redacted = guard.redact_result(result)

    # The display is a copy of the output with colours: with the output redacted it is dropped, so the transcript shows
    # the redacted plain text (a copy cannot be trusted to hold a secret in one piece, see the next tests).
    assert redacted == ToolResult("out [redacted:github-token]", True, "exit 1", display=None)
    assert guard.redactions == 1


GITHUB = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"


def test_a_redactor_that_raises_withholds_the_diff_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    guard = make(tmp_path)

    def broken(self: object, text: str) -> tuple[str, int]:
        raise RuntimeError(f"cannot handle {text}")

    monkeypatch.setattr(type(guard.redactor), "redact", broken)

    shown = guard.redact_preview(Preview("diff", "edit a", "-secret-text"))

    assert shown == Preview("diff", "edit a", "[redaction failed: output withheld]")


def test_redact_preview_of_a_huge_diff_does_not_leave_the_start_of_a_token_at_the_cut(tmp_path: Path) -> None:
    guard = make(tmp_path)
    body = "+x\n" * 50_000
    body = body[: guard_setup.PREVIEW_WINDOW - 10] + " " + GITHUB + "\n" + body

    shown = guard.redact_preview(Preview("diff", "edit a", body))

    assert shown is not None
    assert "ghp_" not in shown.body
