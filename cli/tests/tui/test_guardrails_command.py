"""/guardrails: what the guardrails do in this run, with names only (R17, NFR1)."""

from __future__ import annotations

from pathlib import Path

from src.engine.guardrails.paths import MAX_TRACKED_FILES
from src.engine.guardrails.setup import Guardrails, prepare
from src.tui.widgets.transcript import NoticeEntry

from tests.tui.conftest import AppFactory, scripted, type_and_send, until

SECRET = "ghp_thisIsTheSecretValue0123456789"


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _guard(tmp_path: Path, global_data: dict | None = None, project_data: dict | None = None, env: dict | None = None) -> Guardrails:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return prepare(tmp_path, global_data, project_data, env or {}, home=home)


async def _guardrails_text(app) -> str:
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/guardrails")
        await until(pilot, lambda: any("Protected:" in n for n in _notices(app)))
        (text,) = [n for n in _notices(app) if "Protected:" in n]
    return text.removeprefix(chr(0x2139) + " ")  # the info-notice marker


async def test_guardrails_shows_the_layout_for_this_run(make_app: AppFactory, tmp_path: Path) -> None:
    guard = _guard(
        tmp_path,
        {"guardrails": {"protected": ["~/secrets/**"]}},
        None,
        {"GITHUB_TOKEN": SECRET, "AWS_SECRET_ACCESS_KEY": "another-secret-value", "PATH": "/bin"},
    )
    app = make_app(guard=guard)

    text = await _guardrails_text(app)

    lines = text.splitlines()
    assert lines[0].startswith("Protected: ~/.ssh, ~/.aws, ~/.gnupg, ~/.config/gh, ")
    assert lines[0].endswith("id_ecdsa* (+1 from settings)")
    assert ", .env*, *.pem, *.key, id_rsa*, " in lines[0]
    assert ".env.*" not in lines[0]
    assert lines[1] == "Removed from child environment: AWS_SECRET_ACCESS_KEY, GITHUB_TOKEN"
    assert lines[2] == "Redacted this run: 0"
    assert len(lines) == 3
    assert "andbox" not in text
    assert scripted(app).requests == []


async def test_guardrails_says_when_hard_link_tracking_stopped_at_its_bound(make_app: AppFactory, tmp_path: Path) -> None:
    home = tmp_path / "home"
    folder = home / ".aws"
    folder.mkdir(parents=True)
    for index in range(MAX_TRACKED_FILES + 1):
        (folder / f"f{index}").write_text("x")
    guard = prepare(tmp_path, None, None, {}, home=home)
    app = make_app(guard=guard)

    text = await _guardrails_text(app)

    assert f"Hard links to protected files are tracked for the first {MAX_TRACKED_FILES:,} files only" in text


async def test_guardrails_shows_both_homes_when_there_are_two(make_app: AppFactory, tmp_path: Path) -> None:
    home, other = tmp_path / "home", tmp_path / "account-home"
    home.mkdir()
    other.mkdir()
    from src.engine.guardrails import setup as guard_setup

    guard = guard_setup._build(guard_setup.GuardConfig(), home, {}, also_home=other)
    app = make_app(guard=guard)

    text = await _guardrails_text(app)

    assert f"Home folders: {home}, {other}" in text
