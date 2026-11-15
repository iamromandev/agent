"""/guardrails: what the guardrails do in this run, with names only (R17, NFR1)."""

from __future__ import annotations

from pathlib import Path

from src.engine.guardrails.paths import MAX_TRACKED_FILES
from src.engine.guardrails.sandbox import Sandbox
from src.engine.guardrails.setup import Guardrails, prepare
from src.engine.provider.base import Done, TextDelta
from src.engine.session import sessions_root
from src.tui import app as app_module
from src.tui.widgets.transcript import AssistantEntry, NoticeEntry

from tests.tui.conftest import AppFactory, scripted, type_and_send, until

SECRET = "ghp_thisIsTheSecretValue0123456789"


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _guard(tmp_path: Path, global_data: dict | None = None, project_data: dict | None = None, env: dict | None = None) -> Guardrails:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return prepare(tmp_path, global_data, project_data, env or {}, "Darwin", home=home)


async def _guardrails_text(app) -> str:
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/guardrails")
        await until(pilot, lambda: any("Sandbox:" in n for n in _notices(app)))
        (text,) = [n for n in _notices(app) if "Sandbox:" in n]
    return text.removeprefix("\u2139 ")


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
    assert lines[0] == "Sandbox: not built yet"
    assert lines[1].startswith("Protected: ~/.ssh, ~/.aws, ~/.gnupg, ~/.config/gh, ")
    assert lines[1].endswith("id_ecdsa* (+1 from settings)")
    assert ", .env*, *.pem, *.key, id_rsa*, " in lines[1]
    assert ".env.*" not in lines[1]
    assert lines[2] == "Removed from child environment: AWS_SECRET_ACCESS_KEY, GITHUB_TOKEN"
    assert lines[3] == "Redacted this run: 0"
    assert len(lines) == 4
    assert scripted(app).requests == []


async def test_guardrails_never_shows_a_value_and_no_value_reaches_the_session_file(
    make_app: AppFactory, tmp_path: Path
) -> None:
    guard = _guard(tmp_path, env={"GITHUB_TOKEN": SECRET})
    app = make_app([[TextDelta("hi"), Done(1, 1)]], guard=guard)

    async with app.run_test() as pilot:
        await type_and_send(pilot, "hello")
        await until(pilot, lambda: any(e.text == "hi" for e in app.query(AssistantEntry)))
        await type_and_send(pilot, "/guardrails")
        await until(pilot, lambda: any("Sandbox:" in n for n in _notices(app)))
        transcript = "\n".join(_notices(app))

    assert "GITHUB_TOKEN" in transcript
    assert SECRET not in transcript
    sessions = list(sessions_root(None).rglob("*.jsonl"))
    assert sessions
    assert all(SECRET not in p.read_text() for p in sessions)


async def test_guardrails_without_removed_variables_says_none(make_app: AppFactory, tmp_path: Path) -> None:
    app = make_app(guard=_guard(tmp_path, env={"PATH": "/bin"}))

    text = await _guardrails_text(app)

    assert "Removed from child environment: none" in text.splitlines()


async def test_guardrails_escapes_names_that_come_from_the_environment(make_app: AppFactory, tmp_path: Path) -> None:
    app = make_app(guard=_guard(tmp_path, env={"BAD\x1b[31m_TOKEN": "x"}))

    text = await _guardrails_text(app)

    assert "BAD\\x1b[31m_TOKEN" in text
    assert "\x1b" not in text


async def test_guardrails_shows_network_and_writable_folders_when_a_sandbox_is_on(
    make_app: AppFactory, tmp_path: Path
) -> None:
    guard = _guard(tmp_path)
    guard.sandbox = Sandbox("(version 1)", "on (macOS sandbox-exec)", False, (tmp_path, Path("/tmp/a\x1bb")))
    app = make_app(guard=guard)

    text = await _guardrails_text(app)

    lines = text.splitlines()
    assert lines[0] == "Sandbox: on (macOS sandbox-exec), network off"
    assert lines[1] == f"Writable: {tmp_path}, /tmp/a\\x1bb"


async def test_guardrails_counts_redactions(make_app: AppFactory, tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    guard.redactions = 3
    app = make_app(guard=guard)

    assert "Redacted this run: 3" in (await _guardrails_text(app)).splitlines()


async def test_guardrails_without_a_guard_uses_the_defaults(make_app: AppFactory) -> None:
    text = await _guardrails_text(make_app())

    assert text.splitlines()[0].startswith("Sandbox:")
    assert "(+0 from settings)" not in text


async def test_help_lists_guardrails(make_app: AppFactory) -> None:
    assert "/guardrails" in app_module.HELP
    app = make_app()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/help")
        await pilot.pause()
        (help_text,) = [n for n in _notices(app) if "Enter" in n]
    assert "/guardrails" in help_text
    assert app_module.BUILTINS["guardrails"] == "sandbox, protected paths and redaction"


async def test_guardrails_says_when_hard_link_tracking_stopped_at_its_bound(make_app: AppFactory, tmp_path: Path) -> None:
    home = tmp_path / "home"
    folder = home / ".aws"
    folder.mkdir(parents=True)
    for index in range(MAX_TRACKED_FILES + 1):
        (folder / f"f{index}").write_text("x")
    guard = prepare(tmp_path, None, None, {}, "Darwin", home=home)
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


async def test_guardrails_does_not_list_homes_when_there_is_one(make_app: AppFactory, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    app = make_app(guard=prepare(tmp_path, None, None, {}, "Darwin", home=home))

    assert "Home folders" not in await _guardrails_text(app)
