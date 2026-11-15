"""/skills, its launch warnings, and its place among the built-in commands."""

from __future__ import annotations

from pathlib import Path

from src.engine.commands.loader import load_commands
from src.engine.skills.loader import Skill, Skills
from src.tui import app as app_module
from src.tui.widgets.transcript import NoticeEntry

from tests.tui.conftest import AppFactory, scripted, type_and_send, until


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _skills(*items: Skill, warnings: tuple[str, ...] = ()) -> Skills:
    return Skills({s.name: s for s in items}, warnings)


def _skill(name: str, description: str, source: str = "global") -> Skill:
    return Skill(name, description, "body", source, Path("/x") / name)  # ty: ignore[invalid-argument-type]


async def test_skills_lists_each_skill_in_name_order_then_the_warnings(make_app: AppFactory) -> None:
    app = make_app(
        skills=_skills(
            _skill("review", "House code review checklist."),
            _skill("release", "Steps to cut a release.\nSecond line", "project"),
            _skill("evil", "bad\x1b[31m text"),
            warnings=("x/SKILL.md: no front matter",),
        )
    )
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/skills")
        await pilot.pause()
        (text,) = [n for n in _notices(app) if "(global)" in n]

    assert text.removeprefix("\u2139 ").splitlines() == [
        "evil (global): bad\\x1b[31m text",
        "release (project): Steps to cut a release. Second line",
        "review (global): House code review checklist.",
        "x/SKILL.md: no front matter",
    ]
    assert scripted(app).requests == []


async def test_skills_without_any_prints_no_skills(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/skills")
        await pilot.pause()
        assert any(n.endswith("No skills.") for n in _notices(app))


async def test_launch_warnings_about_skills_are_shown(make_app: AppFactory) -> None:
    app = make_app(skills=_skills(warnings=("a/SKILL.md: no front matter",)))
    async with app.run_test() as pilot:
        await until(pilot, lambda: any("a/SKILL.md: no front matter" in n for n in _notices(app)))
        assert any(n.startswith("⚠") and "no front matter" in n for n in _notices(app))


async def test_help_lists_skills(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/help")
        await pilot.pause()
        (help_text,) = [n for n in _notices(app) if "Enter" in n]
    assert "/skills" in help_text


def test_a_custom_command_named_skills_is_skipped_as_a_builtin(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg" / "agent" / "commands"
    cfg.mkdir(parents=True)
    (cfg / "skills.md").write_text("Do X")
    loaded = load_commands(tmp_path, set(app_module.BUILTINS), {"XDG_CONFIG_HOME": str(tmp_path / "cfg")})
    assert loaded.get("skills") is None and any("built-in" in w for w in loaded.warnings)


async def test_skills_shows_non_ascii_descriptions_readable_and_writes_out_bidi(make_app: AppFactory) -> None:
    app = make_app(skills=_skills(_skill("ru", "Проверка кода \u202e")))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/skills")
        await pilot.pause()
        (text,) = [n for n in _notices(app) if "(global)" in n]

    assert text.removeprefix("\u2139 ") == "ru (global): Проверка кода \\u202e"
