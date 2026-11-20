"""/skills, its launch warnings, and its place among the built-in commands."""

from __future__ import annotations

from pathlib import Path

from src.engine.skills.loader import Skill, Skills
from src.tui.widgets.transcript import NoticeEntry

from tests.tui.conftest import AppFactory, type_and_send, until


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _skills(*items: Skill, warnings: tuple[str, ...] = ()) -> Skills:
    return Skills({s.name: s for s in items}, warnings)


def _skill(name: str, description: str, source: str = "global") -> Skill:
    return Skill(name, description, "body", source, Path("/x") / name)  # ty: ignore[invalid-argument-type]


async def test_launch_warnings_about_skills_are_shown(make_app: AppFactory) -> None:
    app = make_app(skills=_skills(warnings=("a/SKILL.md: no front matter",)))
    async with app.run_test() as pilot:
        await until(pilot, lambda: any("a/SKILL.md: no front matter" in n for n in _notices(app)))
        assert any(n.startswith("⚠") and "no front matter" in n for n in _notices(app))


async def test_skills_shows_non_ascii_descriptions_readable_and_writes_out_bidi(make_app: AppFactory) -> None:
    app = make_app(skills=_skills(_skill("ru", "Проверка кода \u202e")))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/skills")
        await pilot.pause()
        (text,) = [n for n in _notices(app) if "(global)" in n]

    assert text.removeprefix("\u2139 ") == "ru (global): Проверка кода \\u202e"
