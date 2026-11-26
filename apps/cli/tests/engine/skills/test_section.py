"""The "Available skills" prompt section: one-line escaped descriptions inside a fixed budget."""

from __future__ import annotations

from pathlib import Path

from src.engine.skills.loader import SECTION_CHARS, Skill, Skills


def skill(name: str, description: str) -> Skill:
    return Skill(name, description, "body", "global", Path("/x") / name)


def skills(*items: Skill) -> Skills:
    return Skills({s.name: s for s in items}, ())


def test_no_skills_gives_no_section() -> None:
    assert skills().section() is None


def test_lists_skills_in_name_order() -> None:
    text = skills(skill("review", "House checklist."), skill("release", "Cut a release.")).section()
    assert text is not None
    lines = text.splitlines()
    assert lines[0].startswith("Available skills (call the skill tool with a name")
    assert lines[1:] == ["- release: Cut a release.", "- review: House checklist."]


def test_ac8_nfr3_many_long_skills_stay_in_budget() -> None:
    long = "\n".join(["word " * 20] * 3)  # multi-line, ~300 characters
    text = skills(*(skill(f"s{i:03}", long) for i in range(500))).section()
    assert text is not None and len(text) <= SECTION_CHARS
    lines = text.splitlines()
    assert lines[-1].startswith("and ") and lines[-1].endswith(" more skills (see /skills)")
    shown = len(lines) - 2
    assert lines[-1] == f"and {500 - shown} more skills (see /skills)"
    for line in lines[1:-1]:
        assert len(line.split(": ", 1)[1]) <= 200  # cut at 200 characters, the "…" included
    assert lines[1].endswith("\u2026")
