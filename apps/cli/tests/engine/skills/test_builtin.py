"""The skills shipped with the agent: they parse, stay short and keep to the tools the agent has."""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from src.engine.skills.loader import DESCRIPTION_CHARS, FILE, builtin_skills_dir, load_skills, parse_skill

NAMES = ["commit", "debug", "tdd"]
FORBIDDEN = {"skill", "task", "web_fetch", "term_start", "generate_image"}


def body_of(name: str) -> str:
    return parse_skill((builtin_skills_dir() / name / FILE).read_text("utf-8"))[1]


def env(tmp_path: Path) -> dict[str, str]:
    return {"XDG_CONFIG_HOME": str(tmp_path / "cfg")}


def test_the_three_skills_load_with_source_builtin(tmp_path: Path) -> None:
    skills = load_skills(tmp_path / "ws", env(tmp_path), builtin=builtin_skills_dir())
    assert sorted(skills.by_name) == NAMES and not skills.warnings
    assert {s.source for s in skills.sorted()} == {"builtin"}
    assert all(s.folder.is_dir() and s.folder.name == s.name for s in skills.sorted())


@pytest.mark.parametrize("name", NAMES)
def test_ac5_front_matter_and_body(name: str) -> None:
    keys, body = parse_skill((builtin_skills_dir() / name / FILE).read_text("utf-8"))
    assert 0 < len(keys["description"]) <= DESCRIPTION_CHARS
    assert keys.get("name", name) == name
    assert body.strip()


TOOLS = {"read", "glob", "grep", "edit", "write", "bash", "todo"}
KNOWN = {
    # commit: types, git flags and file names named in the text
    "feat", "fix", "docs", "refactor", "perf", "test", "build", "ci", "chore",
    "--force", "--no-verify", ".env",
    # commit: command words and placeholders
    "$", "type(scope):", "'\\''",
}


@pytest.mark.parametrize("name", NAMES)
def test_ac6_bodies_name_only_known_words_in_backticks(name: str) -> None:
    ticked = set(re.findall(r"`(\S+)`", body_of(name)))
    assert not ticked - TOOLS - KNOWN


@pytest.mark.parametrize("name", NAMES)
def test_ac6_bodies_name_no_unavailable_tool(name: str) -> None:
    assert not set(re.findall(r"`([a-z_]+)`", body_of(name))) & FORBIDDEN


def test_ac6_commit_content() -> None:
    body = body_of("commit").lower()
    for phrase in ("ask", "push", "amend", "--no-verify", "git commit -f", "single quotes", "never amend", "secret"):
        assert phrase in body


def test_ac6_tdd_content() -> None:
    body = body_of("tdd").lower()
    for phrase in ("see it fail", "smallest change"):
        assert phrase in body
    assert re.search(r"never weaken, skip or delete a test", body)


def test_ac6_debug_content() -> None:
    body = body_of("debug").lower()
    for phrase in ("reproduce", "root cause", "failing test", "stop"):
        assert phrase in body
    assert body.index("reproduce") < body.index("root cause") < body.index("any `edit`")
    assert "report" in body


def test_ac11_the_section_stays_small(tmp_path: Path) -> None:
    section = load_skills(tmp_path / "ws", env(tmp_path), builtin=builtin_skills_dir()).section()
    assert section is not None
    header = section.splitlines()[0]
    assert len(section) - len(header) <= 600


def test_nfr2_loading_them_is_fast(tmp_path: Path) -> None:
    ws, e, folder = tmp_path / "ws", env(tmp_path), builtin_skills_dir()

    def timed() -> float:
        start = time.perf_counter()
        load_skills(ws, e, builtin=folder)
        return time.perf_counter() - start

    assert min(timed() for _ in range(20)) < 0.005
