"""The built-in skill folder: loaded first, replaced by global and project skills of the same name."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.engine.skills.loader import load_skills


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root


def env(tmp_path: Path) -> dict[str, str]:
    return {"XDG_CONFIG_HOME": str(tmp_path / "cfg")}


def make(base: Path, name: str, body: str, description: str = "d") -> None:
    folder = base / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(f"---\ndescription: {description}\n---\n{body}\n")


def test_ac2_a_global_skill_replaces_a_builtin_one(ws: Path, tmp_path: Path) -> None:
    make(tmp_path / "b", "tdd", "builtin body")
    make(tmp_path / "cfg" / "agent" / "skills", "tdd", "global body")
    skills = load_skills(ws, env(tmp_path), builtin=tmp_path / "b")
    skill = skills.get("tdd")
    assert skill is not None and skill.source == "global" and skill.body.strip() == "global body"


def test_ac3_project_beats_global_beats_builtin(ws: Path, tmp_path: Path) -> None:
    make(tmp_path / "b", "commit", "builtin body")
    make(tmp_path / "cfg" / "agent" / "skills", "commit", "global body")
    make(ws / ".agent" / "skills", "commit", "project body")
    skill = load_skills(ws, env(tmp_path), builtin=tmp_path / "b").get("commit")
    assert skill is not None and skill.source == "project" and skill.body.strip() == "project body"


def test_a_builtin_skill_without_override_has_source_builtin(ws: Path, tmp_path: Path) -> None:
    make(tmp_path / "b", "debug", "x")
    skill = load_skills(ws, env(tmp_path), builtin=tmp_path / "b").get("debug")
    assert skill is not None and skill.source == "builtin" and skill.folder == (tmp_path / "b" / "debug").resolve()


def test_ac7_an_invalid_builtin_skill_warns_and_the_others_load(ws: Path, tmp_path: Path) -> None:
    make(tmp_path / "b", "good", "x")
    bad = tmp_path / "b" / "bad"
    bad.mkdir()
    (bad / "SKILL.md").write_text("---\nname: bad\n---\nbody\n")
    skills = load_skills(ws, env(tmp_path), builtin=tmp_path / "b")
    assert list(skills.by_name) == ["good"]
    assert len(skills.warnings) == 1 and "bad" in skills.warnings[0]


def test_a_missing_builtin_folder_is_silent(ws: Path, tmp_path: Path) -> None:
    skills = load_skills(ws, env(tmp_path), builtin=tmp_path / "nope")
    assert not skills.by_name and not skills.warnings
