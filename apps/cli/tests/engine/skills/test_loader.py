"""Loading skills from the global and project folders: names, front matter, limits, symlinks, warnings."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.engine.skills.loader import load_skills


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "cfg"


def env(home: Path) -> dict[str, str]:
    return {"XDG_CONFIG_HOME": str(home)}


def make(base: Path, name: str, text: str | bytes | None = None, description: str = "does x") -> Path:
    folder = base / name
    folder.mkdir(parents=True)
    if text is None:
        text = f"---\ndescription: {description}\n---\nbody of {name}\n"
    (folder / "SKILL.md").write_bytes(text if isinstance(text, bytes) else text.encode())
    return folder


def gdir(home: Path) -> Path:
    return home / "agent" / "skills"


def pdir(ws: Path) -> Path:
    return ws / ".agent" / "skills"


def test_ac1_loads_valid_ignores_folder_without_skill_md_and_warns_on_bad_name(ws: Path, home: Path) -> None:
    make(gdir(home), "a")
    (gdir(home) / "b").mkdir()
    make(gdir(home), "bad name")
    (gdir(home) / "file.txt").write_text("x")

    skills = load_skills(ws, env(home))

    assert list(skills.by_name) == ["a"]
    assert len(skills.warnings) == 1 and "bad name" in skills.warnings[0]


def test_ac2_front_matter_name_must_equal_folder(ws: Path, home: Path) -> None:
    make(gdir(home), "review", "---\nname: other\ndescription: d\n---\nb")
    make(gdir(home), "ok", "---\nname: ok\ndescription: d\n---\nb")

    skills = load_skills(ws, env(home))

    assert list(skills.by_name) == ["ok"]
    assert len(skills.warnings) == 1 and "other" in skills.warnings[0]


@pytest.mark.parametrize("text", ["---\nname: x\n---\nb"])
def test_ac3_missing_front_matter_or_description_is_skipped(ws: Path, home: Path, text: str) -> None:
    make(gdir(home), "x", text)
    skills = load_skills(ws, env(home))
    assert not skills.by_name and len(skills.warnings) == 1


def test_ac5_project_symlinks_are_refused(ws: Path, home: Path, tmp_path: Path) -> None:
    real = make(tmp_path / "elsewhere", "lnk")
    pdir(ws).mkdir(parents=True)
    (pdir(ws) / "lnk").symlink_to(real)  # skill folder symlink
    folder = make(pdir(ws), "file")
    (folder / "SKILL.md").unlink()
    (folder / "SKILL.md").symlink_to(real / "SKILL.md")

    skills = load_skills(ws, env(home))

    assert not skills.by_name and len(skills.warnings) == 2


def test_ac5_symlinked_project_skills_dir_is_refused(ws: Path, home: Path, tmp_path: Path) -> None:
    make(tmp_path / "other", "x")
    (ws / ".agent").mkdir()
    (ws / ".agent" / "skills").symlink_to(tmp_path / "other")
    skills = load_skills(ws, env(home))
    assert not skills.by_name and len(skills.warnings) == 1


def test_ac6_warnings_are_capped_and_escaped(ws: Path, home: Path) -> None:
    for i in range(24):
        make(gdir(home), f"s{i:02}", "no front matter")
    make(gdir(home), "ev\x1bil", "x")
    skills = load_skills(ws, env(home))
    assert len(skills.warnings) == 21
    assert skills.warnings[-1] == "and 5 more warnings"
    assert all("\x1b" not in w for w in skills.warnings)


def test_ac5_a_project_folder_swapped_for_a_symlink_after_the_check_is_refused_by_the_open(
    ws: Path, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = make(tmp_path / "elsewhere", "lnk")
    pdir(ws).mkdir(parents=True)
    (pdir(ws) / "lnk").symlink_to(real)
    link = pdir(ws) / "lnk"
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: False if self == link else original(self))  # the check passes

    skills = load_skills(ws, env(home))

    assert not skills.by_name and len(skills.warnings) == 1 and "lnk" in skills.warnings[0]
