"""Loading skills from the global and project folders: names, front matter, limits, symlinks, warnings."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from src.engine.skills import loader
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


def test_missing_folders_give_no_skills_and_no_warning(ws: Path, home: Path) -> None:
    skills = load_skills(ws, env(home))
    assert not skills.by_name and skills.warnings == ()


def test_ac2_front_matter_name_must_equal_folder(ws: Path, home: Path) -> None:
    make(gdir(home), "review", "---\nname: other\ndescription: d\n---\nb")
    make(gdir(home), "ok", "---\nname: ok\ndescription: d\n---\nb")

    skills = load_skills(ws, env(home))

    assert list(skills.by_name) == ["ok"]
    assert len(skills.warnings) == 1 and "other" in skills.warnings[0]


@pytest.mark.parametrize("text", ["no front matter", "---\nname: x\n---\nb", "---\ndescription:   \n---\nb"])
def test_ac3_missing_front_matter_or_description_is_skipped(ws: Path, home: Path, text: str) -> None:
    make(gdir(home), "x", text)
    skills = load_skills(ws, env(home))
    assert not skills.by_name and len(skills.warnings) == 1


def test_ac3_body_is_everything_after_closing_marker(ws: Path, home: Path) -> None:
    make(gdir(home), "x", "---\ndescription: d\nother: 1\n---\nline1\n---\nline2\n")
    skill = load_skills(ws, env(home)).get("x")
    assert skill is not None and skill.body == "line1\n---\nline2\n" and skill.description == "d"


def test_ac4_project_replaces_global(ws: Path, home: Path) -> None:
    make(gdir(home), "release", description="global")
    make(pdir(ws), "release", description="project")
    skill = load_skills(ws, env(home)).get("release")
    assert skill is not None and skill.source == "project" and skill.description == "project"
    assert skill.folder == (pdir(ws) / "release").absolute()


def test_folder_is_absolute_for_global(ws: Path, home: Path) -> None:
    make(gdir(home), "g")
    skill = load_skills(ws, env(home)).get("g")
    assert skill is not None and skill.source == "global" and skill.folder.is_absolute()


def test_ac5_too_big_and_non_utf8_are_refused(ws: Path, home: Path) -> None:
    make(gdir(home), "big", b"---\ndescription: d\n---\n" + b"x" * (65 * 1024))
    make(gdir(home), "bin", b"---\ndescription: d\n---\n\xff\xfe")
    skills = load_skills(ws, env(home))
    assert not skills.by_name and len(skills.warnings) == 2


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


def test_ac5_symlinked_project_agent_dir_is_refused(ws: Path, home: Path, tmp_path: Path) -> None:
    make(tmp_path / "other" / "skills", "x")
    (ws / ".agent").symlink_to(tmp_path / "other")
    skills = load_skills(ws, env(home))
    assert not skills.by_name and len(skills.warnings) == 1


def test_ac5_global_symlinked_skill_folder_is_loaded(ws: Path, home: Path, tmp_path: Path) -> None:
    real = make(tmp_path / "elsewhere", "lnk")
    gdir(home).mkdir(parents=True)
    (gdir(home) / "lnk").symlink_to(real)
    skills = load_skills(ws, env(home))
    assert list(skills.by_name) == ["lnk"] and not skills.warnings
    assert skills.by_name["lnk"].folder == real.resolve()


def test_ac6_warnings_are_capped_and_escaped(ws: Path, home: Path) -> None:
    for i in range(24):
        make(gdir(home), f"s{i:02}", "no front matter")
    make(gdir(home), "ev\x1bil", "x")
    skills = load_skills(ws, env(home))
    assert len(skills.warnings) == 21
    assert skills.warnings[-1] == "and 5 more warnings"
    assert all("\x1b" not in w for w in skills.warnings)


def test_relative_xdg_config_home_is_ignored(ws: Path, home: Path) -> None:
    assert loader.global_skills_dir({"XDG_CONFIG_HOME": "rel"}) == Path.home() / ".config" / "agent" / "skills"
    assert loader.global_skills_dir(env(home)) == gdir(home)


def test_nfr1_only_skill_md_is_opened(ws: Path, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = make(gdir(home), "x")
    (folder / "big.bin").write_bytes(b"0" * 1_000_000)
    (folder / "a" / "b").mkdir(parents=True)
    (folder / "a" / "b" / "SKILL.md").write_text("---\ndescription: d\n---\n")
    opened: list[Path] = []
    real = loader.read_regular

    def spy(path: Path, *args: object, **kwargs: object) -> bytes:
        opened.append(path)
        return real(path, *args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(loader, "read_regular", spy)

    load_skills(ws, env(home))

    assert [p.name for p in opened] == ["SKILL.md"] and opened[0].parent.name == "x"


def test_nfr1_hundred_skills_load_quickly(ws: Path, home: Path) -> None:
    for i in range(100):
        make(gdir(home), f"s{i}")
    start = time.perf_counter()
    skills = load_skills(ws, env(home))
    assert len(skills.by_name) == 100 and time.perf_counter() - start < 0.2


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


def test_a_project_skill_that_fails_to_load_leaves_the_global_one_of_the_same_name(ws: Path, home: Path) -> None:
    make(gdir(home), "release", description="global")
    make(pdir(ws), "release", "no front matter")

    skills = load_skills(ws, env(home))

    skill = skills.get("release")
    assert skill is not None and skill.source == "global" and skill.description == "global"
    assert len(skills.warnings) == 1 and "release" in skills.warnings[0]
