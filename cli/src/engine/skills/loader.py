"""Finding and reading skills. Files come from a repository: nothing in them is run."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.engine.commands.loader import NAME, Source, front_matter
from src.engine.fileio import read_regular
from src.engine.permissions.setup import MAX_WARNINGS
from src.engine.text import escape, one_line

MAX_BYTES = 64 * 1024
DESCRIPTION_CHARS = 200
SECTION_CHARS = 8_000
FILE = "SKILL.md"
_HEADER = (
    "Available skills (call the skill tool with a name to load its instructions "
    "before doing a task its description matches):"
)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str  # raw from the file, not escaped
    body: str
    source: Source
    folder: Path  # absolute


@dataclass(frozen=True)
class Skills:
    by_name: Mapping[str, Skill]
    warnings: tuple[str, ...]  # already escaped and capped

    def get(self, name: str) -> Skill | None:
        return self.by_name.get(name)

    def sorted(self) -> list[Skill]:
        return [self.by_name[name] for name in sorted(self.by_name)]

    def section(self) -> str | None:
        """The "Available skills" prompt text, at most `SECTION_CHARS` long; `None` when there are no skills."""
        skills = self.sorted()
        if not skills:
            return None
        lines = [f"- {s.name}: {one_line(s.description, DESCRIPTION_CHARS)}" for s in skills]
        if len(_HEADER) + sum(1 + len(line) for line in lines) <= SECTION_CHARS:
            return "\n".join([_HEADER, *lines])
        budget = SECTION_CHARS - len(f"\nand {len(lines)} more skills (see /skills)")  # the longest tail
        kept: list[str] = []
        total = len(_HEADER)
        for line in lines:
            if total + 1 + len(line) > budget:
                break
            kept.append(line)
            total += 1 + len(line)
        return "\n".join([_HEADER, *kept, f"and {len(lines) - len(kept)} more skills (see /skills)"])


def global_skills_dir(env: Mapping[str, str] = os.environ) -> Path:
    """`$XDG_CONFIG_HOME/agent/skills`, else `~/.config/agent/skills`. A relative XDG_CONFIG_HOME is ignored."""
    config_home = env.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home and Path(config_home).is_absolute() else Path.home() / ".config"
    return base / "agent" / "skills"


def project_skills_dir(workspace: Path) -> Path:
    return workspace / ".agent" / "skills"


def parse_skill(text: str) -> tuple[dict[str, str], str]:
    """`(keys, body)` of a `SKILL.md`. Raises `ValueError` without front matter or when it is never closed."""
    keys, body = front_matter(text)
    if keys is None:
        raise ValueError("no front matter")
    return keys, body


def _read_file(folder: Path, source: Source) -> bytes:
    if source == "global":
        return read_regular(folder / FILE, MAX_BYTES)
    # The folder is opened without following a symlink and SKILL.md inside that open directory, so neither can be
    # swapped for a link between a check and the read.
    dir_fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        return read_regular(Path(FILE), MAX_BYTES, follow=False, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


def _read(folder: Path, name: str, source: Source) -> Skill:
    data = _read_file(folder, source)
    keys, body = parse_skill(data.decode("utf-8"))
    if not keys.get("description"):
        raise ValueError("description is missing or empty")
    if "name" in keys and keys["name"] != name:
        raise ValueError(f"name '{keys['name']}' does not match the folder name '{name}'")
    shown = folder.resolve() if source == "global" else folder.absolute()
    return Skill(name, keys["description"], body, source, shown)


def _load_dir(directory: Path, source: Source, found: dict[str, Skill], warnings: list[str]) -> None:
    try:
        entries = sorted(directory.iterdir())
    except FileNotFoundError:
        return
    except OSError as exc:
        warnings.append(f"{directory}: {exc.strerror or exc}")
        return
    for path in entries:
        if source == "project" and path.is_symlink():
            warnings.append(f"{path}: a symlink is not followed in the project folder")
        elif not path.is_dir():
            continue
        elif not NAME.fullmatch(path.name):
            warnings.append(f"{path}: not a valid skill name")
        else:
            try:
                found[path.name] = _read(path, path.name, source)
            except FileNotFoundError:
                continue  # a folder without SKILL.md is not a skill
            except (OSError, ValueError) as exc:  # UnicodeDecodeError is a ValueError
                reason = exc.strerror or str(exc) if isinstance(exc, OSError) else str(exc)
                warnings.append(f"{path / FILE}: {reason}")


def load_skills(workspace: Path, env: Mapping[str, str] = os.environ) -> Skills:
    """Read both folders once. A project skill replaces a global one; problems become warnings."""
    found: dict[str, Skill] = {}
    warnings: list[str] = []
    _load_dir(global_skills_dir(env), "global", found, warnings)
    project = project_skills_dir(workspace)
    if project.parent.is_symlink() or project.is_symlink():
        warnings.append(f"{project}: a symlink is not followed in the project folder")
    else:
        _load_dir(project, "project", found, warnings)
    shown = [escape(w) for w in warnings]
    if len(shown) > MAX_WARNINGS:
        shown = [*shown[:MAX_WARNINGS], f"and {len(shown) - MAX_WARNINGS} more warnings"]
    return Skills(found, tuple(shown))
