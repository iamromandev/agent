"""Finding, reading and expanding custom commands. Files come from a repository: nothing in them is run."""

from __future__ import annotations

import os
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from core.engine.fileio import read_regular
from core.engine.permissions.setup import MAX_WARNINGS
from core.engine.text import escape

Source = Literal["global", "project"]
NAME = re.compile(r"[A-Za-z0-9_-]+")
MAX_BYTES = 64 * 1024
ARGUMENTS = "$ARGUMENTS"


@dataclass(frozen=True)
class Command:
    name: str
    body: str
    description: str | None
    source: Source

    def expand(self, rest: str) -> str:
        """The prompt for `/name rest`: `$ARGUMENTS` replaced, or `rest` appended after a blank line."""
        rest = rest.strip()
        if ARGUMENTS in self.body:
            text = self.body.replace(ARGUMENTS, rest)
        elif rest and not self.body.strip():  # a blank body: the arguments are the prompt
            text = rest
        elif rest:
            text = f"{self.body.rstrip()}\n\n{rest}"
        else:
            text = self.body
        return text.rstrip()


@dataclass(frozen=True)
class Commands:
    by_name: Mapping[str, Command]
    warnings: tuple[str, ...]
    held: tuple[str, ...] = ()  # global commands not loaded: the config watch holds the global config until trusted

    def get(self, name: str) -> Command | None:
        return self.by_name.get(name)

    def sorted(self) -> list[Command]:
        return [self.by_name[name] for name in sorted(self.by_name)]


def global_commands_dir(env: Mapping[str, str] = os.environ) -> Path:
    """`$XDG_CONFIG_HOME/agent/commands`, else `~/.config/agent/commands`. A relative XDG_CONFIG_HOME is ignored."""
    config_home = env.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home and Path(config_home).is_absolute() else Path.home() / ".config"
    return base / "agent" / "commands"


def project_commands_dir(workspace: Path) -> Path:
    return workspace / ".agent" / "commands"


def front_matter(text: str) -> tuple[dict[str, str] | None, str]:
    """`(keys, body)`: the `name:`, `description:`, `tools:` and `model:` lines of a leading `---` block. `None` keys when there is no block.

    Raises `ValueError` when the block is opened and never closed.
    """
    lines = text.removeprefix("\ufeff").replace("\r\n", "\n").split("\n")
    if lines[0].rstrip() != "---":
        return None, "\n".join(lines)
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].rstrip() == "---")
    except StopIteration:
        raise ValueError("front matter is not closed") from None
    keys: dict[str, str] = {}
    for line in lines[1:end]:
        for key in ("name", "description", "tools", "model"):
            if line.startswith(f"{key}:"):
                keys[key] = line[len(key) + 1 :].strip()
    return keys, "\n".join(lines[end + 1 :])


def parse(text: str) -> tuple[str | None, str]:
    """`(description, body)` of a command file. Raises `ValueError` when front matter is opened and never closed."""
    keys, body = front_matter(text)
    return (keys or {}).get("description"), body


def _read(path: Path, name: str, source: Source) -> Command:
    data = read_regular(path, MAX_BYTES, follow=source == "global")  # a project file is never followed, even if swapped
    description, body = parse(data.decode("utf-8"))
    return Command(name, body, description, source)


def _load_dir(
    directory: Path, source: Source, builtins: Collection[str], found: dict[str, Command], warnings: list[str]
) -> None:
    try:
        entries = sorted(directory.iterdir())
    except FileNotFoundError:
        return
    except OSError as exc:
        warnings.append(f"{directory}: {exc.strerror or exc}")
        return
    for path in entries:
        if path.suffix != ".md":
            continue
        name = path.stem
        if not NAME.fullmatch(name):
            warnings.append(f"{path}: not a valid command name")
        elif name in builtins:
            warnings.append(f"{path}: /{name} is a built-in command; skipped")
        elif source == "project" and path.is_symlink():
            warnings.append(f"{path}: a symlink is not followed in the project folder")
        else:
            try:
                found[name] = _read(path, name, source)
            except (OSError, ValueError) as exc:  # UnicodeDecodeError is a ValueError
                reason = exc.strerror or str(exc) if isinstance(exc, OSError) else str(exc)
                warnings.append(f"{path}: {reason}")


def load_commands(
    workspace: Path, builtins: Collection[str], env: Mapping[str, str] = os.environ, *, hold_global: bool = False
) -> Commands:
    """Read both folders once. A project command replaces a global one; problems become warnings. `hold_global`
    (the config watch's marker): the global ones are read only to name them in `held`, and not loaded."""
    found: dict[str, Command] = {}
    warnings: list[str] = []
    held: dict[str, Command] = {}
    _load_dir(global_commands_dir(env), "global", builtins, held if hold_global else found, [] if hold_global else warnings)
    project = project_commands_dir(workspace)
    if project.parent.is_symlink() or project.is_symlink():
        warnings.append(f"{project}: a symlink is not followed in the project folder")
    else:
        _load_dir(project, "project", builtins, found, warnings)
    shown = [escape(w) for w in warnings]
    if len(shown) > MAX_WARNINGS:
        shown = [*shown[:MAX_WARNINGS], f"and {len(shown) - MAX_WARNINGS} more warnings"]
    return Commands(found, tuple(shown), tuple(sorted(held)))
