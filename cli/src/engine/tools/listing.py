"""File listing that respects .gitignore through git, with a plain walk as the fallback."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def list_files(root: Path) -> list[Path]:
    """Sorted absolute paths of the files under `root` (blocking: call via to_thread)."""
    listed = _git_files(root)
    if listed is None:
        listed = _walk_files(root)
    return sorted(listed)


def _git_files(root: Path) -> list[Path] | None:
    try:
        inside = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return None
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except OSError:  # git is not installed or cannot run
        return None
    if result.returncode != 0:
        return None
    names = (n for n in result.stdout.decode("utf-8", errors="surrogateescape").split("\0") if n)
    found: list[Path] = []
    for path in (root / name for name in names):
        if path.is_file():
            found.append(path)
        elif path.is_dir():  # an untracked nested repository or a submodule is listed as `dir/`
            found.extend(_walk_files(path))
    return found


def _walk_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirs, files in os.walk(root):
        if ".git" in dirs:
            dirs.remove(".git")
        found.extend(Path(dirpath) / name for name in files)
    return found
