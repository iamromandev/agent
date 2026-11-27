from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from core.engine.tools.listing import list_files


def _populate(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print()\n")
    (root / "README.md").write_text("hi\n")
    (root / "build").mkdir()
    (root / "build" / "out.py").write_text("x\n")
    (root / ".gitignore").write_text("build/\n")


def _git_init(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)


def test_git_listing_skips_tracked_files_deleted_from_disk(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    (root / "a.txt").write_text("a")
    (root / "b.txt").write_text("b")
    _git_init(root)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    (root / "a.txt").unlink()

    assert list_files(root) == [root / "b.txt"]


def test_walk_is_used_when_git_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path.resolve()
    _populate(root)
    _git_init(root)
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))

    files = list_files(root)

    assert root / "build" / "out.py" in files
    assert not any(".git" in f.relative_to(root).parts for f in files)
