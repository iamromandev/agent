from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from src.agent.tools.listing import list_files


def _populate(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print()\n")
    (root / "README.md").write_text("hi\n")
    (root / "build").mkdir()
    (root / "build" / "out.py").write_text("x\n")
    (root / ".gitignore").write_text("build/\n")


def _git_init(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)


def test_git_tree_lists_untracked_files_and_respects_gitignore(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _populate(root)
    _git_init(root)

    files = list_files(root)

    assert files == sorted(
        [root / ".gitignore", root / "README.md", root / "src" / "main.py"],
    )
    assert all(".git" not in f.relative_to(root).parts for f in files)
    assert all(f.is_absolute() for f in files)


def test_git_listing_skips_tracked_files_deleted_from_disk(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    (root / "a.txt").write_text("a")
    (root / "b.txt").write_text("b")
    _git_init(root)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    (root / "a.txt").unlink()

    assert list_files(root) == [root / "b.txt"]


def test_subdirectory_of_a_git_tree_lists_only_its_files(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _populate(root)
    _git_init(root)

    assert list_files(root / "src") == [root / "src" / "main.py"]


def test_non_git_folder_is_walked_and_dot_git_is_skipped(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _populate(root)
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n")

    files = list_files(root)

    assert files == sorted(
        [
            root / ".gitignore",
            root / "README.md",
            root / "build" / "out.py",  # no git, so .gitignore is not applied
            root / "src" / "main.py",
        ],
    )


def test_walk_is_used_when_git_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path.resolve()
    _populate(root)
    _git_init(root)
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))

    files = list_files(root)

    assert root / "build" / "out.py" in files
    assert not any(".git" in f.relative_to(root).parts for f in files)


def test_untracked_nested_repo_files_are_listed_without_its_git_dir(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    (root / "top.py").write_text("x\n")
    _git_init(root)
    nested = root / "nested"
    nested.mkdir()
    _git_init(nested)
    (nested / "in.py").write_text("y\n")

    files = list_files(root)

    assert root / "nested" / "in.py" in files
    assert root / "top.py" in files
    assert all(".git" not in f.relative_to(root).parts for f in files)
