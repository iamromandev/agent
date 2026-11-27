"""`FileIndex`: what `@` completion offers."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest
from core.engine.guardrails.setup import Guardrails
from core.engine.mentions_index import FileIndex


def make(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")


def built(root: Path, limit: int = 50_000) -> FileIndex:
    index = FileIndex(root, Guardrails.off(root), limit)
    index.build()
    return index


def names(index: FileIndex, token: str, limit: int = 50) -> list[str]:
    return [s.path for s in index.suggest(token, limit)]


def test_the_cap_limits_what_is_indexed(tmp_path: Path) -> None:
    make(tmp_path, *[f"f{i:03}.txt" for i in range(20)])
    assert len(names(built(tmp_path, limit=5), "")) == 5


def git_init(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)


@pytest.fixture
def no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_a_tracked_file_missing_on_disk_is_not_listed(tmp_path: Path) -> None:
    git_init(tmp_path)
    make(tmp_path, "keep.py", "gone.py")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    (tmp_path / "gone.py").unlink()
    assert names(built(tmp_path), "") == ["keep.py"]


def test_a_path_with_a_newline_or_quote_is_not_offered(tmp_path: Path) -> None:
    make(tmp_path, "ok.py", 'q"uote.py', "new\nline.py")
    assert names(built(tmp_path), "") == ["ok.py"]


@pytest.mark.parametrize("use_git", [False])
def test_symlinks_are_listed_only_when_they_name_a_file_inside_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_git: bool
) -> None:
    if use_git and shutil.which("git") is None:
        pytest.skip("needs git")
    ws = tmp_path / "ws"
    make(ws, "real.py", "dir/x.py")
    make(tmp_path, "outside.txt")
    (ws / "inside.py").symlink_to("real.py")
    (ws / "out.txt").symlink_to(tmp_path / "outside.txt")
    (ws / "dirlink").symlink_to("dir", target_is_directory=True)
    (ws / "broken").symlink_to("nowhere")
    if use_git:
        git_init(ws)
    else:
        monkeypatch.setenv("PATH", "")
    assert names(built(ws), "") == ["dir/", "inside.py", "real.py"]


def test_walk_honours_root_and_nested_gitignore(tmp_path: Path, no_git: None) -> None:
    make(
        tmp_path,
        "keep.py", "skip.log", "out/a.txt", "top-only.tmp", "sub/top-only.tmp", "sub/x.py", "sub/y.dat",
        "sub/deep/skip.log", "docs/a.md", "docs/gen/b.md", "docs/gen/keep.md", "pkg/cache/c.bin", "pkg/src/d.py",
    )  # fmt: skip
    (tmp_path / ".gitignore").write_text(
        "# comment\n*.log\n!sub/deep/skip.log\nout/\n/top-only.tmp\ndocs/gen/*\n!docs/gen/keep.md\n**/cache/\n"
    )
    (tmp_path / "sub" / ".gitignore").write_text("*.dat\n")
    index = built(tmp_path)

    def listing(token: str) -> list[str]:
        return names(index, token)

    assert "skip.log" not in listing("") and "out/" not in listing("")
    assert "keep.py" in listing("")
    assert "top-only.tmp" not in listing("") and "sub/top-only.tmp" in listing("sub/")
    assert "sub/y.dat" not in listing("sub/") and "sub/x.py" in listing("sub/")
    assert "sub/deep/skip.log" in listing("sub/deep/")
    assert listing("docs/gen/") == ["docs/gen/keep.md"]
    assert listing("pkg/") == ["pkg/src/"]


def test_a_walk_stops_at_the_folder_bound(tmp_path: Path, no_git: None, monkeypatch: pytest.MonkeyPatch) -> None:
    import core.engine.mentions_index as module

    monkeypatch.setattr(module, "WALK_FOLDERS", 3)
    make(tmp_path, *[f"d{i}/f.txt" for i in range(10)])
    index = built(tmp_path)
    assert index.ready
    assert len([n for n in names(index, "") if n.endswith("/")]) < 10


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_cancel_kills_a_slow_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "git"
    fake.write_text("#!/bin/sh\nexec /bin/sleep 30\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    ws = tmp_path / "ws"
    make(ws, "a.py")
    index = FileIndex(ws, Guardrails.off(ws))
    threading.Timer(0.3, index.cancel).start()
    start = time.monotonic()
    index.build()
    assert time.monotonic() - start < 5
    assert not index.ready
