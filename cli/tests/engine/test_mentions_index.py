"""`FileIndex`: what `@` completion offers."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest
from src.engine.guardrails.setup import Guardrails
from src.engine.mentions_index import FileIndex, Suggestion


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


def test_it_is_not_ready_before_build_and_suggests_nothing() -> None:
    index = FileIndex(Path("."), Guardrails.off(Path(".")))
    assert not index.ready
    assert index.suggest("") == []


def test_a_plain_folder_lists_top_level_entries_folders_first(tmp_path: Path) -> None:
    make(tmp_path, "src/a.py", "src/deep/b.py", "zed.txt", "alpha.txt")
    index = built(tmp_path)
    assert index.ready
    assert index.suggest("") == [
        Suggestion("src/", True),
        Suggestion("alpha.txt", False),
        Suggestion("zed.txt", False),
    ]


def test_it_lists_the_entries_under_a_folder(tmp_path: Path) -> None:
    make(tmp_path, "src/a.py", "src/deep/b.py", "other.py")
    assert names(built(tmp_path), "src/") == ["src/deep/", "src/a.py"]


def test_prefix_matches_come_before_contains_matches(tmp_path: Path) -> None:
    make(tmp_path, "main.py", "domain.py", "mail.py", "Makefile")
    assert names(built(tmp_path), "ma") == ["mail.py", "main.py", "Makefile", "domain.py"]


def test_a_leading_quote_is_ignored(tmp_path: Path) -> None:
    make(tmp_path, "my file.txt")
    assert names(built(tmp_path), '"my') == ["my file.txt"]


def test_the_limit_caps_suggestions(tmp_path: Path) -> None:
    make(tmp_path, *[f"f{i}.txt" for i in range(10)])
    assert len(names(built(tmp_path), "", limit=3)) == 3


def test_walk_skips_noise_folders_and_protected_files(tmp_path: Path) -> None:
    make(tmp_path, "a.py", ".env", "k.pem", "node_modules/x.js", ".venv/y.py", "__pycache__/z.pyc", ".git/HEAD")
    assert names(built(tmp_path), "") == ["a.py"]


def test_walk_does_not_follow_symlinked_folders(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    make(outside, "secret.txt")
    ws = tmp_path / "ws"
    make(ws, "a.py")
    (ws / "link").symlink_to(outside, target_is_directory=True)
    assert "link/secret.txt" not in names(built(ws), "link/")


def test_the_cap_limits_what_is_indexed(tmp_path: Path) -> None:
    make(tmp_path, *[f"f{i:03}.txt" for i in range(20)])
    assert len(names(built(tmp_path, limit=5), "")) == 5


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_a_git_repo_honours_gitignore_and_lists_untracked_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    make(tmp_path, "src/a.py", "build/x", "new.txt", ".env")
    (tmp_path / ".gitignore").write_text("build/\n")
    subprocess.run(["git", "add", "src/a.py"], cwd=tmp_path, check=True)
    result = names(built(tmp_path), "")
    assert "build/" not in result
    assert ".env" not in result
    assert {"src/", "new.txt", ".gitignore"} <= set(result)
    assert names(built(tmp_path), "build/") == []


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_a_repo_config_cannot_run_code_through_the_index(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    marker = tmp_path / "ran"
    hook = tmp_path / "hook.sh"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)
    subprocess.run(["git", "config", "core.fsmonitor", str(hook)], cwd=tmp_path, check=True)
    built(tmp_path)
    assert not marker.exists()


def test_without_git_on_the_path_it_walks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    make(tmp_path, "a.py")
    assert names(built(tmp_path), "") == ["a.py"]


def git_init(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)


@pytest.fixture
def no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_an_untracked_nested_repo_leaves_no_blank_or_folder_entries(tmp_path: Path) -> None:
    git_init(tmp_path)
    make(tmp_path, "a.py", "sub/inner.py")
    git_init(tmp_path / "sub")
    index = built(tmp_path)
    everything = names(index, "") + names(index, "sub/")
    assert everything == ["a.py"]


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


@pytest.mark.parametrize("use_git", [True, False])
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
    import src.engine.mentions_index as module

    monkeypatch.setattr(module, "WALK_FOLDERS", 3)
    make(tmp_path, *[f"d{i}/f.txt" for i in range(10)])
    index = built(tmp_path)
    assert index.ready
    assert len([n for n in names(index, "") if n.endswith("/")]) < 10


def test_a_walk_stops_at_its_time_bound(tmp_path: Path, no_git: None, monkeypatch: pytest.MonkeyPatch) -> None:

    import src.engine.mentions_index as module

    def slow_walk(top: object, *a: object, **k: object):
        for i in range(10_000):
            time.sleep(0.01)
            yield str(tmp_path), [], [f"f{i}.txt"]

    monkeypatch.setattr(module.os, "walk", slow_walk)
    monkeypatch.setattr(module, "WALK_SECONDS", 0.1)
    start = time.monotonic()
    index = FileIndex(tmp_path, Guardrails.off(tmp_path))
    index.build()
    assert time.monotonic() - start < 2
    assert index.ready


def test_cancel_stops_a_slow_walk_promptly(tmp_path: Path, no_git: None, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    import src.engine.mentions_index as module

    def slow_walk(top: object, *a: object, **k: object):
        for i in range(10_000):
            time.sleep(0.01)
            yield str(tmp_path), [], [f"f{i}.txt"]

    monkeypatch.setattr(module.os, "walk", slow_walk)
    index = FileIndex(tmp_path, Guardrails.off(tmp_path))
    threading.Timer(0.1, index.cancel).start()
    start = time.monotonic()
    index.build()
    assert time.monotonic() - start < 2
    assert not index.ready


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
