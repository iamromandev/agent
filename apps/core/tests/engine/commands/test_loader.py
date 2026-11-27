from __future__ import annotations

import os
from pathlib import Path

import pytest
from core.engine.commands.loader import (
    MAX_BYTES,
    Command,
    global_commands_dir,
    load_commands,
    parse,
)

BUILTINS = {"help", "new"}


def test_parse_reads_description_and_strips_one_newline() -> None:
    assert parse("---\ndescription:  Review it \nother: 1\n---\nBody $ARGUMENTS\n") == ("Review it", "Body $ARGUMENTS\n")


def test_parse_accepts_windows_line_endings() -> None:
    assert parse("---\r\ndescription: D\r\n---\r\nBody") == ("D", "Body")


def test_parse_last_description_wins_and_key_is_case_sensitive() -> None:
    assert parse("---\ndescription: a\nDescription: z\ndescription: b\n---\nx")[0] == "b"


def test_parse_accepts_a_bom_and_trailing_space_on_the_fences() -> None:
    assert parse("\ufeff---\ndescription: D\n---  \nBody") == ("D", "Body")
    assert parse("\ufeffPlain") == (None, "Plain")


def test_parse_unclosed_front_matter_raises() -> None:
    with pytest.raises(ValueError, match="front matter is not closed"):
        parse("---\ndescription: a\nBody")


def test_parse_arguments_in_front_matter_is_not_body() -> None:
    assert parse("---\ndescription: $ARGUMENTS\n---\nBody") == ("$ARGUMENTS", "Body")


def cmd(body: str) -> Command:
    return Command("x", body, None, "project")


@pytest.mark.parametrize(
    ("body", "rest", "expected"),
    [
        ("Focus on: $ARGUMENTS", " error handling ", "Focus on: error handling"),
        ("Do X", "more", "Do X\n\nmore"),
        ("Do X\n", "", "Do X"),
        ("$ARGUMENTS and $ARGUMENTS", "a", "a and a"),
        ("   \n", "", ""),
        ("   \n", "args", "args"),
        ("Do X", "l1\nl2", "Do X\n\nl1\nl2"),
    ],
)
def test_expand(body: str, rest: str, expected: str) -> None:
    assert cmd(body).expand(rest) == expected


def test_global_dir_ignores_relative_xdg() -> None:
    assert global_commands_dir({"XDG_CONFIG_HOME": "rel"}) == Path.home() / ".config" / "agent" / "commands"


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    glob = tmp_path / "cfg" / "agent" / "commands"
    proj = tmp_path / "ws" / ".agent" / "commands"
    glob.mkdir(parents=True)
    proj.mkdir(parents=True)
    return glob, proj, {"XDG_CONFIG_HOME": str(tmp_path / "cfg")}


def load(dirs: tuple[Path, Path, dict[str, str]], builtins=BUILTINS):
    return load_commands(dirs[1].parent.parent, builtins, dirs[2])


def test_project_replaces_global_and_sources(dirs) -> None:
    glob, proj, _ = dirs
    (glob / "review.md").write_text("g")
    (glob / "deploy.md").write_text("d")
    (proj / "review.md").write_text("p")
    (proj / "notes.txt").write_text("ignored")
    result = load(dirs)
    assert [(c.name, c.source, c.body) for c in result.sorted()] == [("deploy", "global", "d"), ("review", "project", "p")]
    assert result.warnings == ()
    assert result.get("nope") is None


def test_bad_name_and_builtin_warn(dirs) -> None:
    _, proj, _ = dirs
    (proj / "bad name.md").write_text("x")
    (proj / "help.md").write_text("x")
    result = load(dirs)
    assert result.by_name == {}
    assert len(result.warnings) == 2
    assert "not a valid command name" in result.warnings[0]
    assert "/help is a built-in command; skipped" in result.warnings[1]


def test_bad_files_each_warn_and_others_load(dirs) -> None:
    _, proj, _ = dirs
    (proj / "ok.md").write_text("ok")
    (proj / "big.md").write_bytes(b"x" * (MAX_BYTES + 1024))
    (proj / "utf.md").write_bytes(b"\xff\xfe\x00bad")
    os.mkfifo(proj / "fifo.md")
    (proj / "dir.md").mkdir()
    result = load(dirs)
    assert list(result.by_name) == ["ok"]
    assert len(result.warnings) == 4


def test_project_symlink_file_refused_global_followed(dirs, tmp_path: Path) -> None:
    glob, proj, _ = dirs
    target = tmp_path / "t.md"
    target.write_text("t")
    (proj / "p.md").symlink_to(target)
    (glob / "g.md").symlink_to(target)
    result = load(dirs)
    assert list(result.by_name) == ["g"]
    assert len(result.warnings) == 1 and "symlink" in result.warnings[0]


def test_project_symlinked_folder_refused(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.md").write_text("a")
    (ws / ".agent").mkdir(parents=True)
    (ws / ".agent" / "commands").symlink_to(real)
    result = load_commands(ws, BUILTINS, {"XDG_CONFIG_HOME": str(tmp_path / "none")})
    assert result.by_name == {} and len(result.warnings) == 1

    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    (ws2 / ".agent").symlink_to(tmp_path)
    result = load_commands(ws2, BUILTINS, {"XDG_CONFIG_HOME": str(tmp_path / "none")})
    assert result.by_name == {} and len(result.warnings) == 1


def test_warnings_are_capped(dirs) -> None:
    for i in range(25):
        (dirs[1] / f"bad {i}.md").write_text("x")
    result = load(dirs)
    assert len(result.warnings) == 21
    assert result.warnings[-1] == "and 5 more warnings"


def test_front_matter_returns_tools_and_model_keys_and_parse_ignores_them() -> None:
    from core.engine.commands.loader import front_matter

    text = "---\ndescription: d\ntools: read, grep\nmodel: m:1\n---\nbody"
    keys, body = front_matter(text)
    assert keys == {"description": "d", "tools": "read, grep", "model": "m:1"}
    assert body == "body"
    assert parse(text) == ("d", "body")
