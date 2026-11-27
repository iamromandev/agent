from __future__ import annotations

import json
from pathlib import Path

from core.engine.mcp.config import (
    ServerConfig,
    load_file,
)


def write(path: Path, content: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    return path


def test_a_server_with_command_args_and_env_is_loaded(tmp_path: Path) -> None:
    path = write(
        tmp_path / "mcp.json",
        {"servers": {"fs-notes": {"command": "uvx", "args": ["notes-mcp"], "env": {"NOTES_DIR": "/n"}}}},
    )
    loaded = load_file(path, "global")
    assert loaded.warnings == []
    assert loaded.servers == [ServerConfig("fs-notes", "uvx", ("notes-mcp",), {"NOTES_DIR": "/n"}, "global")]


def test_a_file_that_is_not_text_gives_a_warning(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_bytes(b"\xff\xfe\x00")
    loaded = load_file(path, "global")
    assert loaded.servers == []
    assert [w.startswith(f"{path}: ") for w in loaded.warnings] == [True]


def test_a_top_level_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path / "mcp.json", [1, 2])
    loaded = load_file(path, "global")
    assert loaded.servers == []
    assert loaded.warnings == [f"{path}: top level must be an object"]


def test_servers_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path / "mcp.json", {"servers": ["a"]})
    loaded = load_file(path, "global")
    assert loaded.servers == []
    assert loaded.warnings == [f'{path}: "servers" must be an object']


def test_an_entry_without_command_is_skipped_and_the_valid_one_loads(tmp_path: Path) -> None:
    path = write(tmp_path / "mcp.json", {"servers": {"good": {"command": "x"}, "bad": {"args": []}}})
    loaded = load_file(path, "global")
    assert [s.name for s in loaded.servers] == ["good"]
    assert loaded.warnings == [f'{path}: server "bad": missing "command"']


def test_each_kind_of_bad_entry_gets_its_own_warning(tmp_path: Path) -> None:
    path = write(
        tmp_path / "mcp.json",
        {
            "servers": {
                "notobject": "x",
                "cmd": {"command": 3},
                "emptycmd": {"command": ""},
                "args": {"command": "x", "args": "a b"},
                "arg_item": {"command": "x", "args": ["a", 1]},
                "env": {"command": "x", "env": ["A=1"]},
                "env_value": {"command": "x", "env": {"A": 1}},
                "ok": {"command": "x"},
            }
        },
    )
    loaded = load_file(path, "global")
    assert [s.name for s in loaded.servers] == ["ok"]
    assert loaded.warnings == [
        f'{path}: server "notobject": must be an object',
        f'{path}: server "cmd": "command" must be a string',
        f'{path}: server "emptycmd": "command" must not be empty',
        f'{path}: server "args": "args" must be a list of strings',
        f'{path}: server "arg_item": "args" must be a list of strings',
        f'{path}: server "env": "env" must be an object of strings',
        f'{path}: server "env_value": "env" must be an object of strings',
    ]


def test_a_server_name_in_a_warning_is_escaped(tmp_path: Path) -> None:
    path = write(tmp_path / "mcp.json", {"servers": {"a\x1b[2J\nb": {"args": []}}})

    (warning,) = load_file(path, "project").warnings

    assert warning.isprintable()
    assert 'server "a\\x1b[2J\\x0ab"' in warning
