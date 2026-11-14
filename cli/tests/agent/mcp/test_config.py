from __future__ import annotations

import json
from pathlib import Path

from src.agent.mcp.config import ServerConfig, global_config_path, load_file


def write(path: Path, content: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    return path


def test_a_missing_file_means_no_servers_and_no_warning(tmp_path: Path) -> None:
    loaded = load_file(tmp_path / "mcp.json", "global")
    assert loaded.servers == []
    assert loaded.warnings == []


def test_a_server_with_command_args_and_env_is_loaded(tmp_path: Path) -> None:
    path = write(
        tmp_path / "mcp.json",
        {"servers": {"fs-notes": {"command": "uvx", "args": ["notes-mcp"], "env": {"NOTES_DIR": "/n"}}}},
    )
    loaded = load_file(path, "global")
    assert loaded.warnings == []
    assert loaded.servers == [ServerConfig("fs-notes", "uvx", ("notes-mcp",), {"NOTES_DIR": "/n"}, "global")]


def test_args_and_env_are_optional_and_unknown_keys_are_ignored(tmp_path: Path) -> None:
    path = write(tmp_path / "mcp.json", {"extra": 1, "servers": {"a": {"command": "x", "timeout": 3}}})
    loaded = load_file(path, "project")
    assert loaded.warnings == []
    assert loaded.servers == [ServerConfig("a", "x", (), {}, "project")]


def test_no_servers_and_an_empty_servers_object_are_fine(tmp_path: Path) -> None:
    loaded = load_file(write(tmp_path / "mcp.json", {"servers": {}}), "global")
    assert (loaded.servers, loaded.warnings) == ([], [])


def test_invalid_json_gives_no_servers_and_a_warning_naming_the_file(tmp_path: Path) -> None:
    path = write(tmp_path / "mcp.json", '{"servers": {')
    loaded = load_file(path, "global")
    assert loaded.servers == []
    assert len(loaded.warnings) == 1
    assert loaded.warnings[0].startswith(f"{path}: invalid JSON: ")


def test_a_file_that_is_not_text_gives_a_warning(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_bytes(b"\xff\xfe\x00")
    loaded = load_file(path, "global")
    assert loaded.servers == []
    assert [w.startswith(f"{path}: ") for w in loaded.warnings] == [True]


def test_a_folder_instead_of_a_file_gives_a_warning(tmp_path: Path) -> None:
    folder = tmp_path / "mcp.json"
    folder.mkdir()
    loaded = load_file(folder, "global")
    assert loaded.servers == []
    assert [w.startswith(f"{folder}: ") for w in loaded.warnings] == [True]


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


def test_a_missing_servers_key_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path / "mcp.json", {})
    assert load_file(path, "global").warnings == [f'{path}: "servers" must be an object']


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


def test_the_global_path_follows_xdg_config_home() -> None:
    assert global_config_path({"XDG_CONFIG_HOME": "/xdg"}) == Path("/xdg/agent/mcp.json")


def test_the_global_path_falls_back_to_the_home_config_folder() -> None:
    assert global_config_path({}) == Path.home() / ".config" / "agent" / "mcp.json"


def test_a_relative_xdg_config_home_is_ignored() -> None:
    assert global_config_path({"XDG_CONFIG_HOME": "rel"}) == Path.home() / ".config" / "agent" / "mcp.json"
