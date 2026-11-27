from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from core.engine.workflows.loader import FORBIDDEN_TOOLS, builtin_dir, global_dir, load_workflows


def reel(**over: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "description": "make a reel",
        "input": {"description": "the topic"},
        "steps": [
            {
                "id": "story",
                "kind": "model",
                "prompt": "Write scenes about {input}",
                "schema": {
                    "type": "object",
                    "properties": {"scenes": {"type": "array", "items": {"type": "object"}}},
                    "required": ["scenes"],
                },
            },
            {
                "id": "pic",
                "kind": "each",
                "over": "story.scenes",
                "tool": "generate_image",
                "args": {"prompt": "{item.image_prompt}", "path": "{folder}/s{index}.png"},
            },
            {
                "id": "video",
                "kind": "tool",
                "tool": "assemble_video",
                "args": {
                    "path": "{folder}/reel.mp4",
                    "scenes": {"$map": "story.scenes", "to": {"image": "{folder}/s{index}.png"}},
                },
            },
        ],
    }
    data.update(over)
    return data


@pytest.fixture
def dirs(tmp_path: Path) -> dict[str, Path]:
    made = {
        "ws": tmp_path / "ws",
        "builtin": tmp_path / "builtin",
        "config": tmp_path / "config",
    }
    made["ws"].mkdir()
    made["builtin"].mkdir()
    (made["config"] / "agent" / "workflows").mkdir(parents=True)
    (made["ws"] / ".agent" / "workflows").mkdir(parents=True)
    return made


def put(folder: Path, name: str, data: Any) -> Path:
    path = folder / f"{name}.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


def load(dirs: dict[str, Path]):
    return load_workflows(dirs["ws"], {"XDG_CONFIG_HOME": str(dirs["config"])}, builtin=dirs["builtin"])


def project(dirs: dict[str, Path]) -> Path:
    return dirs["ws"] / ".agent" / "workflows"


def glob_dir(dirs: dict[str, Path]) -> Path:
    return dirs["config"] / "agent" / "workflows"


def test_valid_workflow_loads(dirs: dict[str, Path]) -> None:
    put(project(dirs), "reel", reel())

    found = load(dirs)

    wf = found.items["reel"]
    assert found.warnings == []
    assert (wf.source, wf.description, wf.input_help) == ("project", "make a reel", "the topic")
    assert [s.id for s in wf.steps] == ["story", "pic", "video"]
    assert wf.folder == "workflows/{name}/{run}"
    assert len(wf.sha256) == 64


def test_builtin_global_project_replace_in_that_order(dirs: dict[str, Path]) -> None:
    put(dirs["builtin"], "a", reel(description="builtin"))
    put(dirs["builtin"], "b", reel(description="builtin b"))
    put(glob_dir(dirs), "a", reel(description="global"))
    put(glob_dir(dirs), "b", reel(description="global b"))
    put(project(dirs), "a", reel(description="project"))

    found = load(dirs)

    assert found.items["a"].description == "project"
    assert found.items["a"].source == "project"
    assert found.items["b"].description == "global b"
    assert found.items["b"].source == "global"


def test_builtin_only(dirs: dict[str, Path]) -> None:
    put(dirs["builtin"], "a", reel())
    assert load(dirs).items["a"].source == "builtin"


def test_default_builtin_folder_ships_the_reel() -> None:
    assert builtin_dir().is_dir()
    assert [p.name for p in builtin_dir().glob("*.json")] == ["reel.json"]


def test_global_dir_uses_xdg_config_home(tmp_path: Path) -> None:
    assert global_dir({"XDG_CONFIG_HOME": str(tmp_path)}) == tmp_path / "agent" / "workflows"


def test_oversized_file_is_a_warning_and_others_load(dirs: dict[str, Path]) -> None:
    put(project(dirs), "big", reel(description="x" * 70_000))
    put(project(dirs), "ok", reel())

    found = load(dirs)

    assert "ok" in found.items
    assert "big" not in found.items
    assert any("big.json" in w and "larger than" in w for w in found.warnings)


def test_unknown_top_level_key_is_a_warning(dirs: dict[str, Path]) -> None:
    put(project(dirs), "bad", reel(extra=1))
    put(project(dirs), "ok", reel())

    found = load(dirs)

    assert set(found.items) == {"ok"}
    assert any("bad.json" in w and "unknown key 'extra'" in w for w in found.warnings)


def test_unknown_step_key_is_a_warning(dirs: dict[str, Path]) -> None:
    data = reel()
    data["steps"][2]["colour"] = "red"
    put(project(dirs), "bad", data)

    found = load(dirs)

    assert "bad" not in found.items
    assert any("unknown key 'colour'" in w for w in found.warnings)


@pytest.mark.parametrize("bad_id", ["Story", "a-b", "", "x y"])
def test_bad_step_id_is_a_warning(dirs: dict[str, Path], bad_id: str) -> None:
    data = reel()
    data["steps"][0]["id"] = bad_id
    put(project(dirs), "bad", data)

    found = load(dirs)

    assert "bad" not in found.items
    assert any("bad.json" in w and "step id" in w for w in found.warnings)


def test_duplicate_step_id_is_a_warning(dirs: dict[str, Path]) -> None:
    data = reel()
    data["steps"][2]["id"] = "pic"
    put(project(dirs), "bad", data)

    assert any("duplicate step id 'pic'" in w for w in load(dirs).warnings)


def test_step_count_is_between_1_and_20(dirs: dict[str, Path]) -> None:
    put(project(dirs), "none", reel(steps=[]))
    many = [{"id": f"t{i}", "kind": "tool", "tool": "speak", "args": {}} for i in range(21)]
    put(project(dirs), "many", reel(steps=many))

    found = load(dirs)

    assert found.items == {}
    assert len(found.warnings) == 2


def test_project_symlinked_file_is_refused(dirs: dict[str, Path], tmp_path: Path) -> None:
    real = put(tmp_path, "real", reel())
    (project(dirs) / "link.json").symlink_to(real)
    put(project(dirs), "ok", reel())

    found = load(dirs)

    assert set(found.items) == {"ok"}
    assert any("link.json" in w and "symlink" in w for w in found.warnings)


def test_project_symlinked_folder_is_refused(dirs: dict[str, Path], tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    put(other, "x", reel())
    (project(dirs)).rmdir()
    project(dirs).symlink_to(other, target_is_directory=True)
    put(glob_dir(dirs), "g", reel())

    found = load(dirs)

    assert set(found.items) == {"g"}
    assert any("symlink" in w for w in found.warnings)


def test_project_symlinked_agent_folder_is_refused(dirs: dict[str, Path], tmp_path: Path) -> None:
    other = tmp_path / "other" / "workflows"
    other.mkdir(parents=True)
    put(other, "x", reel())
    (dirs["ws"] / ".agent" / "workflows").rmdir()
    (dirs["ws"] / ".agent").rmdir()
    (dirs["ws"] / ".agent").symlink_to(other.parent, target_is_directory=True)

    found = load(dirs)

    assert found.items == {}
    assert any("symlink" in w for w in found.warnings)


def test_global_symlink_is_followed(dirs: dict[str, Path], tmp_path: Path) -> None:
    real = put(tmp_path, "real", reel())
    (glob_dir(dirs) / "g.json").symlink_to(real)

    assert "g" in load(dirs).items


def test_invalid_json_and_bad_utf8_are_warnings(dirs: dict[str, Path]) -> None:
    put(project(dirs), "j", "{not json")
    (project(dirs) / "u.json").write_bytes(b"\xff\xfe")

    found = load(dirs)

    assert found.items == {}
    assert len(found.warnings) == 2


def test_bad_file_name_is_a_warning(dirs: dict[str, Path]) -> None:
    put(project(dirs), "bad name", reel())
    assert any("not a valid workflow name" in w for w in load(dirs).warnings)


def test_other_files_are_ignored(dirs: dict[str, Path]) -> None:
    (project(dirs) / "notes.md").write_text("hi")
    found = load(dirs)
    assert found.items == {} and found.warnings == []


def test_warnings_are_escaped_and_capped(dirs: dict[str, Path]) -> None:
    for i in range(25):
        put(project(dirs), f"w{i:02}", "[1]")
    (project(dirs) / "evil\x1b[31m.json").write_text("[1]")

    found = load(dirs)

    assert len(found.warnings) == 21
    assert found.warnings[-1] == "and 6 more warnings"
    assert all("\x1b" not in w for w in found.warnings)


# --- AC2: structure ---------------------------------------------------------------------------


def only_warning(dirs: dict[str, Path], data: dict[str, Any]) -> str:
    put(project(dirs), "bad", data)
    found = load(dirs)
    assert "bad" not in found.items
    assert len(found.warnings) == 1
    return found.warnings[0]


def test_model_step_after_a_tool_step_is_an_error(dirs: dict[str, Path]) -> None:
    data = reel()
    data["steps"].append({"id": "late", "kind": "model", "prompt": "x", "schema": {"type": "object"}})

    assert "model step 'late' comes after a tool step" in only_warning(dirs, data)


def test_item_outside_each_is_an_error(dirs: dict[str, Path]) -> None:
    data = reel()
    data["steps"][2]["args"]["x"] = "{item.x}"

    assert "{item}" in only_warning(dirs, data) or "item" in only_warning(dirs, data)


def test_index_outside_each_is_an_error(dirs: dict[str, Path]) -> None:
    data = reel()
    data["steps"][2]["args"]["x"] = "{index}"

    assert "index" in only_warning(dirs, data)


def test_item_in_a_model_prompt_is_an_error(dirs: dict[str, Path]) -> None:
    data = reel()
    data["steps"][0]["prompt"] = "{item.x}"

    assert "item" in only_warning(dirs, data)


def test_item_inside_map_is_fine_and_inside_each_too(dirs: dict[str, Path]) -> None:
    put(project(dirs), "ok", reel())
    assert "ok" in load(dirs).items


def test_unknown_placeholder_root_is_an_error(dirs: dict[str, Path]) -> None:
    data = reel()
    data["steps"][0]["prompt"] = "{ghost}"

    assert "unknown name 'ghost'" in only_warning(dirs, data)


def test_placeholder_of_a_later_model_step_is_an_error(dirs: dict[str, Path]) -> None:
    data = reel()
    data["steps"].insert(
        1, {"id": "second", "kind": "model", "prompt": "{second}", "schema": {"type": "object"}}
    )

    assert "unknown name 'second'" in only_warning(dirs, data)


def test_placeholder_of_an_earlier_model_step_is_fine(dirs: dict[str, Path]) -> None:
    data = reel()
    data["steps"].insert(
        1, {"id": "second", "kind": "model", "prompt": "{story.scenes}", "schema": {"type": "object"}}
    )
    put(project(dirs), "ok", data)

    assert "ok" in load(dirs).items


def test_each_over_must_name_a_model_step(dirs: dict[str, Path]) -> None:
    data = reel()
    data["steps"][1]["over"] = "video.scenes"

    assert "over" in only_warning(dirs, data)


def test_each_over_needs_step_and_path(dirs: dict[str, Path]) -> None:
    data = reel()
    data["steps"][1]["over"] = "story"

    assert "over" in only_warning(dirs, data)


def test_bad_schema_is_an_error(dirs: dict[str, Path]) -> None:
    data = reel()
    data["steps"][0]["schema"] = {"type": "object", "pattern": "x"}

    assert "unsupported keyword 'pattern'" in only_warning(dirs, data)


def test_broken_template_is_an_error(dirs: dict[str, Path]) -> None:
    data = reel()
    data["steps"][2]["args"]["path"] = "{folder"

    assert "args" in only_warning(dirs, data)


@pytest.mark.parametrize("folder", ["/abs", "a/../b", "../x", ".agent/x", "a/.git/b", "", 3, ".AGENT/x"])
def test_bad_folder_is_an_error(dirs: dict[str, Path], folder: Any) -> None:
    assert "folder" in only_warning(dirs, reel(folder=folder))


def test_custom_folder_template_loads(dirs: dict[str, Path]) -> None:
    put(project(dirs), "ok", reel(folder="out/{name}-{run}"))
    assert load(dirs).items["ok"].folder == "out/{name}-{run}"


def test_folder_may_not_use_step_values(dirs: dict[str, Path]) -> None:
    assert "folder" in only_warning(dirs, reel(folder="out/{story.title}"))


def test_missing_description_is_an_error(dirs: dict[str, Path]) -> None:
    data = reel()
    del data["description"]

    assert "description" in only_warning(dirs, data)


def test_bad_input_block_is_an_error(dirs: dict[str, Path]) -> None:
    assert "input" in only_warning(dirs, reel(input={"description": "x", "more": 1}))
    assert "input" in only_warning(dirs, reel(input="x"))


# --- AC3: forbidden tools --------------------------------------------------------------------


@pytest.mark.parametrize("tool", [*sorted(FORBIDDEN_TOOLS), "mcp__srv__do"])
def test_forbidden_tool_does_not_load(dirs: dict[str, Path], tool: str) -> None:
    data = reel()
    data["steps"][2]["tool"] = tool

    assert f"tool '{tool}' is not allowed in a workflow" in only_warning(dirs, data)


def test_forbidden_tool_in_each_does_not_load(dirs: dict[str, Path]) -> None:
    data = reel()
    data["steps"][1]["tool"] = "task"

    assert "not allowed in a workflow" in only_warning(dirs, data)


@pytest.mark.parametrize("bad_id", ["1story", "9", "1a"])
def test_a_step_id_may_not_start_with_a_digit(dirs: dict[str, Path], bad_id: str) -> None:
    data = reel()
    data["steps"][0]["id"] = bad_id
    data["steps"][1]["over"] = f"{bad_id}.scenes"

    assert "step id" in only_warning(dirs, data)


def test_an_underscore_step_id_works_in_a_placeholder(dirs: dict[str, Path]) -> None:
    data = reel()
    data["steps"][0]["id"] = "_story2"
    data["steps"][1]["over"] = "_story2.scenes"
    data["steps"][1]["args"]["prompt"] = "{_story2.title}"
    data["steps"][2]["args"]["scenes"]["$map"] = "_story2.scenes"
    put(project(dirs), "ok", data)

    assert "ok" in load(dirs).items


def test_bash_is_not_allowed_in_a_workflow(dirs: dict[str, Path]) -> None:
    assert "bash" in FORBIDDEN_TOOLS
    data = reel()
    data["steps"][2]["tool"] = "bash"

    assert "not allowed in a workflow" in only_warning(dirs, data)


def test_the_folder_must_use_run(dirs: dict[str, Path]) -> None:
    assert "{run}" in only_warning(dirs, reel(folder="out/{name}"))
    assert "{run}" in only_warning(dirs, reel(folder="out/fixed"))
