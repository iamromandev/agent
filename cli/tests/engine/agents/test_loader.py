"""Agent files: where they are found, what is refused, and how `tools:` and `model:` resolve."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from src.engine.agents.loader import MAX_BYTES, READ_ONLY_TOOLS, load_agents

GOOD = "---\ndescription: Reviews code\n---\nYou review.\n"


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    config = tmp_path / "config"
    glob = config / "agent" / "agents"
    glob.mkdir(parents=True)
    ws = tmp_path / "ws"
    proj = ws / ".agent" / "agents"
    proj.mkdir(parents=True)
    return glob, proj, {"XDG_CONFIG_HOME": str(config)}


def test_ac1_a_global_agent_is_loaded(dirs, tmp_path: Path) -> None:
    glob, _, env = dirs
    (glob / "reviewer.md").write_text(GOOD)
    agents = load_agents(tmp_path / "ws", env)
    agent = agents.get("reviewer")
    assert agent is not None
    assert (agent.source, agent.description, agent.prompt) == ("global", "Reviews code", "You review.\n")
    assert agent.tools is None and agent.model is None and agent.resolved_model is None
    assert agents.warnings == ()
    assert agents.names() == ["reviewer"]


def test_ac2_a_project_agent_replaces_a_global_one(dirs, tmp_path: Path) -> None:
    glob, proj, env = dirs
    (glob / "reviewer.md").write_text(GOOD)
    (proj / "reviewer.md").write_text(GOOD.replace("Reviews code", "Project one"))
    agent = load_agents(tmp_path / "ws", env).get("reviewer")
    assert agent is not None and (agent.source, agent.description) == ("project", "Project one")


@pytest.mark.parametrize(
    "text",
    [
        "no front matter\n",
        "---\ndescription: x\nbody never closed\n",
        "---\ndescription:\n---\nbody\n",
        "---\ntools: read\n---\nbody\n",
        "---\ndescription: x\n---\n  \n\n",
    ],
)
def test_ac3_invalid_files_give_one_warning_and_valid_ones_still_load(dirs, tmp_path: Path, text: str) -> None:
    glob, _, env = dirs
    (glob / "bad.md").write_text(text)
    (glob / "ok.md").write_text(GOOD)
    agents = load_agents(tmp_path / "ws", env)
    assert agents.names() == ["ok"]
    assert len(agents.warnings) == 1 and "bad.md" in agents.warnings[0]


def test_ac3_bytes_that_are_not_utf8_too_large_or_badly_named(dirs, tmp_path: Path) -> None:
    glob, _, env = dirs
    (glob / "latin.md").write_bytes(b"---\ndescription: \xff\n---\nbody")
    (glob / "big.md").write_text("---\ndescription: x\n---\n" + "a" * MAX_BYTES)
    (glob / "bad name.md").write_text(GOOD)
    (glob / "notes.txt").write_text("ignored")
    (glob / "ok.md").write_text(GOOD)
    agents = load_agents(tmp_path / "ws", env)
    assert agents.names() == ["ok"]
    assert len(agents.warnings) == 3


def test_warnings_are_escaped(dirs, tmp_path: Path) -> None:
    glob, _, env = dirs
    (glob / "bad‮name.md").write_text(GOOD)
    (warning,) = load_agents(tmp_path / "ws", env).warnings
    assert "‮" not in warning and "\\u202e" in warning


def test_ac4_symlinks_are_refused_in_the_project_and_followed_globally(dirs, tmp_path: Path) -> None:
    glob, proj, env = dirs
    target = tmp_path / "target.md"
    target.write_text(GOOD)
    (glob / "g.md").symlink_to(target)
    (proj / "p.md").symlink_to(target)
    agents = load_agents(tmp_path / "ws", env)
    assert agents.names() == ["g"]
    assert len(agents.warnings) == 1 and "p.md" in agents.warnings[0]


def test_ac4_a_symlinked_project_folder_is_refused(dirs, tmp_path: Path) -> None:
    _, proj, env = dirs
    real = tmp_path / "real"
    real.mkdir()
    (real / "p.md").write_text(GOOD)
    proj.rmdir()
    proj.symlink_to(real)
    agents = load_agents(tmp_path / "ws", env)
    assert agents.names() == []
    assert len(agents.warnings) == 1 and "symlink" in agents.warnings[0]


def test_ac7_default_is_reserved(dirs, tmp_path: Path) -> None:
    glob, _, env = dirs
    (glob / "default.md").write_text(GOOD)
    agents = load_agents(tmp_path / "ws", env)
    assert agents.names() == []
    assert len(agents.warnings) == 1 and "default" in agents.warnings[0]


def test_ac5_tools_are_parsed_deduplicated_and_unknown_ones_warn(dirs, tmp_path: Path) -> None:
    glob, _, env = dirs
    (glob / "a.md").write_text("---\ndescription: d\ntools: read, term_start, write, web_fetch, read,\n---\nb")
    agents = load_agents(tmp_path / "ws", env)
    agent = agents.get("a")
    assert agent is not None and agent.tools == ("read", "write", "web_fetch")
    assert len(agents.warnings) == 1 and "term_start" in agents.warnings[0]


def test_an_empty_or_missing_tools_key_means_the_default_set(dirs, tmp_path: Path) -> None:
    glob, _, env = dirs
    (glob / "blank.md").write_text("---\ndescription: d\ntools:  \n---\nb")
    (glob / "commas.md").write_text("---\ndescription: d\ntools: , ,\n---\nb")
    (glob / "dflt.md").write_text(GOOD)
    agents = load_agents(tmp_path / "ws", env)
    for name in ("blank", "commas", "dflt"):
        assert agents.get(name).tools is None  # ty: ignore[unresolved-attribute]
    assert agents.warnings == ()
    assert "web_search" in READ_ONLY_TOOLS


def test_a_tools_list_with_no_valid_name_means_no_tools_and_warns(dirs, tmp_path: Path) -> None:
    glob, _, env = dirs
    (glob / "a.md").write_text("---\ndescription: d\ntools: term_start, task\n---\nb")
    agents = load_agents(tmp_path / "ws", env)
    assert agents.get("a").tools == ()  # ty: ignore[unresolved-attribute]
    assert len(agents.warnings) == 3
    assert "no valid tools; the agent has none" in agents.warnings[-1]


def test_a_blank_model_warns_and_is_dropped(dirs, tmp_path: Path) -> None:
    glob, _, env = dirs
    (glob / "a.md").write_text("---\ndescription: d\nmodel:  \n---\nb")
    agents = load_agents(tmp_path / "ws", env)
    assert agents.get("a").model is None  # ty: ignore[unresolved-attribute]
    assert len(agents.warnings) == 1 and "model is empty" in agents.warnings[0]


def test_warnings_are_capped(dirs, tmp_path: Path) -> None:
    glob, _, env = dirs
    for i in range(25):
        (glob / f"b{i}.md").write_text("nope")
    warnings = load_agents(tmp_path / "ws", env).warnings
    assert len(warnings) == 21 and warnings[-1] == "and 5 more warnings"


def test_ac15_with_models_checks_the_installed_list(dirs, tmp_path: Path) -> None:
    glob, _, env = dirs
    (glob / "ok.md").write_text("---\ndescription: d\nmodel: other\n---\nb")
    (glob / "tag.md").write_text("---\ndescription: d\nmodel: tagged\n---\nb")
    (glob / "gone.md").write_text("---\ndescription: d\nmodel: missing\n---\nb")
    (glob / "plain.md").write_text(GOOD)
    agents = load_agents(tmp_path / "ws", env)
    assert agents.needs_models()
    resolved = agents.with_models(["other", "tagged:latest"])
    assert resolved.get("ok").resolved_model == "other"  # ty: ignore[unresolved-attribute]
    assert resolved.get("tag").resolved_model == "tagged:latest"  # ty: ignore[unresolved-attribute]
    assert resolved.get("gone").resolved_model is None  # ty: ignore[unresolved-attribute]
    (warning,) = resolved.warnings
    assert "gone.md" in warning and "'missing' is not installed; using the session's model" in warning
    assert agents.get("ok").resolved_model is None  # ty: ignore[unresolved-attribute]  # the original is unchanged


def test_with_models_none_takes_names_as_given(dirs, tmp_path: Path) -> None:
    glob, _, env = dirs
    (glob / "a.md").write_text("---\ndescription: d\nmodel: whatever\n---\nb")
    resolved = load_agents(tmp_path / "ws", env).with_models(None)
    assert resolved.get("a").resolved_model == "whatever"  # ty: ignore[unresolved-attribute]
    assert resolved.warnings == ()


def test_no_models_needed_without_model_keys(dirs, tmp_path: Path) -> None:
    glob, _, env = dirs
    (glob / "a.md").write_text(GOOD)
    assert not load_agents(tmp_path / "ws", env).needs_models()


def test_nfr3_fifty_files_load_fast(dirs, tmp_path: Path) -> None:
    glob, _, env = dirs
    for i in range(50):
        (glob / f"a{i}.md").write_text(f"---\ndescription: d{i}\ntools: read, grep\nmodel: m{i}\n---\nbody {i}\n")
    start = time.perf_counter()
    agents = load_agents(tmp_path / "ws", env)
    assert time.perf_counter() - start < 0.2  # generous: the target is 20 ms on a quiet machine
    assert len(agents.by_name) == 50


def test_with_models_caps_loader_and_model_warnings_once(dirs, tmp_path: Path) -> None:
    glob, _, env = dirs
    for i in range(25):
        (glob / f"b{i}.md").write_text("nope")
    (glob / "gone.md").write_text("---\ndescription: d\nmodel: missing\n---\nb")
    resolved = load_agents(tmp_path / "ws", env).with_models(["other"])
    assert len(resolved.warnings) == 21
    assert resolved.warnings[-1] == "and 6 more warnings"
    assert sum(w.startswith("and ") for w in resolved.warnings) == 1
    assert resolved.with_models(["other"]).warnings == resolved.warnings


def test_an_agents_built_directly_keeps_its_warnings_through_with_models() -> None:
    from src.engine.agents.loader import Agents

    agents = Agents({}, ("a.md: odd",))
    assert agents.with_models(["x"]).warnings == ("a.md: odd",)
    assert agents.with_models(None).warnings == ("a.md: odd",)


def test_generate_image_is_not_allowed_in_an_agent_file(dirs, tmp_path: Path) -> None:
    glob, _, env = dirs
    (glob / "a.md").write_text("---\ndescription: d\ntools: read, generate_image\n---\nb")
    agents = load_agents(tmp_path / "ws", env)
    agent = agents.get("a")
    assert agent is not None and agent.tools == ("read",)
    assert len(agents.warnings) == 1 and "tool 'generate_image' is not allowed in an agent; ignored" in agents.warnings[0]
