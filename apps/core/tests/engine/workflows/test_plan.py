from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from core.engine.permissions.config import RuleFile
from core.engine.permissions.rules import parse_rule
from core.engine.permissions.setup import Rules
from core.engine.tools.base import Tool, ToolContext
from core.engine.workflows.model import Step, Workflow
from core.engine.workflows.plan import PlanError, plan_calls
from core.engine.workflows.template import Scope
from core.testing.workflow_fakes import AssembleVideo, GenerateImage, Speak, make_workspace, tools_by_name

FOLDER = "workflows/reel/r1"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return make_workspace(tmp_path)


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, bash_timeout=5.0)


def workflow(pic_path: str = "{folder}/s{index}.png", prompt: str = "{item.image_prompt}") -> Workflow:
    steps = (
        Step("story", "model", prompt="x", schema={"type": "object"}),
        Step("pic", "each", tool="generate_image", over="story.scenes", args={"prompt": prompt, "path": pic_path}),
        Step("voice", "each", tool="speak", over="story.scenes", args={"prompt": "{item.narration}", "path": "{folder}/v{index}.wav"}),
        Step(
            "video",
            "tool",
            tool="assemble_video",
            args={"path": "{folder}/reel.mp4", "scenes": {"$map": "story.scenes", "to": {"image": "{folder}/s{index}.png"}}},
        ),
    )
    return Workflow("reel", "project", Path("/w/reel.json"), "0" * 64, "d", "", "workflows/{name}/{run}", steps)


def scope(scenes: int = 5, prompt: str = "a cat") -> Scope:
    story = {"scenes": [{"image_prompt": f"{prompt} {i}", "narration": f"n{i}"} for i in range(scenes)]}
    return Scope({"folder": FOLDER, "run": "r1", "name": "reel", "input": "cats", "story": story})


def tools() -> dict[str, Tool]:
    return tools_by_name(GenerateImage(), Speak(), AssembleVideo())


def plan(workspace: Path, ctx: ToolContext, wf: Workflow | None = None, s: Scope | None = None, rules: Rules | None = None, **kw: Any):
    folder = workspace / FOLDER
    return plan_calls(wf or workflow(**kw), s or scope(), tools(), ctx, rules or Rules.empty(), folder)


def test_five_scenes_make_eleven_calls_in_order(workspace: Path, ctx: ToolContext) -> None:
    calls = plan(workspace, ctx)

    assert len(calls) == 11
    counts: dict[str, int] = {}
    for c in calls:
        counts[c.tool] = counts.get(c.tool, 0) + 1
    assert counts == {"generate_image": 5, "speak": 5, "assemble_video": 1}
    assert [c.key for c in calls][:2] == ["pic/1", "pic/2"]
    assert calls[5].key == "voice/1"
    assert calls[-1].key == "video"
    assert calls[-1].index is None and calls[0].index == 1 and calls[0].step == "pic"


def test_args_are_rendered_and_map_builds_the_scene_list(workspace: Path, ctx: ToolContext) -> None:
    calls = plan(workspace, ctx)

    assert calls[0].args == {"prompt": "a cat 0", "path": f"{FOLDER}/s1.png"}
    assert calls[-1].args["scenes"] == [{"image": f"{FOLDER}/s{i}.png"} for i in range(1, 6)]


def test_digest_is_the_sha256_of_the_canonical_args(workspace: Path, ctx: ToolContext) -> None:
    call = plan(workspace, ctx)[0]
    canonical = json.dumps(call.args, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    assert call.digest == hashlib.sha256(canonical.encode()).hexdigest()


def test_a_prompt_with_dotdot_only_changes_the_prompt_text(workspace: Path, ctx: ToolContext) -> None:
    calls = plan(workspace, ctx, s=scope(2, prompt="../../x"))

    assert calls[0].args["prompt"] == "../../x 0"
    assert calls[0].args["path"] == f"{FOLDER}/s1.png"


def test_a_path_built_from_model_text_is_refused(workspace: Path, ctx: ToolContext) -> None:
    with pytest.raises(PlanError, match=r"pic/1: path leaves the run folder"):
        plan(workspace, ctx, wf=workflow(pic_path="{item.image_prompt}.png"), s=scope(2, prompt="../../x"))


def test_a_path_outside_the_run_folder_is_refused(workspace: Path, ctx: ToolContext) -> None:
    with pytest.raises(PlanError, match=r"pic/1: path leaves the run folder"):
        plan(workspace, ctx, wf=workflow(pic_path="elsewhere/s{index}.png"))


def test_the_run_folder_itself_is_not_a_valid_path(workspace: Path, ctx: ToolContext) -> None:
    with pytest.raises(PlanError, match="path leaves the run folder"):
        plan(workspace, ctx, wf=workflow(pic_path="{folder}"))


def test_a_symlink_inside_the_run_folder_that_points_out_is_refused(workspace: Path, ctx: ToolContext, tmp_path: Path) -> None:
    run = workspace / FOLDER
    run.mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    (run / "link").symlink_to(tmp_path / "outside", target_is_directory=True)

    with pytest.raises(PlanError, match="path leaves the run folder"):
        plan(workspace, ctx, wf=workflow(pic_path="{folder}/link/s{index}.png"))


def test_a_tool_that_is_not_available_is_refused(workspace: Path, ctx: ToolContext) -> None:
    folder = workspace / FOLDER
    with pytest.raises(PlanError, match="tool speak is not available"):
        plan_calls(workflow(), scope(), tools_by_name(GenerateImage(), AssembleVideo()), ctx, Rules.empty(), folder)


def test_bad_arguments_are_refused_with_the_call_key(workspace: Path, ctx: ToolContext) -> None:
    wf = workflow(prompt="{item}")  # an object where the prompt must be a string

    with pytest.raises(PlanError, match=r"pic/1: generate_image: \"prompt\" must be a string"):
        plan(workspace, ctx, wf=wf)


def test_a_missing_value_is_a_plan_error_naming_the_step(workspace: Path, ctx: ToolContext) -> None:
    with pytest.raises(PlanError, match=r"step pic: .*\{item\.image_prompt\}"):
        plan(workspace, ctx, s=Scope({**scope().values, "story": {"scenes": [{"narration": "x"}]}}))


def test_each_over_a_non_list_is_a_plan_error(workspace: Path, ctx: ToolContext) -> None:
    with pytest.raises(PlanError, match=r"step pic: story\.scenes is not a list"):
        plan(workspace, ctx, s=Scope({**scope().values, "story": {"scenes": "nope"}}))


def test_a_deny_rule_refuses_the_plan(workspace: Path, ctx: ToolContext) -> None:
    rules = Rules(RuleFile((), (parse_rule("speak"),), (), None, None), RuleFile((), (), (), None, None), True)

    with pytest.raises(PlanError, match=r"voice/1: denied by rule speak \(global\)"):
        plan(workspace, ctx, rules=rules)


def test_an_empty_each_list_plans_no_calls_for_it(workspace: Path, ctx: ToolContext) -> None:
    calls = plan(workspace, ctx, s=scope(0))

    assert [c.key for c in calls] == ["video"]


# --- input confinement ------------------------------------------------------------------------


def video_wf(scene: dict[str, Any], model_text: str = "x") -> Workflow:
    steps = (
        Step("story", "model", prompt="x", schema={"type": "object"}),
        Step(
            "video",
            "tool",
            tool="assemble_video",
            args={"path": "{folder}/reel.mp4", "scenes": {"$map": "story.scenes", "to": scene}},
        ),
    )
    return Workflow("reel", "project", Path("/w/reel.json"), "0" * 64, "d", "", "workflows/{name}/{run}", steps)


def plan_video(workspace: Path, ctx: ToolContext, scene: dict[str, Any], **kw: Any):
    return plan(workspace, ctx, wf=video_wf(scene), s=scope(1, **kw))


def test_scene_inputs_inside_the_run_folder_are_fine(workspace: Path, ctx: ToolContext) -> None:
    calls = plan_video(workspace, ctx, {"image": "{folder}/s{index}.png", "audio": "{folder}/v{index}.wav"})

    assert [c.key for c in calls] == ["video"]


@pytest.mark.parametrize(
    "bad",
    ["../../../etc/passwd", "/etc/passwd", ".agent/settings.json", "src/.git/config", ".GIT/config", "~/x.png"],
)
def test_an_input_outside_the_workspace_or_in_agent_or_git_is_refused(workspace: Path, ctx: ToolContext, bad: str) -> None:
    for key in ("image", "audio", "music", "mask"):
        with pytest.raises(PlanError, match="input"):
            plan_video(workspace, ctx, {key: bad})


def test_an_input_in_a_list_is_checked(workspace: Path, ctx: ToolContext) -> None:
    with pytest.raises(PlanError, match="input"):
        plan_video(workspace, ctx, {"image": "{folder}/a.png", "references": ["{folder}/a.png", ".agent/x.png"]})


def test_an_input_that_is_not_a_string_is_refused(workspace: Path, ctx: ToolContext) -> None:
    with pytest.raises(PlanError, match="input"):
        plan_video(workspace, ctx, {"image": 3})


def test_an_input_in_the_run_folder_that_is_a_symlink_out_is_refused(workspace: Path, ctx: ToolContext, tmp_path: Path) -> None:
    run = workspace / FOLDER
    run.mkdir(parents=True)
    (workspace / "docs").mkdir()
    (run / "link").symlink_to(workspace / "docs", target_is_directory=True)

    with pytest.raises(PlanError, match="input"):
        plan_video(workspace, ctx, {"image": "{folder}/link/a.png"})


def test_an_input_that_resolves_into_agent_through_a_symlink_is_refused(workspace: Path, ctx: ToolContext) -> None:
    (workspace / ".agent").mkdir()
    (workspace / "innocent").symlink_to(workspace / ".agent", target_is_directory=True)

    with pytest.raises(PlanError, match="input"):
        plan_video(workspace, ctx, {"image": "innocent/x.png"})


def test_a_workspace_file_outside_the_run_folder_may_be_an_input(workspace: Path, ctx: ToolContext) -> None:
    calls = plan_video(workspace, ctx, {"image": "assets/logo.png"})

    assert calls[0].args["scenes"] == [{"image": "assets/logo.png"}]


def test_a_protected_input_is_refused(workspace: Path, ctx: ToolContext) -> None:
    with pytest.raises(PlanError, match="input"):
        plan_video(workspace, ctx, {"image": "secrets/key.pem"})
