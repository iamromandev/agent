from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from src.engine.events import AgentEvent, WorkflowLine, WorkflowProgress
from src.engine.loop import Agent, AgentBusy
from src.engine.permissions.config import RuleFile
from src.engine.permissions.rules import parse_rule
from src.engine.permissions.setup import Rules
from src.engine.provider.base import Done, ProviderError, TextDelta
from src.engine.tools.base import Tool, ToolContext
from src.engine.workflows.loader import parse_workflow
from src.engine.workflows.manifest import file_sha256, load, save
from src.engine.workflows.model import PlanSummary, RunResult, Workflow
from src.engine.workflows.runner import WorkflowRunner
from src.engine.workflows.runs import RunIndex
from src.engine.workflows.timings import TimingStore

from tests.engine.workflows.fakes import AssembleVideo, GenerateImage, SlowTool, Speak
from tests.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider

NOW = datetime(2026, 11, 20, 10, 30, tzinfo=UTC)
RUN = "20261120-1030-cats"
FOLDER = f"workflows/reel/{RUN}"

SCHEMA = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {"image_prompt": {"type": "string"}, "narration": {"type": "string"}},
                "required": ["image_prompt", "narration"],
            },
        }
    },
    "required": ["scenes"],
}


def reel_data(pic_path: str = "{folder}/s{index}.png", folder: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "description": "make a reel",
        "steps": [
            {"id": "story", "kind": "model", "prompt": "Write scenes about {input}", "schema": SCHEMA},
            {
                "id": "pic",
                "kind": "each",
                "over": "story.scenes",
                "tool": "generate_image",
                "args": {"prompt": "{item.image_prompt}", "path": pic_path},
            },
            {
                "id": "voice",
                "kind": "each",
                "over": "story.scenes",
                "tool": "speak",
                "args": {"prompt": "{item.narration}", "path": "{folder}/v{index}.wav"},
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
    if folder is not None:
        data["folder"] = folder
    return data


def workflow(data: dict[str, Any] | None = None, tmp: Path = Path("/w/reel.json")) -> Workflow:
    return parse_workflow("reel", "project", tmp, json.dumps(data or reel_data()).encode())


def scenes(n: int) -> str:
    return json.dumps({"scenes": [{"image_prompt": f"p{i}", "narration": f"n{i}"} for i in range(1, n + 1)]})


def answer(text: str) -> list[Any]:
    return [TextDelta(text), Done(20, 10)]


class Sink:
    def __init__(self) -> None:
        self.items: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.items.append(event)

    @property
    def lines(self) -> list[tuple[str, str]]:
        return [(e.level, e.text) for e in self.items if isinstance(e, WorkflowLine)]

    @property
    def progress(self) -> list[str]:
        return [e.text for e in self.items if isinstance(e, WorkflowProgress)]


@dataclass
class Env:
    workspace: Path
    agent: Agent
    provider: ScriptedProvider
    approver: RecordingApprover
    recorder: MemoryRecorder
    runner: WorkflowRunner
    index: RunIndex
    img: GenerateImage
    voice: Speak
    video: AssembleVideo
    approved: list[PlanSummary] = field(default_factory=list)
    answer_yes: bool = True

    async def approve(self, plan: PlanSummary) -> bool:
        self.approved.append(plan)
        return self.answer_yes

    @property
    def run_folder(self) -> Path:
        return self.workspace / FOLDER


def build(
    tmp_path: Path,
    script: list[list[Any]],
    *,
    rules: Rules | None = None,
    extra: list[Tool] | None = None,
    skip: tuple[type[Tool], ...] = (),
    timings: TimingStore | None = None,
) -> Env:
    workspace = (tmp_path / "ws").resolve()
    workspace.mkdir(exist_ok=True)
    ctx = ToolContext(workspace=workspace, bash_timeout=5.0)
    provider = ScriptedProvider(script, context_window=10_000)
    approver = RecordingApprover()
    recorder = MemoryRecorder()
    the_rules = rules or Rules.empty()
    agent = Agent(provider, [], approver, ctx, 25, today=lambda: date(2026, 11, 20), recorder=recorder, rules=the_rules)
    img, voice, video = GenerateImage(), Speak(), AssembleVideo()
    tools: list[Tool] = [t for t in (img, voice, video) if not isinstance(t, skip)] + (extra or [])
    index = RunIndex(tmp_path / "data" / "runs.json")
    runner = WorkflowRunner(agent, lambda: tools, ctx, the_rules, index, now=lambda: NOW, timings=timings)
    return Env(workspace, agent, provider, approver, recorder, runner, index, img, voice, video)


async def run(env: Env, wf: Workflow | None = None, text: str = "cats", sink: Sink | None = None) -> RunResult:
    return await env.runner.run(wf or workflow(), text, sink or Sink(), env.approve)


def indexed_state(env: Env) -> str:
    entry = env.index.find(RUN)
    assert entry is not None
    return entry.state


def lookup(wf: Workflow):
    return lambda name: wf if name == wf.name else None


# --- AC5: the model step ----------------------------------------------------------------------


async def test_a_valid_answer_is_taken_and_the_plan_follows(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(2))])

    result = await run(env)

    assert result.ok, result.line
    assert len(env.approved) == 1 and len(env.approved[0].calls) == 5


@pytest.mark.parametrize("wrap", ["```json\n{}\n```", "Here you go:\n```\n{}\n```\nEnjoy", "Sure! {} Hope that helps."])
async def test_a_wrapped_answer_is_taken(tmp_path: Path, wrap: str) -> None:
    env = build(tmp_path, [answer(wrap.replace("{}", scenes(1)))])

    result = await run(env)

    assert result.ok, result.line
    assert len(env.approved[0].calls) == 3


async def test_the_first_json_value_is_used(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1) + "\n" + scenes(4))])

    await run(env)

    assert len(env.approved[0].calls) == 3


async def test_the_request_asks_for_json_matching_the_schema(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1))])

    await run(env)

    messages, tools = env.provider.requests[0]
    assert tools == []
    assert [m.role for m in messages] == ["system", "user"]
    assert "only one JSON value" in messages[0].content
    assert json.dumps(SCHEMA, sort_keys=True) in messages[0].content or '"minItems"' in messages[0].content
    assert messages[1].content == "Write scenes about cats"


async def test_an_invalid_answer_is_retried_once_with_the_error(tmp_path: Path) -> None:
    env = build(tmp_path, [answer('{"scenes": []}'), answer(scenes(1))])

    result = await run(env)

    assert result.ok, result.line
    retry = env.provider.requests[1][0]
    assert [m.role for m in retry] == ["system", "user", "assistant", "user"]
    assert retry[2].content == '{"scenes": []}'
    assert "$.scenes: needs at least 1 items" in retry[3].content


async def test_two_invalid_answers_stop_the_run_with_the_reply_escaped(tmp_path: Path) -> None:
    bad = "\x1b[31m" + "x" * 400
    env = build(tmp_path, [answer(bad), answer(bad)])

    result = await run(env)

    assert not result.ok
    assert result.line.startswith("workflow reel failed at story: the answer is not JSON; ")
    assert "\x1b" not in result.line
    assert "\\x1b[31m" + "x" * 194 in result.line  # the first 200 characters of the reply, escaped
    assert "x" * 195 not in result.line.replace("\\x1b[31m" + "x" * 194, "")
    assert env.approved == []
    assert len(env.provider.requests) == 2


async def test_a_provider_error_stops_the_run(tmp_path: Path) -> None:
    env = build(tmp_path, [[ProviderError("model is down")]])

    result = await run(env)

    assert not result.ok
    assert result.line == "workflow reel failed at story: model is down"


async def test_the_model_steps_leave_the_history_and_the_session_alone(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1))])

    await run(env)

    assert env.agent.history == []
    assert env.recorder.messages == []
    assert [r.workflow for r in env.recorder.usage_records] == [True]
    assert env.agent.usage.total.workflow_calls == 1


# --- AC3 / AC6: the plan ----------------------------------------------------------------------


async def test_a_tool_that_is_not_available_stops_before_the_approval(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1))], skip=(Speak,))

    result = await run(env)

    assert not result.ok
    assert result.line == "workflow reel failed at plan: tool speak is not available"
    assert env.approved == []


async def test_five_scenes_plan_eleven_calls_with_counts(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(5))])

    await run(env)

    plan = env.approved[0]
    assert (plan.workflow, plan.run, plan.folder) == ("reel", RUN, FOLDER)
    assert len(plan.calls) == 11
    assert plan.counts == {"generate_image": 5, "speak": 5, "assemble_video": 1}
    assert plan.estimate_s == round(5 * 1024 * 1024 * 9 * 4.24e-5 + 5 * 4.0 + 5 * 6.5)
    assert plan.estimate_source == "a rough guess"


async def test_a_path_built_from_model_text_is_refused_before_the_approval(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1))])
    wf = workflow(reel_data(pic_path="{item.image_prompt}.png"))

    result = await run(env, wf)

    assert not result.ok
    assert result.line == "workflow reel failed at plan: pic/1: path leaves the run folder"
    assert env.approved == []
    assert env.img.calls == []


async def test_a_deny_rule_refuses_the_plan_before_the_approval(tmp_path: Path) -> None:
    rules = Rules(RuleFile((), (parse_rule("speak"),), (), None, None), RuleFile((), (), (), None, None), True)
    env = build(tmp_path, [answer(scenes(1))], rules=rules)

    result = await run(env)

    assert not result.ok
    assert "voice/1: denied by rule speak (global)" in result.line
    assert env.approved == []


async def test_a_declined_plan_makes_no_calls_and_no_files(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(2))])
    env.answer_yes = False

    result = await run(env)

    assert not result.ok
    assert result.line == "workflow reel: the plan was not approved"
    assert env.img.calls == env.voice.calls == env.video.calls == []
    assert not env.run_folder.exists()
    assert env.index.find(RUN) is None


async def test_the_run_folder_must_stay_inside_the_workspace(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1))])
    outside = tmp_path / "ws" / "link"
    outside.symlink_to(tmp_path, target_is_directory=True)
    wf = workflow(reel_data(folder="link/{run}"))

    result = await run(env, wf)

    assert not result.ok
    assert "run folder" in result.line and "workspace" in result.line
    assert env.approved == []


async def test_the_run_folder_must_not_be_protected(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1))])
    wf = workflow(reel_data(folder="out/{run}.key"))  # `*.key` names are protected

    result = await run(env, wf)

    assert not result.ok
    assert "protected" in result.line


async def test_run_id_slug_and_folder_collision(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1)), answer(scenes(1))])
    (env.workspace / "workflows" / "reel" / "20261120-1030-hello-world-the-end").mkdir(parents=True)

    first = await run(env, text="Hello, World! The End")
    second = await run(env, text="")

    assert first.run == "20261120-1030-hello-world-the-end-2"
    assert second.run == "20261120-1030-run"


async def test_the_slug_is_at_most_24_characters(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1))])

    result = await run(env, text="a" * 60)

    assert result.run == "20261120-1030-" + "a" * 24


# --- AC7: running -----------------------------------------------------------------------------


async def test_calls_run_in_order_without_asking(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(2))])
    order: list[str] = []
    for tool in (env.img, env.voice, env.video):
        tool.log = order

    sink = Sink()
    result = await run(env, sink=sink)

    assert result.ok, result.line
    assert order == [
        "generate_image:s1.png",
        "generate_image:s2.png",
        "speak:v1.wav",
        "speak:v2.wav",
        "assemble_video:reel.mp4",
    ]
    assert env.approver.requests == []
    assert result.line == f"workflow reel finished: 5 calls, files in {FOLDER}"
    assert sink.lines[0] == ("info", f"✓ generate_image {FOLDER}/s1.png (1 file)")
    assert len(sink.lines) == 5
    manifest = load(env.run_folder)
    assert manifest is not None
    assert [c["state"] for c in manifest["calls"]] == ["done"] * 5
    assert indexed_state(env) == "done"


async def test_child_progress_is_forwarded(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(2))])
    sink = Sink()

    await run(env, sink=sink)

    assert "workflow reel · pic 1/2 · halfway" in sink.progress
    assert "workflow reel · video 1/1 · halfway" in sink.progress
    assert "workflow reel · asking the model · story" in sink.progress


async def test_a_failing_call_stops_the_run(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(2))])
    env.voice.fail_on = {f"{FOLDER}/v2.wav"}
    sink = Sink()

    result = await run(env, sink=sink)

    assert not result.ok
    assert result.line == (
        "workflow reel failed at voice 2/2: it broke; /workflow resume to continue"
    )
    assert env.video.calls == []
    manifest = load(env.run_folder)
    assert manifest is not None
    assert [c["state"] for c in manifest["calls"]] == ["done", "done", "done", "failed", "pending"]
    assert manifest["calls"][3]["summary"] == "failed"
    assert sink.lines[-1][0] == "warn" and sink.lines[-1][1].startswith("✗ speak")
    assert indexed_state(env) == "failed"


async def test_cancelling_a_slow_call_marks_it_failed_saves_the_manifest_and_reraises(tmp_path: Path) -> None:
    gate = Gate()
    env = build(tmp_path, [answer(scenes(1))], extra=[SlowTool(gate)])
    data = reel_data()
    data["steps"].append({"id": "last", "kind": "tool", "tool": "slow", "args": {"path": "{folder}/slow.txt"}})
    task = asyncio.ensure_future(run(env, workflow(data)))

    await asyncio.wait_for(gate.reached.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    manifest = load(env.run_folder)
    assert manifest is not None
    assert [c["state"] for c in manifest["calls"]] == ["done", "done", "done", "failed"]
    assert "interrupted" in manifest["calls"][3]["summary"]
    assert indexed_state(env) == "failed"
    assert not env.agent.busy


async def test_run_raises_when_the_agent_is_busy(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1))])

    async with env.agent.operation("x", Sink()):
        with pytest.raises(AgentBusy):
            await run(env)

    assert env.provider.requests == []


# --- AC8: resume ------------------------------------------------------------------------------


async def failed_run(tmp_path: Path, scene_count: int = 2) -> tuple[Env, Workflow]:
    env = build(tmp_path, [answer(scenes(scene_count))])
    env.voice.fail_on = {f"{FOLDER}/v2.wav"}
    wf = workflow()
    await run(env, wf)
    env.voice.fail_on = set()
    for tool in (env.img, env.voice, env.video):
        tool.calls.clear()
    return env, wf


async def test_resume_runs_only_the_calls_still_pending(tmp_path: Path) -> None:
    env, wf = await failed_run(tmp_path)
    requests_before = len(env.provider.requests)

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert result.ok, result.line
    assert result.line == f"workflow reel finished: 2 calls, files in {FOLDER}"
    assert [Path(c["path"]).name for c in env.voice.calls] == ["v2.wav"]
    assert [Path(c["path"]).name for c in env.video.calls] == ["reel.mp4"]
    assert env.img.calls == []
    assert len(env.provider.requests) == requests_before  # the model steps did not run again
    manifest = load(env.run_folder)
    assert manifest is not None and [c["state"] for c in manifest["calls"]] == ["done"] * 5
    assert [len(p.calls) for p in env.approved] == [5, 2]  # the plan shown again holds the remaining calls
    assert env.approver.requests == []


async def test_resume_by_run_id(tmp_path: Path) -> None:
    env, wf = await failed_run(tmp_path)

    result = await env.runner.resume(lookup(wf), RUN[:12], Sink(), env.approve)

    assert result.ok, result.line


async def test_a_done_call_whose_output_was_edited_runs_again(tmp_path: Path) -> None:
    env, wf = await failed_run(tmp_path)
    (env.run_folder / "s1.png").write_text("edited by hand")

    await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert [Path(c["path"]).name for c in env.img.calls] == ["s1.png"]
    assert (env.run_folder / "s1.png").read_text() == "p1"


async def test_a_done_call_with_a_deleted_output_runs_again(tmp_path: Path) -> None:
    env, wf = await failed_run(tmp_path)
    (env.run_folder / "v1.wav").unlink()

    await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert [Path(c["path"]).name for c in env.voice.calls] == ["v1.wav", "v2.wav"]


async def test_resume_is_refused_when_the_model_steps_changed(tmp_path: Path) -> None:
    env, _ = await failed_run(tmp_path)
    data = reel_data()
    data["steps"][0]["prompt"] = "Write different scenes about {input}"

    result = await env.runner.resume(lookup(workflow(data)), None, Sink(), env.approve)

    assert not result.ok
    assert result.line == "the workflow's model steps changed; start a new run"
    assert env.voice.calls == []


async def test_resume_is_refused_when_the_tool_steps_changed(tmp_path: Path) -> None:
    env, _ = await failed_run(tmp_path)
    data = reel_data()
    data["steps"][2]["args"]["prompt"] = "different"

    result = await env.runner.resume(lookup(workflow(data)), None, Sink(), env.approve)

    assert not result.ok
    assert result.line == "the workflow's tool steps changed; start a new run"


async def test_resume_with_everything_done_says_so(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1))])
    wf = workflow()
    await run(env, wf)

    result = await env.runner.resume(lookup(wf), RUN, Sink(), env.approve)

    assert result.line == "nothing to resume"
    assert result.ok
    assert len(env.approved) == 1


async def test_resume_without_a_run_says_so(tmp_path: Path) -> None:
    env = build(tmp_path, [])

    result = await env.runner.resume(lookup(workflow()), None, Sink(), env.approve)

    assert not result.ok
    assert result.line == "no unfinished workflow run to resume"


async def test_resume_of_an_unknown_run_id(tmp_path: Path) -> None:
    env = build(tmp_path, [])

    result = await env.runner.resume(lookup(workflow()), "nope", Sink(), env.approve)

    assert result.line == "no workflow run 'nope'"


async def test_resume_needs_the_workflow_to_exist(tmp_path: Path) -> None:
    env, _ = await failed_run(tmp_path)

    result = await env.runner.resume(lambda name: None, None, Sink(), env.approve)

    assert result.line == "workflow reel is not available"


async def test_resume_declined_runs_nothing(tmp_path: Path) -> None:
    env, wf = await failed_run(tmp_path)
    env.answer_yes = False

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert not result.ok
    assert env.voice.calls == []


async def test_a_run_cancelled_mid_call_resumes_with_that_call_and_the_ones_after(tmp_path: Path) -> None:
    gate = Gate()
    slow = SlowTool(gate)
    env = build(tmp_path, [answer(scenes(1))], extra=[slow])
    data = reel_data()
    data["steps"].append({"id": "last", "kind": "tool", "tool": "slow", "args": {"path": "{folder}/slow.txt"}})
    wf = workflow(data)
    task = asyncio.ensure_future(run(env, wf))
    await asyncio.wait_for(gate.reached.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    slow.calls.clear()
    gate.release.set()
    for tool in (env.img, env.voice, env.video):
        tool.calls.clear()

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert result.ok, result.line
    assert [Path(c["path"]).name for c in slow.calls] == ["slow.txt"]
    assert env.img.calls == env.voice.calls == env.video.calls == []


async def test_resume_notices_a_manifest_edited_to_leave_the_folder(tmp_path: Path) -> None:
    env, wf = await failed_run(tmp_path)
    manifest = load(env.run_folder)
    assert manifest is not None
    manifest["calls"][3]["args"]["path"] = "../../escape.wav"
    save(env.run_folder, manifest)

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert not result.ok
    assert env.voice.calls == []
    assert not (env.workspace / "escape.wav").exists()


# --- AC10: undo -------------------------------------------------------------------------------


async def test_a_run_is_one_undo_unit(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(2))])

    result = await run(env)

    assert result.ok, result.line
    diff = await env.agent.last_turn_diff()
    assert diff is not None
    assert {i.path for i in diff.items} == {
        f"{FOLDER}/{n}" for n in ("s1.png", "s2.png", "v1.wav", "v2.wav", "reel.mp4")
    }
    assert env.agent.undo_log.peek() is not None

    prompt = await env.agent.undo(Sink())

    assert prompt == "/workflow run reel cats"
    for name in ("s1.png", "s2.png", "v1.wav", "v2.wav", "reel.mp4"):
        assert not (env.run_folder / name).exists()
    assert env.agent.undo_log.peek() is None  # one unit: nothing left to undo


async def test_undoing_a_resume_restores_the_file_it_overwrote(tmp_path: Path) -> None:
    env, wf = await failed_run(tmp_path)
    (env.run_folder / "s1.png").write_text("edited by hand")

    await env.runner.resume(lookup(wf), None, Sink(), env.approve)
    assert (env.run_folder / "s1.png").read_text() == "p1"

    await env.agent.undo(Sink())

    assert (env.run_folder / "s1.png").read_text() == "edited by hand"


async def test_resume_is_a_new_undo_unit(tmp_path: Path) -> None:
    env, wf = await failed_run(tmp_path)

    await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert await env.agent.undo(Sink()) == f"/workflow resume {RUN}"
    assert (env.run_folder / "s1.png").exists()  # the first run's files stay
    assert not (env.run_folder / "reel.mp4").exists()
    assert await env.agent.undo(Sink()) == "/workflow run reel cats"
    assert not (env.run_folder / "s1.png").exists()


async def test_the_manifest_hashes_each_output(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1))])

    await run(env)

    manifest = load(env.run_folder)
    assert manifest is not None
    first = manifest["calls"][0]
    assert first["output_sha256"] == file_sha256(env.run_folder / "s1.png")
    assert manifest["model_steps_sha256"] and manifest["outputs"]["story"]["scenes"][0]["narration"] == "n1"
    assert manifest["folder"] == FOLDER and manifest["input"] == "cats"


# --- review fixes -----------------------------------------------------------------------------


def two_model_steps() -> dict[str, Any]:
    data = reel_data()
    data["steps"].insert(
        1, {"id": "titles", "kind": "model", "prompt": "{story.scenes}", "schema": {"type": "object"}}
    )
    return data


async def test_a_prompt_that_is_one_placeholder_gets_the_json_text(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1)), answer("{}")])

    result = await run(env, workflow(two_model_steps()))

    assert result.ok, result.line
    sent = env.provider.requests[1][0][1].content
    assert sent == json.dumps(json.loads(scenes(1))["scenes"], ensure_ascii=False)
    assert "'" not in sent


@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity", '{"a": NaN}', "[1, Infinity]", "x [NaN] y"])
def test_first_json_refuses_non_standard_constants(text: str) -> None:
    from src.engine.workflows.runner import first_json

    assert first_json(text) == (False, None)


def test_first_json_survives_very_deep_nesting_quickly() -> None:
    from src.engine.workflows.runner import first_json

    assert first_json("[" * 200_000) == (False, None)
    assert first_json("[" * 5_000 + "]" * 5_000) == (False, None)
    assert first_json("{" * 100_000) == (False, None)


def test_first_json_still_finds_a_value_after_junk() -> None:
    from src.engine.workflows.runner import first_json

    assert first_json("a [ b {\"k\": 1} c") == (True, {"k": 1})
    assert first_json('say "hi" then [1, 2]') == (True, [1, 2])


async def test_a_nan_answer_is_not_taken(tmp_path: Path) -> None:
    bad = '{"scenes": [{"image_prompt": "p", "narration": NaN}]}'
    env = build(tmp_path, [answer(bad), answer(bad)])

    result = await run(env)

    assert not result.ok
    assert "the answer is not JSON" in result.line


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m["calls"][0].pop("state"),
        lambda m: m["calls"][0].update(state=5),
        lambda m: m["calls"][0].update(state="weird"),
        lambda m: m["calls"][0].pop("key"),
        lambda m: m["calls"][0].update(key=3),
        lambda m: m["calls"][0].update(output_sha256=5),
        lambda m: m["calls"][0].update(summary=None),
        lambda m: m["calls"].__setitem__(0, "x"),
        lambda m: m.update(outputs=[]),
        lambda m: m.update(outputs={}),
        lambda m: m.update(input=5),
        lambda m: m.pop("outputs"),
        lambda m: m.pop("model_steps_sha256"),
    ],
)
async def test_a_malformed_manifest_is_refused_cleanly(tmp_path: Path, mutate: Any) -> None:
    env, wf = await failed_run(tmp_path)
    manifest = load(env.run_folder)
    assert manifest is not None
    mutate(manifest)
    save(env.run_folder, manifest)

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert not result.ok
    assert result.line in (
        "the manifest was changed by hand; start a new run",
        "the workflow's model steps changed; start a new run",
    )
    assert env.voice.calls == []


async def test_a_done_call_edited_by_hand_reruns_the_calls_built_from_it(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(2))])
    wf = workflow()
    await run(env, wf)
    for tool in (env.img, env.voice, env.video):
        tool.calls.clear()
    (env.run_folder / "s1.png").write_text("edited by hand")

    result = await env.runner.resume(lookup(wf), RUN, Sink(), env.approve)

    assert result.ok, result.line
    assert [Path(c["path"]).name for c in env.img.calls] == ["s1.png"]
    assert [Path(c["path"]).name for c in env.video.calls] == ["reel.mp4"]  # its scenes name s1.png
    assert env.voice.calls == []


def named(name: str) -> Workflow:
    return parse_workflow(name, "project", Path(f"/w/{name}.json"), json.dumps(reel_data()).encode())


async def test_two_workflows_in_the_same_minute_keep_separate_resumable_runs(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(2)), answer(scenes(2))])
    env.voice.fail_on = {f"workflows/reel/{RUN}/v2.wav", f"workflows/other/{RUN}/v2.wav"}
    first, second = named("reel"), named("other")

    a = await env.runner.run(first, "cats", Sink(), env.approve)
    b = await env.runner.run(second, "cats", Sink(), env.approve)
    env.voice.fail_on = set()

    assert a.run == b.run == RUN and a.folder != b.folder
    both = {"reel": first, "other": second}
    resumed_b = await env.runner.resume(both.get, None, Sink(), env.approve)  # type: ignore[arg-type]
    resumed_a = await env.runner.resume(both.get, None, Sink(), env.approve)  # type: ignore[arg-type]
    assert resumed_b.ok and resumed_b.folder == b.folder
    assert resumed_a.ok and resumed_a.folder == a.folder


async def test_cancelling_the_approval_leaves_nothing_behind(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1))])
    waiting = asyncio.Event()

    async def approve(plan: PlanSummary) -> bool:
        waiting.set()
        await asyncio.Event().wait()
        return True

    task = asyncio.ensure_future(env.runner.run(workflow(), "cats", Sink(), approve))
    await asyncio.wait_for(waiting.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not env.agent.busy
    assert not env.run_folder.exists()
    assert env.index.find(RUN) is None
    assert env.agent.undo_log.peek() is None
    assert env.img.calls == []


async def test_cancelling_a_model_step_leaves_nothing_behind(tmp_path: Path) -> None:
    gate = Gate()
    env = build(tmp_path, [[gate, *answer(scenes(1))]])
    task = asyncio.ensure_future(run(env))

    await asyncio.wait_for(gate.reached.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not env.agent.busy
    assert not env.run_folder.exists()
    assert env.index.find(RUN) is None
    assert env.approved == []
    assert env.agent.undo_log.peek() is None


async def test_saved_is_false_when_cancelled_before_anything_was_written(tmp_path: Path) -> None:
    gate = Gate()
    env = build(tmp_path, [[gate, *answer(scenes(1))]])
    task = asyncio.ensure_future(run(env))
    await asyncio.wait_for(gate.reached.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert env.runner.saved is False


async def test_saved_is_true_when_cancelled_mid_call(tmp_path: Path) -> None:
    gate = Gate()
    env = build(tmp_path, [answer(scenes(1))], extra=[SlowTool(gate)])
    data = reel_data()
    data["steps"].append({"id": "last", "kind": "tool", "tool": "slow", "args": {"path": "{folder}/slow.txt"}})
    task = asyncio.ensure_future(run(env, workflow(data)))
    await asyncio.wait_for(gate.reached.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert env.runner.saved is True


async def test_a_resume_finds_the_older_run_when_the_newest_has_no_manifest(tmp_path: Path) -> None:
    env, wf = await failed_run(tmp_path)
    env.index.record("later", "reel", "workflows/reel/later", "failed")  # e.g. its run was undone

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert result.ok, result.line
    assert result.run == RUN


async def test_a_declined_plan_leaves_no_undo_unit(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1))])
    env.answer_yes = False

    await run(env)

    assert env.agent.undo_log.peek() is None


async def test_a_plan_that_fails_leaves_no_undo_unit(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1))], skip=(Speak,))

    await run(env)

    assert env.agent.undo_log.peek() is None


async def test_a_declined_resume_leaves_no_new_undo_unit(tmp_path: Path) -> None:
    env, wf = await failed_run(tmp_path)
    env.answer_yes = False

    await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert await env.agent.undo(Sink()) == "/workflow run reel cats"  # the run's unit, not a resume's


async def test_the_folder_is_escaped_in_the_result_line(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1))])
    data = reel_data(folder="o\x1b[31mut/{run}")

    result = await run(env, workflow(data))

    assert result.ok, result.line
    assert "\x1b" not in result.line


async def test_a_manifest_save_that_fails_is_a_clean_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.engine.workflows.runner as runner_module

    env = build(tmp_path, [answer(scenes(1))])
    real = runner_module.save
    saves: list[int] = []

    def flaky(folder: Path, manifest: dict[str, Any]) -> None:
        saves.append(1)
        if len(saves) > 2:
            raise OSError(28, "No space left on device")
        real(folder, manifest)

    monkeypatch.setattr(runner_module, "save", flaky)

    result = await run(env)

    assert not result.ok
    assert "manifest" in result.line and "No space left" in result.line
    assert not env.agent.busy


async def test_the_first_manifest_save_failing_is_a_clean_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.engine.workflows.runner as runner_module

    env = build(tmp_path, [answer(scenes(1))])

    def broken(folder: Path, manifest: dict[str, Any]) -> None:
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(runner_module, "save", broken)

    result = await run(env)

    assert not result.ok and "manifest" in result.line
    assert env.img.calls == []


async def test_a_failing_save_on_cancel_does_not_hide_the_cancel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.engine.workflows.runner as runner_module

    gate = Gate()
    env = build(tmp_path, [answer(scenes(1))], extra=[SlowTool(gate)])
    data = reel_data()
    data["steps"].append({"id": "last", "kind": "tool", "tool": "slow", "args": {"path": "{folder}/slow.txt"}})
    real = runner_module.save
    broken = False

    def flaky(folder: Path, manifest: dict[str, Any]) -> None:
        if broken:
            raise OSError(28, "No space left on device")
        real(folder, manifest)

    monkeypatch.setattr(runner_module, "save", flaky)
    task = asyncio.ensure_future(run(env, workflow(data)))
    await asyncio.wait_for(gate.reached.wait(), 5)
    broken = True
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert not env.agent.busy
