"""Retry on resume: the error is saved, `resume` asks the model again, re-plans and reuses finished calls."""

from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path
from typing import Any, ClassVar

import pytest
from src.engine.provider.base import ProviderError
from src.engine.workflows.estimates import DEFAULT_RATE
from src.engine.workflows.manifest import load, save
from src.engine.workflows.model import PlanSummary, Workflow
from src.engine.workflows.timings import TimingStore

from tests.engine.workflows.fakes import WriteFileTool
from tests.engine.workflows.test_runner import (
    FOLDER,
    RUN,
    Env,
    Sink,
    answer,
    build,
    indexed_state,
    lookup,
    reel_data,
    run,
    scenes,
    workflow,
)


async def test_an_invalid_failure_records_the_error_and_says_resume_asks_the_model(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(2))])
    env.video.invalid_on = {f"{FOLDER}/reel.mp4": "scene 2 is 0.8 s; at least 1 s\nsecond line"}

    result = await run(env)

    assert result.line == (
        "workflow reel failed at video 1/1: scene 2 is 0.8 s; at least 1 s; /workflow resume asks the model again"
    )
    manifest = load(env.run_folder)
    assert manifest is not None
    assert manifest["calls"][-1]["error"] == "scene 2 is 0.8 s; at least 1 s"
    assert manifest["calls"][-1]["summary"] == "invalid"


async def test_a_plain_failure_keeps_the_old_ending(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(2))])
    env.voice.fail_on = {f"{FOLDER}/v2.wav"}

    result = await run(env)

    assert result.line.endswith("/workflow resume to continue")


async def test_an_invalid_failure_of_a_constant_step_keeps_the_old_ending(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1))])
    data = reel_data()
    data["steps"].append({"id": "last", "kind": "tool", "tool": "generate_image", "args": {"path": "{folder}/c.png"}})
    env.img.invalid_on = {f"{FOLDER}/c.png": "bad"}

    result = await run(env, workflow(data))

    assert result.line.endswith("/workflow resume to continue")


ERR = "scene 2 is 0.8 s; at least 1 s"


def scenes_with(narrations: list[str]) -> str:
    return json.dumps(
        {"scenes": [{"image_prompt": f"p{i}", "narration": n} for i, n in enumerate(narrations, 1)]}
    )


async def invalid_run(tmp_path: Path, second: str, error: str = ERR, **kwargs: Any) -> tuple[Env, Workflow]:
    """A 3-scene run whose video was refused as `invalid`; the provider will answer `second` to the re-ask."""
    env = build(tmp_path, [answer(scenes(3)), answer(second)], **kwargs)
    env.video.invalid_on = {f"{FOLDER}/reel.mp4": error}
    wf = workflow()
    await run(env, wf)
    env.video.invalid_on = {}
    clear(env)
    env.approved.clear()
    return env, wf


def clear(env: Env) -> None:
    for tool in (env.img, env.voice, env.video):
        tool.calls.clear()


def names(tool: Any) -> list[str]:
    return [Path(c["path"]).name for c in tool.calls]


async def test_resume_asks_only_the_story_with_prompt_answer_and_error(tmp_path: Path) -> None:
    env, wf = await invalid_run(tmp_path, scenes_with(["n1", "a longer narration", "n3"]))

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert result.ok, result.line
    assert len(env.provider.requests) == 2
    messages = env.provider.requests[1][0]
    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
    assert "only one JSON value" in messages[0].content
    assert messages[1].content == "Write scenes about cats"
    assert json.loads(messages[2].content) == json.loads(scenes(3))
    assert f"The tool assemble_video refused the call video: {ERR}" in messages[3].content


async def test_a_changed_narration_runs_only_that_voice_and_the_video(tmp_path: Path) -> None:
    env, wf = await invalid_run(tmp_path, scenes_with(["n1", "a longer narration", "n3"]))
    sink = Sink()

    result = await env.runner.resume(lookup(wf), None, sink, env.approve)

    assert result.ok, result.line
    plan = env.approved[0]
    assert [c.key for c in plan.calls] == ["voice/2", "video"]
    assert plan.reused == 5
    assert plan.notes == (f"re-asked story: {ERR}",)
    assert env.img.calls == []
    assert names(env.voice) == ["v2.wav"] and names(env.video) == ["reel.mp4"]
    assert "asking the model" in " ".join(sink.progress) and "story (retry)" in " ".join(sink.progress)
    manifest = load(env.run_folder)
    assert manifest is not None
    assert manifest["outputs"]["story"]["scenes"][1]["narration"] == "a longer narration"
    assert all(c["state"] == "done" for c in manifest["calls"]) and all("error" not in c for c in manifest["calls"])
    again = await env.runner.resume(lookup(wf), RUN, Sink(), env.approve)
    assert again.line == "nothing to resume"


async def test_four_scenes_instead_of_three_runs_the_new_calls_only(tmp_path: Path) -> None:
    env, wf = await invalid_run(tmp_path, scenes(4))

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert result.ok, result.line
    assert [c.key for c in env.approved[0].calls] == ["pic/4", "voice/4", "video"]
    assert names(env.img) == ["s4.png"] and names(env.voice) == ["v4.wav"]
    manifest = load(env.run_folder)
    assert manifest is not None
    assert [c["key"] for c in manifest["calls"]][-1] == "video" and len(manifest["calls"]) == 9


async def test_three_scenes_to_two_drops_the_saved_calls_that_no_longer_exist(tmp_path: Path) -> None:
    env, wf = await invalid_run(tmp_path, scenes(2))

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert result.ok, result.line
    manifest = load(env.run_folder)
    assert manifest is not None
    assert [c["key"] for c in manifest["calls"]] == ["pic/1", "pic/2", "voice/1", "voice/2", "video"]


async def test_a_declined_redo_leaves_the_manifest_and_the_undo_log_alone(tmp_path: Path) -> None:
    env, wf = await invalid_run(tmp_path, scenes(4))
    before = (env.run_folder / "manifest.json").read_bytes()
    env.answer_yes = False

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert not result.ok and result.line == "workflow reel: the plan was not approved"
    assert (env.run_folder / "manifest.json").read_bytes() == before
    assert env.img.calls == env.voice.calls == env.video.calls == []
    assert await env.agent.undo(Sink()) == "/workflow run reel cats"
    assert indexed_state(env) == "failed"


async def test_a_declined_redo_asks_again_on_the_next_resume(tmp_path: Path) -> None:
    env, wf = await invalid_run(tmp_path, scenes(4))
    env.provider.script.append(answer(scenes(4)))
    env.answer_yes = False
    await env.runner.resume(lookup(wf), None, Sink(), env.approve)
    env.answer_yes = True

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert result.ok, result.line
    assert len(env.provider.requests) == 3


async def test_a_cancel_at_the_dialog_leaves_the_manifest_alone(tmp_path: Path) -> None:
    env, wf = await invalid_run(tmp_path, scenes(4))
    before = (env.run_folder / "manifest.json").read_bytes()

    async def esc(plan: PlanSummary) -> bool:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await env.runner.resume(lookup(wf), None, Sink(), esc)

    assert (env.run_folder / "manifest.json").read_bytes() == before
    assert not env.runner.saved and not env.agent.busy
    assert await env.agent.undo(Sink()) == "/workflow run reel cats"


async def test_an_unusable_new_answer_leaves_the_manifest_alone(tmp_path: Path) -> None:
    env, wf = await invalid_run(tmp_path, "nonsense")
    env.provider.script.append(answer("still nonsense"))
    before = (env.run_folder / "manifest.json").read_bytes()

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert not result.ok and result.line.startswith("workflow reel failed at story:")
    assert (env.run_folder / "manifest.json").read_bytes() == before
    assert env.approved == []


async def test_a_provider_error_in_the_reask_leaves_the_manifest_alone(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(3)), [ProviderError("down")]])
    env.video.invalid_on = {f"{FOLDER}/reel.mp4": ERR}
    wf = workflow()
    await run(env, wf)
    before = (env.run_folder / "manifest.json").read_bytes()

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert result.line == "workflow reel failed at story: down"
    assert (env.run_folder / "manifest.json").read_bytes() == before


async def test_a_redo_plan_that_fails_the_checks_leaves_the_manifest_alone(tmp_path: Path) -> None:
    env, _ = await invalid_run(tmp_path, scenes(4))
    before = (env.run_folder / "manifest.json").read_bytes()
    data = reel_data(pic_path="{item.image_prompt}.png")  # a path built from model text leaves the run folder
    bad = workflow(data)

    result = await env.runner.resume(lambda n: bad if n == "reel" else None, None, Sink(), env.approve)

    assert not result.ok
    assert (env.run_folder / "manifest.json").read_bytes() == before


async def test_a_timeout_reruns_the_call_without_asking_the_model(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(2))])
    env.video.timeout_on = {f"{FOLDER}/reel.mp4"}
    wf = workflow()
    result = await run(env, wf)
    assert result.line.endswith("/workflow resume to continue")
    env.video.timeout_on = set()
    clear(env)

    again = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert again.ok, again.line
    assert len(env.provider.requests) == 1
    assert names(env.video) == ["reel.mp4"] and env.img.calls == env.voice.calls == []


async def test_redo_asks_the_story_again_on_a_finished_run_with_the_user_note(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(2)), answer(scenes_with(["n1", "different"]))])
    wf = workflow()
    await run(env, wf)
    clear(env)
    env.approved.clear()

    result = await env.runner.resume(lookup(wf), RUN, Sink(), env.approve, redo="story")

    assert result.ok, result.line
    messages = env.provider.requests[1][0]
    assert messages[3].content.startswith("The user asked for a new answer.")
    assert env.approved[0].notes == ("re-asked story: asked by the user",)
    assert [c.key for c in env.approved[0].calls] == ["voice/2"]  # the fake video does not take the narrations
    assert names(env.voice) == ["v2.wav"]
    assert indexed_state(env) == "done"


async def test_redo_with_the_same_answer_changes_nothing(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(2)), answer(scenes(2))])
    wf = workflow()
    await run(env, wf)
    before = (env.run_folder / "manifest.json").read_bytes()
    env.approved.clear()

    result = await env.runner.resume(lookup(wf), RUN, Sink(), env.approve, redo="story")

    assert result.ok and result.line == "nothing changed; nothing to run"
    assert env.approved == []
    assert (env.run_folder / "manifest.json").read_bytes() == before


async def test_redo_without_an_id_falls_back_to_the_latest_finished_run(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(2)), answer(scenes_with(["n1", "different"]))])
    wf = workflow()
    await run(env, wf)
    assert indexed_state(env) == "done"
    clear(env)
    env.approved.clear()

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve, redo="story")

    assert result.ok, result.line
    assert result.run == RUN and names(env.voice) == ["v2.wav"]


async def test_resume_without_an_id_and_without_redo_still_needs_an_unfinished_run(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(1))])
    await run(env)

    result = await env.runner.resume(lookup(workflow()), None, Sink(), env.approve)

    assert not result.ok and result.line == "no unfinished workflow run to resume"


async def test_esc_in_the_reask_after_a_declined_redo_leaves_the_manifest_alone(tmp_path: Path) -> None:
    from tests.fakes import Gate

    gate = Gate()
    env, wf = await invalid_run(tmp_path, scenes(4))
    before = (env.run_folder / "manifest.json").read_bytes()
    env.provider.script.append([gate])  # the first re-ask used the answer `invalid_run` queued
    env.answer_yes = False
    await env.runner.resume(lookup(wf), None, Sink(), env.approve)
    assert (env.run_folder / "manifest.json").read_bytes() == before

    task = asyncio.ensure_future(env.runner.resume(lookup(wf), None, Sink(), env.approve))
    await gate.reached.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (env.run_folder / "manifest.json").read_bytes() == before
    assert not env.runner.saved and not env.agent.busy
    assert await env.agent.undo(Sink()) == "/workflow run reel cats"


# --- the dialog says what else changed --------------------------------------------------------


async def test_the_redo_dialog_says_when_tool_steps_changed_since_the_run(tmp_path: Path) -> None:
    env, _ = await invalid_run(tmp_path, scenes(3))
    moved =workflow(reel_data(pic_path="{folder}/img{index}.png"))

    result = await env.runner.resume(lookup(moved), None, Sink(), env.approve)

    assert result.ok, result.line
    assert "tool steps changed since this run" in env.approved[0].notes
    assert not any("dropped" in note for note in env.approved[0].notes)  # the keys are the same, only digests differ
    assert env.approved[0].notes[0].startswith("re-asked story")


async def test_the_redo_dialog_says_how_many_saved_calls_were_dropped(tmp_path: Path) -> None:
    env, wf = await invalid_run(tmp_path, scenes(2))

    await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert env.approved[0].notes[-1] == "2 saved calls dropped"
    assert "tool steps changed since this run" not in env.approved[0].notes


async def test_nothing_pending_but_dropped_entries_still_asks_and_saves_the_cleaned_manifest(tmp_path: Path) -> None:
    script = [answer(scenes(2)), answer('{"c": "one"}'), answer(scenes(2)), answer('{"c": "one"}')]
    env = build(tmp_path, script, extra=[WriteFileTool()])
    wf = workflow(chained_data())
    await run(env, wf)
    env.approved.clear()
    trimmed = chained_data()
    trimmed["steps"] = [s for s in trimmed["steps"] if s["id"] != "cap"]
    smaller = workflow(trimmed)

    result = await env.runner.resume(lookup(smaller), RUN, Sink(), env.approve, redo="story")

    assert result.ok, result.line
    assert len(env.approved) == 1 and env.approved[0].calls == ()
    assert "1 saved call dropped" in env.approved[0].notes
    assert "tool steps changed since this run" in env.approved[0].notes
    manifest = load(env.run_folder)
    assert manifest is not None and "cap" not in [c["key"] for c in manifest["calls"]]


async def test_nothing_pending_but_dropped_entries_declined_leaves_the_manifest_alone(tmp_path: Path) -> None:
    script = [answer(scenes(2)), answer('{"c": "one"}'), answer(scenes(2)), answer('{"c": "one"}')]
    env = build(tmp_path, script, extra=[WriteFileTool()])
    wf = workflow(chained_data())
    await run(env, wf)
    before = (env.run_folder / "manifest.json").read_bytes()
    trimmed = chained_data()
    trimmed["steps"] = [s for s in trimmed["steps"] if s["id"] != "cap"]
    env.answer_yes = False

    result = await env.runner.resume(lookup(workflow(trimmed)), RUN, Sink(), env.approve, redo="story")

    assert not result.ok and result.line == "workflow reel: the plan was not approved"
    assert (env.run_folder / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("step", ["pic", "nope"])
async def test_redo_of_a_step_that_is_not_a_model_step_is_an_error_listing_them(tmp_path: Path, step: str) -> None:
    env = build(tmp_path, [answer(scenes(1))])
    wf = workflow()
    await run(env, wf)

    result = await env.runner.resume(lookup(wf), RUN, Sink(), env.approve, redo=step)

    assert not result.ok
    assert result.line == f"no model step '{step}' in workflow reel (model steps: story)"
    assert len(env.provider.requests) == 1


async def test_a_hostile_error_is_escaped_in_the_dialog_and_redacted_in_the_request(tmp_path: Path) -> None:
    token = "ghp_" + "a" * 36
    hostile = f"bad \x1b[31m‮ {token}"
    env, wf = await invalid_run(tmp_path, scenes(4), error=hostile)
    sink = Sink()

    await env.runner.resume(lookup(wf), None, sink, env.approve)

    request = env.provider.requests[1][0][3].content
    assert token not in request and "[redacted:github-token]" in request
    note = env.approved[0].notes[0]
    assert token not in note and "\x1b" not in note and "‮" not in note
    assert {c.key for c in env.approved[0].calls} == {"pic/4", "voice/4", "video"}


async def test_an_old_manifest_without_an_error_retries_with_the_fallback_text(tmp_path: Path) -> None:
    env, wf = await invalid_run(tmp_path, scenes(4))
    manifest = load(env.run_folder)
    assert manifest is not None
    for c in manifest["calls"]:
        c.pop("error", None)
    save(env.run_folder, manifest)

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert result.ok, result.line
    assert "refused the call video: the tool refused the call" in env.provider.requests[1][0][3].content
    assert env.approved[0].notes == ("re-asked story: the tool refused the call",)


async def test_a_redo_is_one_undo_unit_of_the_calls_that_ran(tmp_path: Path) -> None:
    env, wf = await invalid_run(tmp_path, scenes_with(["n1", "a longer narration", "n3"]))
    await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    diff = await env.agent.last_turn_diff()

    assert diff is not None
    assert {i.path for i in diff.items} == {f"{FOLDER}/v2.wav", f"{FOLDER}/reel.mp4"}
    assert await env.agent.undo(Sink()) == f"/workflow resume {RUN}"
    assert (env.run_folder / "v2.wav").read_text() == "n2"


class BrokenStore(TimingStore):
    def add(self, key: str, value: float) -> None:
        raise OSError(28, "No space left on device")


# --- timings ------------------------------------------------------------------------------------


async def test_successful_calls_are_timed_per_unit_and_failures_are_not(tmp_path: Path) -> None:
    store = TimingStore(None)
    env = build(tmp_path, [answer(scenes(3))], timings=store)
    env.img.fail_on = {f"{FOLDER}/s2.png"}

    await run(env)

    assert len(store.values("generate_image")) == 1  # s1 only; the run stopped at s2
    assert store.values("speak") == []


async def test_all_successes_are_recorded_normalised(tmp_path: Path) -> None:
    store = TimingStore(None)
    env = build(tmp_path, [answer(scenes(2))], timings=store)

    result = await run(env)

    assert result.ok, result.line
    assert len(store.values("generate_image")) == 2
    assert len(store.values("speak")) == 2
    assert len(store.values("assemble_video")) == 1
    assert all(0 < v < 1e-3 for v in store.values("generate_image"))  # seconds per pixel x step


async def test_a_learned_estimate_reaches_the_plan(tmp_path: Path) -> None:
    store = TimingStore(None)
    for _ in range(2):
        store.add("speak", 8.0)
    env = build(tmp_path, [answer(scenes(1))], timings=store)

    await run(env)

    assert env.approved[0].estimate_source.startswith("partly learned")


async def test_a_failing_timing_store_gives_one_warning_and_the_run_goes_on(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(2))], timings=BrokenStore(None))
    sink = Sink()

    result = await run(env, sink=sink)

    assert result.ok, result.line
    warnings = [t for level, t in sink.lines if t.startswith("could not save the call timings")]
    assert warnings == ["could not save the call timings: No space left on device"]


def chained_data() -> dict[str, Any]:
    data = reel_data()
    data["steps"].insert(
        1,
        {
            "id": "caption",
            "kind": "model",
            "prompt": "One caption for {story.scenes}",
            "schema": {"type": "object", "properties": {"c": {"type": "string"}}, "required": ["c"]},
        },
    )
    data["steps"].append({"id": "cap", "kind": "tool", "tool": "write_file", "args": {"path": "{folder}/cap.txt", "prompt": "{caption.c}"}})
    return data


async def test_a_dependent_model_step_is_told_an_earlier_answer_was_corrected(tmp_path: Path) -> None:
    script = [answer(scenes(3)), answer('{"c": "one"}'), answer(scenes(3)), answer('{"c": "two"}')]
    env = build(tmp_path, script, extra=[WriteFileTool()])
    env.voice.invalid_on = {f"{FOLDER}/v2.wav": ERR}
    wf = workflow(chained_data())
    await run(env, wf)
    env.voice.invalid_on = {}

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert result.ok, result.line
    story_feedback = env.provider.requests[2][0][3].content
    caption_feedback = env.provider.requests[3][0][3].content
    assert "The tool speak refused the call voice/2" in story_feedback
    assert "earlier answer" in caption_feedback and ERR in caption_feedback
    assert "fix the problem" not in caption_feedback


async def test_a_dependent_model_step_after_a_user_redo_is_told_the_earlier_answer_changed(tmp_path: Path) -> None:
    script = [answer(scenes(3)), answer('{"c": "one"}'), answer(scenes(3)), answer('{"c": "one"}')]
    env = build(tmp_path, script, extra=[WriteFileTool()])
    wf = workflow(chained_data())
    await run(env, wf)

    await env.runner.resume(lookup(wf), RUN, Sink(), env.approve, redo="story")

    assert "The user asked for a new answer" in env.provider.requests[2][0][3].content
    assert "asked again" in env.provider.requests[3][0][3].content


async def test_a_huge_hand_edited_error_is_capped_and_redacted_in_the_request_and_the_dialog(tmp_path: Path) -> None:
    token = "ghp_" + "b" * 36
    env, wf = await invalid_run(tmp_path, scenes(4))
    manifest = load(env.run_folder)
    assert manifest is not None
    last = [c for c in manifest["calls"] if c["state"] == "failed"][-1]
    last["error"] = "x" * 100_000 + token
    last["tool"] = token + "t" * 100_000
    last["key"] = "video/" + "k" * 100_000 + token
    save(env.run_folder, manifest)

    await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    request = env.provider.requests[1][0][3].content
    assert len(request) < 2_000 and token not in request
    assert all(len(n) < 1_000 and token not in n for n in env.approved[0].notes)


def break_saves(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.engine.workflows.runner as runner_module

    def broken(folder: Path, manifest: dict[str, Any]) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(runner_module, "save", broken)


async def test_a_failed_first_save_on_redo_leaves_no_undo_unit_and_the_manifest_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, wf = await invalid_run(tmp_path, scenes(4))
    before = (env.run_folder / "manifest.json").read_bytes()
    break_saves(monkeypatch)

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert not result.ok and "could not save the manifest" in result.line
    assert (env.run_folder / "manifest.json").read_bytes() == before
    assert await env.agent.undo(Sink()) == "/workflow run reel cats"  # the run's unit, not a redo's


async def test_a_failed_first_save_on_a_plain_resume_leaves_no_undo_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = build(tmp_path, [answer(scenes(2))])
    env.img.fail_on = {f"{FOLDER}/s2.png"}
    wf = workflow()
    await run(env, wf)
    env.img.fail_on = set()
    break_saves(monkeypatch)

    result = await env.runner.resume(lookup(wf), None, Sink(), env.approve)

    assert not result.ok and "could not save the manifest" in result.line
    assert await env.agent.undo(Sink()) == "/workflow run reel cats"


async def test_a_failed_first_save_on_run_leaves_no_undo_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = build(tmp_path, [answer(scenes(1))])
    break_saves(monkeypatch)

    result = await run(env)

    assert not result.ok and "could not save the manifest" in result.line
    assert env.agent.undo_log.peek() is None


def png(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"


def upscale_workflow() -> Workflow:
    data = {
        "description": "upscale",
        "steps": [
            {
                "id": "up",
                "kind": "tool",
                "tool": "upscale_image",
                "args": {"image": "src.png", "path": "{folder}/up.png", "scale": 2},
            }
        ],
    }
    return workflow(data)


class UpscaleImage(WriteFileTool):
    name: ClassVar[str] = "upscale_image"
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"image": {"type": "string"}, "path": {"type": "string"}, "scale": {"type": "integer"}},
        "required": ["image", "path"],
    }


async def test_an_upscale_call_is_estimated_from_its_source_file(tmp_path: Path) -> None:
    env = build(tmp_path, [], extra=[UpscaleImage()])
    (env.workspace / "src.png").write_bytes(png(512, 512))

    await run(env, upscale_workflow(), text="x")

    assert env.approved[0].estimate_s == round(1024 * 1024 * DEFAULT_RATE["upscale_image"])  # 512 -> 1024


async def test_an_upscale_call_without_its_source_assumes_a_1024_source(tmp_path: Path) -> None:
    env = build(tmp_path, [], extra=[UpscaleImage()])

    await run(env, upscale_workflow(), text="x")

    assert env.approved[0].estimate_s == round(2048 * 2048 * DEFAULT_RATE["upscale_image"])


def test_the_source_pixels_follow_upscale_size_and_the_scale(tmp_path: Path) -> None:
    env = build(tmp_path, [])
    (env.workspace / "src.png").write_bytes(png(333, 250))

    assert env.runner._source_pixels({"image": "src.png", "scale": 2}) == 660 * 496
    assert env.runner._source_pixels({"image": "src.png"}) == 660 * 496
    assert env.runner._source_pixels({"image": "missing.png", "scale": 2}) is None
    assert env.runner._source_pixels({"image": "clipboard:1"}) is None
    assert env.runner._source_pixels({}) is None


def test_the_output_pixels_come_from_the_png_header(tmp_path: Path) -> None:
    env = build(tmp_path, [])
    (env.workspace / "out.png").write_bytes(png(2000, 1500))
    (env.workspace / "bad.png").write_bytes(b"not a png")

    assert env.runner._output_pixels({"path": "out.png"}) == 3_000_000
    assert env.runner._output_pixels({"path": "bad.png"}) is None
    assert env.runner._output_pixels({"path": "missing.png"}) is None
    assert env.runner._output_pixels({"path": 3}) is None


class OddStore(TimingStore):
    def add(self, key: str, value: float) -> None:
        raise OverflowError("int too large")


async def test_a_timing_store_raising_anything_never_aborts_the_run(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(scenes(2))], timings=OddStore(None))
    sink = Sink()

    result = await run(env, sink=sink)

    assert result.ok, result.line
    warnings = [t for level, t in sink.lines if t.startswith("could not save the call timings")]
    assert warnings == ["could not save the call timings: int too large"]


async def test_a_run_with_a_huge_integer_in_the_timing_file_completes(tmp_path: Path) -> None:
    path = tmp_path / "timings" / "t.json"
    path.parent.mkdir()
    huge = "1" + "0" * 400
    path.write_text('{"timings": 1, "keys": {"speak": [' + huge + '], "generate_image": [' + huge + "]}}")
    env = build(tmp_path, [answer(scenes(2))], timings=TimingStore(path))

    result = await run(env)

    assert result.ok, result.line
    assert len(TimingStore(path).values("speak")) == 2


_ = (Sink, lookup)
