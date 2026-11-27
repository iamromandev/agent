"""/workflow in the app: list, run with the plan dialog, refusals, Esc, resume, cancel."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from core.engine.commands.builtins import BUILTINS
from core.engine.workflows.loader import Workflows
from core.engine.workflows.runs import RunIndex
from core.testing.fakes import Gate
from core.testing.workflow_data import answer, reel_data, scenes, workflow
from core.testing.workflow_fakes import AssembleVideo, GenerateImage, SlowTool, Speak
from src.tui.app import HELP
from src.tui.screens.approval import ApprovalScreen
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import NoticeEntry
from textual.widgets import OptionList

from tests.tui.conftest import AppFactory, choose, type_and_send, until

RUN_GLOB = "workflows/reel/*"


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _make(make_app: AppFactory, tmp_path: Path, script, *, extra=None, data=None, **kwargs):
    tools = extra if extra is not None else [GenerateImage(), Speak(), AssembleVideo()]
    wf = workflow(data or reel_data())
    app = make_app(
        script,
        tools=False,
        workflows=Workflows({"reel": wf}, []),
        run_index=RunIndex(tmp_path / "data" / "runs.json"),
        extra_tools=tools,
        **kwargs,
    )
    return app, tools


def _files(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in tmp_path.glob(f"{RUN_GLOB}/*"))


def test_workflow_is_a_builtin_with_a_help_entry_and_is_reserved() -> None:
    from evals.tasks import RESERVED

    assert "workflow" in BUILTINS
    assert "workflow" in RESERVED
    assert "/workflow" in HELP


async def test_workflow_lists_the_loaded_workflows(make_app: AppFactory, tmp_path: Path) -> None:
    app, _ = _make(make_app, tmp_path, [])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow")
        await until(pilot, lambda: any("reel" in n for n in _notices(app)))
        assert any("make a reel" in n for n in _notices(app))


async def test_a_run_shows_the_plan_and_runs_the_calls_when_allowed(make_app: AppFactory, tmp_path: Path) -> None:
    app, _ = _make(make_app, tmp_path, [answer(scenes(2))])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow run reel cats")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        screen = app.screen
        assert isinstance(screen, ApprovalScreen)
        ids = [o.id for o in screen.query_one(OptionList).options]
        assert "always" not in ids and "project" not in ids
        assert screen.request.preview is not None and screen.request.preview.kind == "content"
        await choose(pilot, "once")
        await until(pilot, lambda: any("finished" in n for n in _notices(app)))
        assert _files(tmp_path) == ["manifest.json", "reel.mp4", "s1.png", "s2.png", "v1.wav", "v2.wav"]
        assert app.query_one(StatusBar).workflow == ""
        assert not app._turn_active and not app._workflow_active


async def test_a_denied_plan_makes_no_files(make_app: AppFactory, tmp_path: Path) -> None:
    app, _ = _make(make_app, tmp_path, [answer(scenes(2))])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow run reel cats")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await choose(pilot, "deny")
        await pilot.press("enter")
        await until(pilot, lambda: any("not approved" in n for n in _notices(app)))
        assert _files(tmp_path) == []
        assert not app._turn_active


async def test_a_run_needs_a_known_workflow_and_an_input(make_app: AppFactory, tmp_path: Path) -> None:
    app, _ = _make(make_app, tmp_path, [])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow run nope x")
        await until(pilot, lambda: any("no workflow named" in n for n in _notices(app)))
        await type_and_send(pilot, "/workflow bogus")
        await until(pilot, lambda: any("usage: /workflow" in n for n in _notices(app)))
        assert not app._workflow_active


async def test_plan_mode_refuses_a_run(make_app: AppFactory, tmp_path: Path) -> None:
    app, _ = _make(make_app, tmp_path, [])
    async with app.run_test() as pilot:
        await pilot.press("tab")
        await pilot.press("tab")
        assert app.agent.mode.value == "plan"
        await type_and_send(pilot, "/workflow run reel cats")
        await until(pilot, lambda: any("workflows need build or edits mode" in n for n in _notices(app)))
        assert not app._workflow_active


async def test_messages_are_refused_during_a_run_but_usage_works(make_app: AppFactory, tmp_path: Path) -> None:
    gate = Gate()
    app, _ = _make(make_app, tmp_path, [answer(scenes(1))], extra=[GenerateImage(), Speak(), AssembleVideo(), SlowTool(gate)], data=_slow(reel_data()))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow run reel cats")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await choose(pilot, "once")
        await asyncio.wait_for(gate.reached.wait(), 5)
        await type_and_send(pilot, "hello")
        await until(pilot, lambda: any("a workflow is running" in n for n in _notices(app)))
        assert app.query_one(ChatInput).text == "hello"
        assert not app._queue.entries
        app.query_one(ChatInput).load_text("")
        await type_and_send(pilot, "/usage")
        await until(pilot, lambda: type(app.screen).__name__ == "UsageScreen")
        await pilot.press("escape")
        gate.release.set()
        await until(pilot, lambda: not app._workflow_active)


def _slow(data):
    data["steps"].append({"id": "last", "kind": "tool", "tool": "slow", "args": {"path": "{folder}/slow.txt"}})
    return data


async def test_esc_stops_a_run_and_resume_finishes_it(make_app: AppFactory, tmp_path: Path) -> None:
    gate = Gate()
    app, _ = _make(make_app, tmp_path, [answer(scenes(1))], extra=[GenerateImage(), Speak(), AssembleVideo(), SlowTool(gate)], data=_slow(reel_data()))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow run reel cats")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await choose(pilot, "once")
        await asyncio.wait_for(gate.reached.wait(), 5)
        await pilot.press("escape")
        await until(pilot, lambda: not app._workflow_active)
        assert any("interrupted" in n or "stopped" in n for n in _notices(app))
        assert not app._turn_active
        gate.release.set()
        await type_and_send(pilot, "/workflow resume")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await choose(pilot, "once")
        await until(pilot, lambda: any("finished" in n for n in _notices(app)))
        assert "slow.txt" in _files(tmp_path)


async def test_workflow_cancel_stops_a_run(make_app: AppFactory, tmp_path: Path) -> None:
    gate = Gate()
    app, _ = _make(make_app, tmp_path, [answer(scenes(1))], extra=[GenerateImage(), Speak(), AssembleVideo(), SlowTool(gate)], data=_slow(reel_data()))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow cancel")
        await until(pilot, lambda: any("no workflow is running" in n for n in _notices(app)))
        await type_and_send(pilot, "/workflow run reel cats")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await choose(pilot, "once")
        await asyncio.wait_for(gate.reached.wait(), 5)
        await type_and_send(pilot, "/workflow cancel")
        await until(pilot, lambda: not app._workflow_active)
        assert not app._turn_active


async def test_resume_with_nothing_to_resume(make_app: AppFactory, tmp_path: Path) -> None:
    app, _ = _make(make_app, tmp_path, [])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow resume")
        await until(pilot, lambda: any("no unfinished workflow run" in n for n in _notices(app)))
        assert not app._workflow_active


async def test_a_failed_run_is_resumed_after_the_fix(make_app: AppFactory, tmp_path: Path) -> None:
    failing = {"on": True}

    class FlakySpeak(Speak):
        async def prepare(self, args, ctx):
            if failing["on"] and args["path"].endswith("v2.wav"):
                args = {**args, "fail": True}
            return await super().prepare(args, ctx)

    app, _ = _make(make_app, tmp_path, [answer(scenes(2))], extra=[GenerateImage(), FlakySpeak(), AssembleVideo()])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow run reel cats")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await choose(pilot, "once")
        await until(pilot, lambda: any("failed at voice 2/2" in n for n in _notices(app)))
        failing["on"] = False
        await type_and_send(pilot, "/workflow resume")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await choose(pilot, "once")
        await until(pilot, lambda: any("finished" in n for n in _notices(app)))
        assert "reel.mp4" in _files(tmp_path)


async def test_run_without_a_name_is_a_usage_error(make_app: AppFactory, tmp_path: Path) -> None:
    app, _ = _make(make_app, tmp_path, [])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow run")
        await until(pilot, lambda: any("usage: /workflow" in n for n in _notices(app)))
        assert not app._workflow_active


async def test_run_with_a_name_and_no_input_starts_with_an_empty_input(make_app: AppFactory, tmp_path: Path) -> None:
    app, _ = _make(make_app, tmp_path, [answer(scenes(1))])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow run reel")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        screen = app.screen
        assert isinstance(screen, ApprovalScreen) and screen.request.preview is not None
        assert screen.request.preview.title.endswith("-run")  # the slug of an empty input
        await choose(pilot, "once")
        await until(pilot, lambda: any("finished" in n for n in _notices(app)))


async def test_the_usage_line_names_the_input_as_optional() -> None:
    from src.tui.app import WORKFLOW_USAGE

    assert "run <name> [input]" in WORKFLOW_USAGE
    assert "resume [id] [--redo <step>]" in WORKFLOW_USAGE


@pytest.mark.parametrize("command", ["resume --redo", "resume a b", "resume --redo story --redo x", "resume --bogus"])
async def test_resume_usage_errors(make_app: AppFactory, tmp_path: Path, command: str) -> None:
    app, _ = _make(make_app, tmp_path, [])
    async with app.run_test() as pilot:
        await type_and_send(pilot, f"/workflow {command}")
        await until(pilot, lambda: any("usage: /workflow" in n for n in _notices(app)))
        assert not app._workflow_active


async def _finish_reel(pilot, app) -> None:
    await type_and_send(pilot, "/workflow run reel cats")
    await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
    await choose(pilot, "once")
    await until(pilot, lambda: any("finished" in n for n in _notices(app)))


async def test_redo_shows_the_note_the_reused_calls_and_the_estimate_source_and_undo_reverts_it(
    make_app: AppFactory, tmp_path: Path
) -> None:
    from core.testing.workflow_data import scenes_with

    app, _ = _make(make_app, tmp_path, [answer(scenes(2)), answer(scenes_with(["n1", "different"]))])
    async with app.run_test() as pilot:
        await _finish_reel(pilot, app)
        run_id = next(tmp_path.glob(RUN_GLOB)).name
        await type_and_send(pilot, f"/workflow resume {run_id} --redo story")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        preview = _plan_screen(app).request.preview
        assert preview is not None
        lines = preview.body.splitlines()
        assert lines[0] == "1 calls: 1 speak"
        assert "reused 4 calls" in lines
        assert "re-asked story: asked by the user" in lines
        assert any(line == "about 1 min (learned from 2 calls)" for line in lines)  # the first run's two speak calls
        await choose(pilot, "once")
        await until(pilot, lambda: not app._workflow_active)
        await type_and_send(pilot, "/undo")
        await until(pilot, lambda: any("Undid the last turn" in n for n in _notices(app)))
        assert (next(tmp_path.glob(RUN_GLOB)) / "v2.wav").read_text() == "n2"  # the first run's narration is back
        assert app.query_one(ChatInput).text == f"/workflow resume {run_id} --redo story"


async def test_redo_flag_may_come_before_the_id(make_app: AppFactory, tmp_path: Path) -> None:
    app, _ = _make(make_app, tmp_path, [answer(scenes(1)), answer(scenes(1))])
    async with app.run_test() as pilot:
        await _finish_reel(pilot, app)
        run_id = next(tmp_path.glob(RUN_GLOB)).name
        await type_and_send(pilot, f"/workflow resume --redo story {run_id}")
        await until(pilot, lambda: any("nothing changed" in n for n in _notices(app)))


async def test_redo_of_a_non_model_step_names_the_model_steps(make_app: AppFactory, tmp_path: Path) -> None:
    app, _ = _make(make_app, tmp_path, [answer(scenes(1))])
    async with app.run_test() as pilot:
        await _finish_reel(pilot, app)
        run_id = next(tmp_path.glob(RUN_GLOB)).name
        await type_and_send(pilot, f"/workflow resume {run_id} --redo pic")
        await until(pilot, lambda: any("no model step 'pic'" in n and "model steps: story" in n for n in _notices(app)))


async def test_workflow_list_is_the_same_as_bare_workflow(make_app: AppFactory, tmp_path: Path) -> None:
    app, _ = _make(make_app, tmp_path, [])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow list")
        await until(pilot, lambda: any("make a reel" in n for n in _notices(app)))


async def _held(pilot, app, gate: Gate) -> None:
    """Start a run whose last call waits at `gate`, and wait for it to be there."""
    await type_and_send(pilot, "/workflow run reel cats")
    await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
    await choose(pilot, "once")
    await asyncio.wait_for(gate.reached.wait(), 5)


def _held_app(make_app: AppFactory, tmp_path: Path, gate: Gate, **kwargs):
    tools = [GenerateImage(), Speak(), AssembleVideo(), SlowTool(gate)]
    return _make(make_app, tmp_path, [answer(scenes(1))], extra=tools, data=_slow(reel_data()), **kwargs)


async def test_image_pull_is_refused_during_a_run_but_status_and_cancel_work(
    make_app: AppFactory, tmp_path: Path
) -> None:
    from tests.tui.test_images_command import _puller

    gate = Gate()
    puller = _puller(tmp_path)
    app, _ = _held_app(make_app, tmp_path, gate, images=puller)
    async with app.run_test() as pilot:
        await _held(pilot, app, gate)
        await type_and_send(pilot, "/images pull")
        await until(pilot, lambda: any("workflow is running" in n and "/images" in n for n in _notices(app)))
        assert not puller.busy
        await type_and_send(pilot, "/images")
        await until(pilot, lambda: any("generate" in n and "missing" in n for n in _notices(app)))
        await type_and_send(pilot, "/images cancel")
        await until(pilot, lambda: any("no image pull is running" in n for n in _notices(app)))
        gate.release.set()
        await until(pilot, lambda: not app._workflow_active)


@pytest.mark.parametrize("command", ["/new", "/model", "/model other", "/sessions", "/compact", "/x go"])
async def test_session_commands_and_custom_commands_are_refused_during_a_run(
    make_app: AppFactory, tmp_path: Path, command: str
) -> None:
    from core.engine.commands.loader import Command, Commands

    gate = Gate()
    app, _ = _held_app(
        make_app, tmp_path, gate, commands=Commands({"x": Command("x", "Do X", None, "global")}, ())
    )
    async with app.run_test() as pilot:
        await _held(pilot, app, gate)
        before = len(app.agent.history)
        await type_and_send(pilot, command)
        await until(pilot, lambda: any("busy" in n or "workflow is running" in n for n in _notices(app)))
        assert app._workflow_active and not app._queue.entries
        assert len(app.agent.history) == before
        gate.release.set()
        await until(pilot, lambda: not app._workflow_active)


def _plan_screen(app) -> ApprovalScreen:
    screen = app.screen
    assert isinstance(screen, ApprovalScreen)
    return screen


async def test_the_plan_dialog_lists_each_calls_key_argument_cut_and_escaped(
    make_app: AppFactory, tmp_path: Path
) -> None:
    hostile = "\x1b[31mred‮evil [b]bold[/b]\x00 " + "z" * 100
    story = json.dumps({"scenes": [{"image_prompt": hostile, "narration": "n1"}]})
    app, _ = _make(make_app, tmp_path, [answer(story)])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow run reel cats")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        preview = _plan_screen(app).request.preview
        assert preview is not None
        body = preview.body
        assert "\x1b" not in body and "‮" not in body and "\x00" not in body
        assert "\\x1b[31mred" in body and "\\u202e" in body
        assert "[b]bold[/b]" in body  # markup is data: the dialog renders a Text, never markup
        line = next(ln for ln in body.splitlines() if ln.startswith("pic/1"))
        assert 0 < line.count("z") <= 60
        assert "n1" in next(ln for ln in body.splitlines() if ln.startswith("voice/1"))
        await choose(pilot, "deny")
        await pilot.press("enter")


async def test_the_plan_dialog_title_is_escaped(make_app: AppFactory, tmp_path: Path) -> None:
    app, _ = _make(make_app, tmp_path, [answer(scenes(1))])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow run reel cats")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        request = _plan_screen(app).request
        assert request.preview is not None and request.preview.title == request.label
        title = app._plan_title("a\x1b[2Jb", "f‮g")
        assert "\x1b" not in title and "‮" not in title and title.startswith("workflow ")
        await choose(pilot, "deny")
        await pilot.press("enter")


async def test_the_plan_dialog_lists_at_most_forty_calls(make_app: AppFactory, tmp_path: Path) -> None:
    data = reel_data()
    data["steps"][0]["schema"] = {**data["steps"][0]["schema"]}
    data["steps"][0]["schema"]["properties"] = {
        "scenes": {**data["steps"][0]["schema"]["properties"]["scenes"], "maxItems": 30}
    }
    app, _ = _make(make_app, tmp_path, [answer(scenes(21))], data=data)  # 21 images + 21 voices + 1 video = 43 calls
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow run reel cats")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        preview = _plan_screen(app).request.preview
        assert preview is not None
        lines = preview.body.splitlines()
        listed = [ln for ln in lines if ln.startswith(("pic/", "voice/", "video"))]
        assert len(listed) == 40
        assert lines[-1] == "and 3 more calls"
        await choose(pilot, "deny")
        await pilot.press("enter")


async def test_an_unexpected_exception_in_the_worker_is_an_error_notice(
    make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(self, *args, **kwargs):
        raise RuntimeError("boom \x1b[31m" + "x" * 300)

    monkeypatch.setattr("src.tui.app.WorkflowRunner.run", boom)
    app, _ = _make(make_app, tmp_path, [])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow run reel cats")
        await until(pilot, lambda: any("workflow failed: " in n for n in _notices(app)))
        text = next(n for n in _notices(app) if "workflow failed: " in n)
        shown = text.split("workflow failed: ", 1)[1]
        assert "\x1b" not in text and "boom \\x1b[31m" in text and len(shown) <= 200
        await until(pilot, lambda: not app._workflow_active)
        assert not app._turn_active


async def test_esc_during_a_model_step_says_nothing_was_saved(make_app: AppFactory, tmp_path: Path) -> None:
    gate = Gate()
    app, _ = _make(make_app, tmp_path, [[gate, *answer(scenes(1))]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow run reel cats")
        await asyncio.wait_for(gate.reached.wait(), 5)
        await pilot.press("escape")
        await until(pilot, lambda: not app._workflow_active)
        assert any("workflow cancelled; nothing was saved" in n for n in _notices(app))
        assert not any("resume continues" in n for n in _notices(app))


async def test_esc_during_a_call_says_resume_continues_the_run(make_app: AppFactory, tmp_path: Path) -> None:
    gate = Gate()
    app, _ = _held_app(make_app, tmp_path, gate)
    async with app.run_test() as pilot:
        await _held(pilot, app, gate)
        await pilot.press("escape")
        await until(pilot, lambda: not app._workflow_active)
        assert any("/workflow resume continues" in n for n in _notices(app))
        assert not any("nothing was saved" in n for n in _notices(app))


async def test_esc_then_resume_runs_only_the_interrupted_call(make_app: AppFactory, tmp_path: Path) -> None:
    gate = Gate()
    img, voice, video, slow = GenerateImage(), Speak(), AssembleVideo(), SlowTool(gate)
    app, _ = _make(make_app, tmp_path, [answer(scenes(1))], extra=[img, voice, video, slow], data=_slow(reel_data()))
    async with app.run_test() as pilot:
        await _held(pilot, app, gate)
        await pilot.press("escape")
        await until(pilot, lambda: not app._workflow_active)
        gate.release.set()
        await type_and_send(pilot, "/workflow resume")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await choose(pilot, "once")
        await until(pilot, lambda: any("finished" in n for n in _notices(app)))
        assert [len(t.calls) for t in (img, voice, video)] == [1, 1, 1]
        assert len(slow.calls) == 1


async def test_the_status_bar_shows_the_workflow_step_and_a_running_clock(
    make_app: AppFactory, tmp_path: Path
) -> None:
    gate = Gate()
    app, _ = _held_app(make_app, tmp_path, gate)
    async with app.run_test() as pilot:
        now = [100.0]
        status = app.query_one(StatusBar)
        status.clock = lambda: now[0]
        await _held(pilot, app, gate)
        await until(pilot, lambda: "workflow reel · last 1/1 · 0:0" in status.line)
        now[0] = 165.0
        await until(pilot, lambda: "workflow reel · last 1/1 · 1:05" in status.line)
        gate.release.set()
        await until(pilot, lambda: not app._workflow_active)
        assert "workflow" not in status.line


async def test_undo_after_a_finished_run_undoes_the_whole_run(make_app: AppFactory, tmp_path: Path) -> None:
    app, _ = _make(make_app, tmp_path, [answer(scenes(2))])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/workflow run reel cats")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await choose(pilot, "once")
        await until(pilot, lambda: any("finished" in n for n in _notices(app)))
        assert "reel.mp4" in _files(tmp_path)
        await type_and_send(pilot, "/undo")
        await until(pilot, lambda: any("Undid the last turn" in n for n in _notices(app)))
        assert [n for n in _files(tmp_path) if n != "manifest.json"] == []
        assert app.query_one(ChatInput).text == "/workflow run reel cats"
