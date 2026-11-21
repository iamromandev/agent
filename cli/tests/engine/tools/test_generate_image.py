"""generate_image: validation, preview and execution against a fake mflux (AC5, AC6, AC8, AC9, AC11, AC12, AC14)."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from src.engine.events import ToolProgress
from src.engine.guardrails.setup import Guardrails
from src.engine.imagegen import MfluxGenerator, png_size
from src.engine.tools.base import Prepared, ToolContext, ToolResult
from src.engine.tools.image import GenerateImageTool

from tests.image_fakes import fake_mflux


class Journal:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def before(self, path: Path) -> None:
        self.calls.append(("before", path))

    def after(self, path: Path) -> None:
        self.calls.append(("after", path))


@pytest.fixture
def tool(tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> GenerateImageTool:
    folder = tmp_path / "bin"
    folder.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    return GenerateImageTool(MfluxGenerator(fake_mflux(folder), "x/y", 20, Guardrails.off(workspace)))


async def _prepare(tool: GenerateImageTool, ctx: ToolContext, **args: Any) -> Prepared | ToolResult:
    args.setdefault("prompt", "a paper plane")
    args.setdefault("path", "icon.png")
    return await tool.prepare(args, ctx)


async def _refused(tool: GenerateImageTool, ctx: ToolContext, **args: Any) -> ToolResult:
    result = await _prepare(tool, ctx, **args)
    assert isinstance(result, ToolResult), "a refusal must not produce a Prepared"
    assert result.is_error
    return result


async def _prepared(tool: GenerateImageTool, ctx: ToolContext, **args: Any) -> Prepared:
    result = await _prepare(tool, ctx, **args)
    assert isinstance(result, Prepared), result
    return result


def test_the_tool_asks_every_time_hides_in_plan_mode_and_can_be_interrupted(tool: GenerateImageTool) -> None:
    assert tool.name == "generate_image"
    assert tool.mutating and tool.needs_approval
    assert tool.runs_to_completion() is False
    assert tool.grant_arg == "prompt"


@pytest.mark.parametrize(
    "args, message",
    [
        ({"prompt": ""}, "prompt must be 1-2000 characters"),
        ({"prompt": "   "}, "prompt must be 1-2000 characters"),
        ({"prompt": "x" * 2001}, "prompt must be 1-2000 characters"),
        ({"width": 1000}, "width must be 256-2048 and a multiple of 16"),
        ({"width": 128}, "width must be 256-2048 and a multiple of 16"),
        ({"height": 2064}, "height must be 256-2048 and a multiple of 16"),
        ({"steps": 0}, "steps must be 1-50"),
        ({"steps": 51}, "steps must be 1-50"),
        ({"seed": -1}, "seed must be 0-4294967295"),
        ({"seed": 2**32}, "seed must be 0-4294967295"),
    ],
)
async def test_bad_arguments_are_errors_without_a_prepared(
    tool: GenerateImageTool, ctx: ToolContext, args: dict[str, Any], message: str
) -> None:
    result = await _refused(tool, ctx, **args)
    assert result.output == message
    assert result.summary == "invalid"


async def test_a_valid_call_passes_validation_at_the_limits(tool: GenerateImageTool, ctx: ToolContext) -> None:
    await _prepared(tool, ctx, prompt="x" * 2000, width=256, height=2048, steps=50, seed=0)


async def test_a_wrong_argument_type_is_an_error(tool: GenerateImageTool, ctx: ToolContext) -> None:
    result = await _refused(tool, ctx, width="big")
    assert "width" in result.output


@pytest.mark.parametrize(
    "path, message",
    [
        ("../x.png", "refused: outside workspace"),
        ("/etc/x.png", "refused: outside workspace"),
        ("a.jpg", "a.jpg must end in .png"),
        ("noext", "noext must end in .png"),
    ],
)
async def test_path_checks(tool: GenerateImageTool, ctx: ToolContext, path: str, message: str) -> None:
    assert (await _refused(tool, ctx, path=path)).output == message


async def test_a_protected_path_is_refused(tool: GenerateImageTool, workspace: Path) -> None:
    guard = Guardrails.off(workspace)
    ctx = ToolContext(workspace, 5.0, guard=dataclasses.replace(guard, paths=_protect(guard, "secret.png")))
    assert (await _refused(tool, ctx, path="secret.png")).output == "protected path"


def _protect(guard: Guardrails, name: str) -> Any:
    class Paths:
        def is_protected(self, path: Path) -> bool:
            return path.name == name

    return Paths()


async def test_a_directory_is_refused(tool: GenerateImageTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "d.png").mkdir()
    assert (await _refused(tool, ctx, path="d.png")).output == "d.png is a directory"


async def test_the_extension_is_case_insensitive(tool: GenerateImageTool, ctx: ToolContext) -> None:
    await _prepared(tool, ctx, path="ICON.PNG")


async def test_the_preview_lists_prompt_size_model_and_the_download_warning(tool: GenerateImageTool, ctx: ToolContext) -> None:
    run = await _prepared(tool, ctx, width=512, height=256, steps=5, seed=42, prompt="a\x1b[2Jb")
    assert run.target is None
    assert run.preview is not None and run.preview.kind == "content"
    assert run.preview.title == "generate image icon.png"
    body = run.preview.body
    assert "\x1b" not in body and "a\\x1b[2Jb" in body.replace("\\u001b", "\\x1b")
    assert "size: 512\u00d7256, 5 steps, seed 42" in body
    assert "model: x/y" in body
    assert "model x/y is not downloaded; this call may download several GB" in body
    assert "overwrites" not in body


async def test_no_download_line_when_the_model_is_cached(tool: GenerateImageTool, ctx: ToolContext, tmp_path: Path) -> None:
    snap = tmp_path / "hub" / "models--x--y" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "w").write_text("x")
    run = await _prepared(tool, ctx)
    assert run.preview is not None and "not downloaded" not in run.preview.body


async def test_the_preview_says_when_a_file_will_be_overwritten(tool: GenerateImageTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "icon.png").write_bytes(b"old")
    run = await _prepared(tool, ctx)
    assert run.preview is not None and "[overwrites the existing file]" in run.preview.body


async def test_a_random_seed_is_chosen_once_and_shown(tool: GenerateImageTool, ctx: ToolContext) -> None:
    run = await _prepared(tool, ctx)
    assert run.preview is not None
    seed = int(run.preview.body.split("seed ")[1].split()[0])
    assert 0 <= seed < 2**32
    result = await run.execute()
    assert f"seed {seed})" in result.output


async def test_success_writes_the_png_and_reports_it(tool: GenerateImageTool, ctx: ToolContext, workspace: Path) -> None:
    run = await _prepared(tool, ctx, path="assets/icon.png", width=512, height=512, seed=1234567)
    result = await run.execute()
    path = workspace / "assets" / "icon.png"
    assert not result.is_error, result
    assert png_size(path.read_bytes()) == (512, 512)
    size = path.stat().st_size
    kb = f"{max(round(size / 1024), 1)} KB"
    assert result.output == f"generated assets/icon.png (PNG, 512\u00d7512, {kb}, seed 1234567)"
    assert result.summary == f"PNG, 512\u00d7512, {kb}"
    assert sorted(p.name for p in path.parent.iterdir()) == ["icon.png"]  # no temp file left


async def test_success_replaces_an_existing_file_and_tells_the_journal(
    tool: GenerateImageTool, ctx: ToolContext, workspace: Path
) -> None:
    journal = Journal()
    jctx = dataclasses.replace(ctx, journal=journal)
    (workspace / "icon.png").write_bytes(b"old")
    run = await _prepared(tool, jctx, width=256, height=256)
    await run.execute()
    target = workspace / "icon.png"
    assert png_size(target.read_bytes()) == (256, 256)
    assert journal.calls == [("before", target), ("after", target)]


@pytest.mark.parametrize(
    "mode, message",
    [
        ("nothing", "mflux wrote no image"),
        ("jpeg", "mflux wrote a JPEG instead of a PNG"),
        ("truncated", "mflux wrote an incomplete PNG"),
        ("text", "mflux wrote an unknown file instead of a PNG"),
        ("wrongsize", "mflux wrote 256\u00d7256, expected 512\u00d7512"),
        ("fail", "mflux failed (exit 1): "),
    ],
)
async def test_failures_leave_the_old_file_alone(
    tool: GenerateImageTool, ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch, mode: str, message: str
) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", mode)
    (workspace / "icon.png").write_bytes(b"old")
    run = await _prepared(tool, ctx, width=512, height=512)
    result = await run.execute()
    assert result.is_error and result.output.startswith(message)
    assert result.summary == "failed"
    assert (workspace / "icon.png").read_bytes() == b"old"
    assert [p.name for p in workspace.iterdir()] == ["icon.png"]


async def test_a_failure_with_no_existing_file_leaves_nothing(
    tool: GenerateImageTool, ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "jpeg")
    result = await (await _prepared(tool, ctx, path="new/a.png")).execute()
    assert result.is_error
    assert not (workspace / "new" / "a.png").exists()


async def test_a_timeout_is_a_timed_out_error_and_an_old_file_is_unchanged(
    tmp_path: Path, workspace: Path, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "slow"
    folder.mkdir()
    monkeypatch.setenv("FAKE_MFLUX_MODE", "sleep")
    monkeypatch.setenv("FAKE_MFLUX_PIDS", str(tmp_path / "pids"))
    slow = GenerateImageTool(MfluxGenerator(fake_mflux(folder), "x/y", 1, Guardrails.off(workspace)))
    (workspace / "icon.png").write_bytes(b"old")
    result = await (await _prepared(slow, ctx)).execute()
    assert result.output == "timed out after 1 s"
    assert result.summary == "timed out"
    assert (workspace / "icon.png").read_bytes() == b"old"
    assert [p.name for p in workspace.iterdir()] == ["icon.png"]


async def test_a_cancel_propagates_and_leaves_no_file(
    tool: GenerateImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pids = tmp_path / "pids"
    monkeypatch.setenv("FAKE_MFLUX_MODE", "sleep")
    monkeypatch.setenv("FAKE_MFLUX_PIDS", str(pids))
    run = await _prepared(tool, ctx)
    task = asyncio.ensure_future(run.execute())
    for _ in range(250):
        if pids.exists() and len(pids.read_text().split()) == 2:
            break
        await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert list(workspace.iterdir()) == []


async def test_the_temp_folder_is_removed(tool: GenerateImageTool, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setattr("tempfile.tempdir", None)
    await (await _prepared(tool, ctx)).execute()
    assert list(temp.iterdir()) == []


async def test_a_folder_swapped_for_a_symlink_during_the_run_is_refused(
    tool: GenerateImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    run = await _prepared(tool, ctx, path="sub/icon.png")
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "sub").symlink_to(outside)
    result = await run.execute()
    assert result.is_error and result.output == "refused: outside workspace"
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
async def test_an_unwritable_folder_is_a_cannot_write_error(tool: GenerateImageTool, ctx: ToolContext, workspace: Path) -> None:
    folder = workspace / "ro"
    folder.mkdir()
    run = await _prepared(tool, ctx, path="ro/icon.png")
    folder.chmod(0o500)
    try:
        result = await run.execute()
    finally:
        folder.chmod(0o700)
    assert result.is_error and result.output.startswith("Cannot write ro/icon.png: ")
    assert result.summary == "write failed"


async def test_a_cancel_during_placement_never_leaves_a_file_without_a_journal_entry(
    tool: GenerateImageTool, ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    real_replace = os.replace
    started = threading.Event()

    def slow_replace(src: Any, dst: Any) -> None:
        started.set()
        time.sleep(0.3)
        real_replace(src, dst)

    monkeypatch.setattr("src.engine.tools.image.os.replace", slow_replace)
    journal = Journal()
    run = await _prepared(tool, dataclasses.replace(ctx, journal=journal))
    task = asyncio.ensure_future(run.execute())
    for _ in range(500):
        if started.is_set():
            break
        await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    target = workspace / "icon.png"
    assert target.exists() == (("after", target) in journal.calls)
    assert target.exists()  # placement that started is finished, then the cancel is re-raised
    assert [p.name for p in workspace.iterdir()] == ["icon.png"]


async def test_an_output_over_the_size_limit_is_refused(
    tool: GenerateImageTool, ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.engine.tools.image.MAX_OUTPUT_BYTES", 10)
    result = await (await _prepared(tool, ctx)).execute()
    assert result.is_error and result.output == "mflux wrote an image over 64 MB"
    assert list(workspace.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root")
async def test_an_overwrite_keeps_the_permission_bits(tool: GenerateImageTool, ctx: ToolContext, workspace: Path) -> None:
    target = workspace / "icon.png"
    target.write_bytes(b"old")
    target.chmod(0o640)
    await (await _prepared(tool, ctx)).execute()
    assert target.stat().st_mode & 0o7777 == 0o640
    assert png_size(target.read_bytes()) == (1024, 1024)


def _collect(ctx: ToolContext, sink: Any) -> ToolContext:
    return dataclasses.replace(ctx, progress=sink)


async def test_progress_lines_follow_the_phases(
    tool: GenerateImageTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.engine.tools.image.PROGRESS_TICK", 0.02)
    monkeypatch.setenv("FAKE_MFLUX_MODE", "progress")
    monkeypatch.setenv("FAKE_MFLUX_DELAY", "0.1")
    events: list[Any] = []
    run = await _prepared(tool, _collect(ctx, events.append), width=256, height=256, steps=4)
    result = await run.execute()
    assert not result.is_error, result
    texts = [e.text for e in events]
    assert all(isinstance(e, ToolProgress) for e in events)
    assert texts[0].startswith("starting")
    assert any(t.startswith("downloading model") for t in texts)
    steps = [int(t.split()[1].split("/")[0]) for t in texts if t.startswith("step ")]
    assert steps and steps == sorted(steps) and steps[-1] >= 3  # the last step and saving arrive together
    assert any(t.startswith("step 2/4 ") and "left" in t for t in texts)
    assert texts[-1].startswith("saving")
    assert all(len(t) <= 80 and t.isprintable() for t in texts)


async def test_a_flood_reaches_the_sink_at_most_five_times_a_second(
    tool: GenerateImageTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "flood")
    stamps: list[float] = []
    sink = lambda e: stamps.append(time.monotonic())  # noqa: E731
    run = await _prepared(tool, _collect(ctx, sink), width=256, height=256)
    started = time.monotonic()
    result = await run.execute()
    assert not result.is_error, result
    seconds = max(time.monotonic() - started, 1.0)
    assert len(stamps) <= 5 * seconds + 1


async def test_no_sink_and_a_raising_sink_give_the_same_result(
    tool: GenerateImageTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "progress")

    def boom(event: Any) -> None:
        raise RuntimeError("no")

    plain = await (await _prepared(tool, ctx, seed=5, width=256, height=256)).execute()
    raising = await (await _prepared(tool, _collect(ctx, boom), seed=5, width=256, height=256)).execute()
    assert not plain.is_error and plain.output == raising.output and plain.summary == raising.summary


async def test_no_event_arrives_after_the_result(
    tool: GenerateImageTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.engine.tools.image.PROGRESS_TICK", 0.01)
    monkeypatch.setenv("FAKE_MFLUX_MODE", "progress")
    events: list[Any] = []
    await (await _prepared(tool, _collect(ctx, events.append), width=256, height=256)).execute()
    count = len(events)
    await asyncio.sleep(0.1)
    assert len(events) == count


async def test_a_failure_after_progress_keeps_the_error_text(
    tool: GenerateImageTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "progressfail")
    result = await (await _prepared(tool, _collect(ctx, lambda e: None), width=256, height=256)).execute()
    assert result.is_error and result.output.startswith("mflux failed (exit 1): ")
    assert "step " not in result.output and "%|" not in result.output


def test_the_label_names_the_file_not_the_prompt() -> None:
    tool = GenerateImageTool(MfluxGenerator(Path("/nonexistent/mflux"), "x/y", 60, Guardrails.off(Path.cwd())))
    assert tool.label({"prompt": "a very long prompt " * 50, "path": "images/fox.png"}) == "generate_image images/fox.png"
    assert tool.label({"prompt": "p"}) == "generate_image"
