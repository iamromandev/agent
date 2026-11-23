"""The real image-model smoke test: `make -C cli image-eval`.

Opt-in (Apple Silicon, the `image` extra, the models in the Hugging Face cache); not part of `make test`, `make eval`
or CI. The orchestrator (`main`) runs one worker process per case (`python -m evals.images --worker <case> ...`)
in a shared temp workspace: a fresh worker means its `RUSAGE_CHILDREN` peak is exactly that case's mflux/helper
processes. The workers build the real tools as the runtime does (real guardrails and sandbox, no user settings
files). The pixel checks run in `pixel_compare.py` in their own processes, so no process of the eval imports Pillow.

Cleanup: mflux and `hf` run in their own sessions, so killing the worker's group does not reach them. The worker
therefore records the group id of every process it starts in a file (`_track_groups`, a wrapper around
`asyncio.create_subprocess_exec` installed only in the worker, so `cli/src/` is untouched) and the orchestrator
SIGKILLs those groups whenever a worker ends abnormally, times out or is interrupted. The worker's `TMPDIR` is a
folder inside the eval's workspace, so the `agent-image-*` folders of a killed worker go with it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from evals.report import REPO, commit_hash, write_json

CASES = ("pull", "generate", "edit", "masked")
PROMPTS = {"generate": "a red apple on a white table", "edit": "make it blue", "masked": "make it blue"}
SIDE, STEPS = 256, 2
SEEDS = {"generate": 1, "edit": 2, "masked": 3}
REGION = [{"shape": "rect", "x": 0, "y": 0, "w": 128, "h": 256}]
FEATHER = 8
UNTOUCHED_FROM = 136  # the region's right edge (128) plus the feather
REASON_LIMIT = 200
INTERRUPT_WAIT = 10.0
CASE_TIMEOUT = 1800.0
MIN_CHANGED = 0.05  # the share of the checked area that must differ for an edit to count as one
REGION_CHANGED_TO = UNTOUCHED_FROM - 16  # the region's right edge (128) minus the feather: fully replaced columns
RSS_NOTE = "ru_maxrss of the largest child; excludes most GPU (Metal) memory"
RESULTS = REPO / ".eval"
NOT_CACHED = "model not downloaded"
NO_COMMAND = "mflux command not found"
GROUPS_ENV = "IMAGE_EVAL_PGIDS"


@dataclass(frozen=True)
class Options:
    cases: tuple[str, ...]
    download: bool
    keep: bool
    case_timeout: float = CASE_TIMEOUT


@dataclass(frozen=True)
class Tools:
    generate: str | None
    edit: str | None
    hf: str | None
    helper: str
    compare: str
    generate_model: str
    edit_model: str
    timeout: float


@dataclass
class CaseResult:
    name: str
    result: Literal["pass", "fail", "skipped"]
    reason: str | None
    seconds: float
    peak_rss_bytes: int | None


def options_from_env(env: Mapping[str, str]) -> Options | str:
    """The run's options, or the usage error text (an unknown case name)."""
    wanted = [name.strip() for name in env.get("IMAGE_EVAL_CASES", "").split(",") if name.strip()]
    unknown = [name for name in wanted if name not in CASES]
    if unknown:
        return f"unknown image-eval case(s): {', '.join(unknown)}; valid: {', '.join(CASES)}"
    cases = tuple(name for name in CASES if name in wanted) if wanted else CASES
    timeout = CASE_TIMEOUT
    if "IMAGE_EVAL_CASE_TIMEOUT" in env:
        try:
            timeout = float(env["IMAGE_EVAL_CASE_TIMEOUT"])
        except ValueError:
            timeout = 0.0
        if not 0 < timeout < float("inf"):
            return "IMAGE_EVAL_CASE_TIMEOUT must be a positive number of seconds"
    return Options(cases, env.get("IMAGE_EVAL_DOWNLOAD") == "1", env.get("IMAGE_EVAL_KEEP") == "1", timeout)


def discover(settings: Any) -> Tools | str:
    """The commands and models of this machine, or the reason the eval cannot run here."""
    from src.engine.image_pull import HF_COMMAND
    from src.engine.imagegen import EDIT_COMMAND, find_mflux
    from src.engine.tools import image as image_tools

    if sys.platform != "darwin" or platform.machine() != "arm64":
        return "needs an Apple Silicon Mac"
    generate, edit = find_mflux(), find_mflux(command=EDIT_COMMAND)
    if generate is None and edit is None:
        return "mflux commands not found (uv sync --directory cli --extra image)"
    helper = image_tools.COMPOSITE_HELPER
    if helper is None or not helper.is_file():
        return "the compositing helper is missing"
    hf = find_mflux(command=HF_COMMAND)
    return Tools(
        generate=None if generate is None else str(generate),
        edit=None if edit is None else str(edit),
        hf=None if hf is None else str(hf),
        helper=str(helper),
        compare=str(Path(__file__).resolve().parent / "pixel_compare.py"),
        generate_model=settings.image_model,
        edit_model=settings.image_edit_model,
        timeout=settings.image_timeout,
    )


def plan_cases(opts: Options, tools: Tools, cached: Callable[[str], bool]) -> list[tuple[str, str | None]]:
    """(case, skip reason or None) for each selected case, in order."""
    gen_ready, edit_ready = cached(tools.generate_model), cached(tools.edit_model)
    plan: list[tuple[str, str | None]] = []
    for name in opts.cases:
        reason: str | None = None
        if name == "pull":
            if tools.generate is None and tools.edit is None:
                reason = NO_COMMAND
            elif not opts.download and not (gen_ready and edit_ready):
                reason = NOT_CACHED
        elif name == "generate":
            if tools.generate is None:
                reason = NO_COMMAND
            elif not opts.download and not gen_ready:
                reason = NOT_CACHED
        elif tools.edit is None:
            reason = NO_COMMAND
        elif not opts.download and not edit_ready:
            reason = NOT_CACHED
        plan.append((name, reason))
    return plan


def _cut(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= REASON_LIMIT else text[: REASON_LIMIT - 1] + "…"


# worker


def _peak_rss() -> int | None:
    """Peak resident size of the largest reaped child, in bytes (macOS counts bytes, Linux kilobytes)."""
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak = peak if sys.platform == "darwin" else peak * 1024
    return peak if peak > 0 else None


async def _pixels(tools: Tools, reason: str, *args: object) -> str | None:
    """Run one `pixel_compare.py` mode: None when the check holds, `reason` when it fails (exit 1), else the error."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-I", tools.compare, *map(str, args),
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )  # fmt: skip
    try:
        _, stderr = await proc.communicate()
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        raise
    if proc.returncode == 0:
        return None
    if proc.returncode == 1:
        return reason
    tail = " ".join(stderr.decode(errors="replace").split()[-40:])
    return f"pixel compare failed: {tail or f'exit {proc.returncode}'}"


def _track_groups(path: Path) -> None:
    """Record the process group id of every session this process starts (start_new_session: pid == pgid) in `path`."""
    asyncio_module: Any = importlib.import_module("asyncio")
    original = asyncio_module.create_subprocess_exec

    async def tracked(*args: Any, **kwargs: Any) -> Any:
        proc = await original(*args, **kwargs)
        if kwargs.get("start_new_session"):
            with path.open("a") as handle:
                handle.write(f"{proc.pid}\n")
        return proc

    asyncio_module.create_subprocess_exec = tracked


async def run_case(name: str, workspace: Path, tools: Tools) -> CaseResult:
    """Run one case in this process and return its outcome (time and peak RSS filled in)."""
    started = time.monotonic()
    try:
        failure = await _run_case(name, workspace, tools)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    return CaseResult(
        name,
        "pass" if failure is None else "fail",
        None if failure is None else _cut(failure),
        time.monotonic() - started,
        _peak_rss(),
    )


async def _run_case(name: str, workspace: Path, tools: Tools) -> str | None:
    from src.engine.guardrails.setup import prepare
    from src.engine.image_pull import ImagePuller, grey_png
    from src.engine.imagegen import EDIT_EXTRA, MfluxGenerator, png_size
    from src.engine.tools import image as image_tools
    from src.engine.tools.base import Prepared, ToolContext, ToolResult
    from src.engine.tools.image import EditImageTool, GenerateImageTool

    guard = prepare(workspace, None, None, os.environ, platform.system())
    ctx = ToolContext(workspace=workspace, bash_timeout=120.0, guard=guard)
    generator = (
        None if tools.generate is None else MfluxGenerator(Path(tools.generate), tools.generate_model, tools.timeout, guard)
    )
    editor = (
        None
        if tools.edit is None
        else MfluxGenerator(Path(tools.edit), tools.edit_model, tools.timeout, guard, extra=EDIT_EXTRA)
    )

    async def call(tool: Any, args: dict[str, Any]) -> tuple[Path, ToolResult]:
        prepared = await tool.prepare(args, ctx)
        result = prepared if isinstance(prepared, ToolResult) else await prepared.execute()
        assert isinstance(prepared, (Prepared, ToolResult))
        return workspace / args["path"], result

    def check(output: Path, result: ToolResult) -> str | None:
        if result.is_error:
            return result.output
        size = png_size(output.read_bytes()) if output.is_file() else None
        return None if size == (SIDE, SIDE) else f"output is not a {SIDE}x{SIDE} PNG (got {size})"

    if name == "pull":
        # With the models cached this proves that the warm-up (dry run, then the real tiny generation) runs, not that
        # downloading works; IMAGE_EVAL_DOWNLOAD=1 with a missing model exercises the download.
        notices: list[tuple[str, str]] = []

        async def emit(level: str, text: str) -> None:
            notices.append((level, text))

        hf = Path(tools.hf) if tools.hf else None
        puller = ImagePuller({"generate": generator, "edit": editor}, hf, guard)
        await puller.pull(("generate", "edit"), emit, lambda text: None)  # this eval covers the two older models
        bad = [text for _, text in notices if "pull failed" in text or "not enough disk space" in text]
        if bad:
            return bad[0]
        ready = [text for _, text in notices if " model ready (" in text]
        return None if len(ready) == 2 else f"expected 2 ready notices, got {len(ready)}"

    if name == "generate":
        if generator is None:
            return NO_COMMAND
        args = {"prompt": PROMPTS[name], "path": "gen.png", "width": SIDE, "height": SIDE, "steps": STEPS, "seed": SEEDS[name]}
        output, result = await call(GenerateImageTool(generator), args)
        if (problem := check(output, result)) is not None:
            return problem
        return await _pixels(tools, "the generated image is uniform", "varied", output)

    if editor is None:
        return NO_COMMAND
    image_tools.COMPOSITE_HELPER = Path(tools.helper)
    source = workspace / "gen.png"
    if not (source.is_file() and png_size(source.read_bytes()) == (SIDE, SIDE)):
        source = workspace / "source.png"
        source.write_bytes(grey_png(SIDE, SIDE))
    args: dict[str, Any] = {"prompt": PROMPTS[name], "image": source.name, "path": f"{name}.png", "steps": STEPS, "seed": SEEDS[name]}
    if name == "edit":
        args |= {"width": SIDE, "height": SIDE}
        output, result = await call(EditImageTool(editor), args)
        if (problem := check(output, result)) is not None:
            return problem
        return await _pixels(tools, "the edit changed nothing", "changed", source, output, SIDE, int(SIDE * SIDE * MIN_CHANGED))
    args |= {"region": REGION, "feather": FEATHER}
    output, result = await call(EditImageTool(editor), args)
    if (problem := check(output, result)) is not None:
        return problem
    if not result.output.rstrip().endswith("masked)"):
        return f"result does not say masked: {result.output}"
    outside = await _pixels(tools, "pixels outside the region changed", "same", source, output, UNTOUCHED_FROM)
    if outside is not None:
        return outside
    minimum = int(REGION_CHANGED_TO * SIDE * MIN_CHANGED)
    return await _pixels(tools, "the edit changed nothing inside the region", "changed", source, output, REGION_CHANGED_TO, minimum)


def run_worker(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="evals.images --worker")
    parser.add_argument("--worker", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--state", required=True)
    ns = parser.parse_args(argv)
    tools = Tools(**json.loads(Path(ns.state).read_text()))
    workspace = Path(ns.workspace).resolve()
    if groups := os.environ.get(GROUPS_ENV):
        _track_groups(Path(groups))

    async def go() -> CaseResult:
        task = asyncio.ensure_future(run_case(ns.worker, workspace, tools))
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, task.cancel)
        return await task

    try:
        result = asyncio.run(go())
    except asyncio.CancelledError:
        return 130
    print(json.dumps(asdict(result)), flush=True)
    return 0


# orchestrator


def _sweep(groups: Path) -> None:
    """SIGKILL every process group the worker recorded (a dead one is ignored)."""
    try:
        lines = groups.read_text().split()
    except OSError:
        return
    for line in lines:
        if line.isdigit() and int(line) > 1:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(int(line), signal.SIGKILL)


def _spawn(name: str, workspace: Path, state: Path, timeout: float | None) -> CaseResult:
    """Run one worker. Its output goes to files, not pipes, so a process it leaves behind cannot block the read."""
    stem = workspace.parent / f"{workspace.name}.{name}"
    out_file, err_file, groups = (stem.with_suffix(ext) for ext in (".out", ".err", ".pgids"))
    tmp = workspace / ".tmp"
    tmp.mkdir(exist_ok=True)
    env = {**os.environ, "TMPDIR": str(tmp), GROUPS_ENV: str(groups)}
    cmd = [sys.executable, "-m", "evals.images", "--worker", name, "--workspace", str(workspace), "--state", str(state)]
    started = time.monotonic()
    try:
        with out_file.open("wb") as out, err_file.open("wb") as err:
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, start_new_session=True, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
            try:
                proc.wait(timeout)
            except subprocess.TimeoutExpired:
                _stop(proc)
                _sweep(groups)
                return CaseResult(name, "fail", f"timed out after {timeout:g} s", time.monotonic() - started, None)
            except BaseException:
                _stop(proc)
                _sweep(groups)
                raise
        lines = out_file.read_text(errors="replace").strip().splitlines()
        if proc.returncode == 0 and lines:
            try:
                data = json.loads(lines[-1])
                return CaseResult(name, data["result"], data["reason"], data["seconds"], data["peak_rss_bytes"])
            except (ValueError, KeyError, TypeError):
                pass
        _sweep(groups)
        tail = " ".join((err_file.read_text(errors="replace").strip().splitlines() or [""])[-3:])
        return CaseResult(name, "fail", _cut(f"worker exited {proc.returncode}: {tail}"), time.monotonic() - started, None)
    finally:
        for file in (out_file, err_file, groups):
            file.unlink(missing_ok=True)


def _stop(proc: subprocess.Popen[bytes]) -> None:
    """Interrupt the worker's group (the worker then kills mflux's own group), then SIGKILL it after a wait."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(INTERRUPT_WAIT)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _time(seconds: float) -> str:
    whole = round(seconds)
    return f"{whole // 60}m{whole % 60:02d}s" if whole >= 120 else f"{whole}s"


def _rss(value: int | None) -> str:
    return "-" if value is None else f"{value / 1e9:.1f} GB"


def _result_text(case: CaseResult) -> str:
    if case.result == "pass":
        return "pass"
    return f"fail: {case.reason}" if case.result == "fail" else f"skipped ({case.reason})"


def table(cases: list[CaseResult]) -> str:
    rows = [("case", "result", "time", "peak RSS (excl. GPU)")]
    rows += [(c.name, _result_text(c), _time(c.seconds), _rss(c.peak_rss_bytes)) for c in cases]
    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row[:3], widths, strict=True)) + "  " + row[3] for row in rows
    )


def _mflux_version() -> str | None:
    try:
        return importlib.metadata.version("mflux")
    except importlib.metadata.PackageNotFoundError:
        return None


def to_json(cases: list[CaseResult], tools: Tools, started: str) -> dict[str, Any]:
    return {
        "started": started,
        "machine": f"{platform.system()} {platform.machine()}",
        "mflux": _mflux_version(),
        "commit": commit_hash(),
        "models": {"generate": tools.generate_model, "edit": tools.edit_model},
        "rss_note": RSS_NOTE,
        "cases": [
            {
                "name": c.name,
                "result": c.result,
                "reason": c.reason,
                "seconds": round(c.seconds, 2),
                "peak_rss_bytes": c.peak_rss_bytes,
            }
            for c in cases
        ],
    }


def main(
    argv: list[str],
    env: Mapping[str, str] = os.environ,
    settings_factory: Callable[[], Any] | None = None,
    discover: Callable[[Any], Tools | str] = discover,
    results_dir: Path = RESULTS,
) -> int:
    if argv[:1] == ["--worker"]:
        return run_worker(argv)
    if argv:
        print("usage: python -m evals.images (options come from IMAGE_EVAL_* variables)", file=sys.stderr)
        return 2
    opts = options_from_env(env)
    if isinstance(opts, str):
        print(opts, file=sys.stderr)
        return 2
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        if settings_factory is None:
            from src.config.settings import Settings

            settings_factory = Settings
        found = discover(settings_factory())
        if isinstance(found, str):
            print(f"image-eval skipped: {found}")
            return 0
        tools = found
        from src.engine.imagegen import model_cached

        plan = plan_cases(opts, tools, lambda model: model_cached(model, env))
    except KeyboardInterrupt:
        print("image-eval: interrupted", file=sys.stderr)
        return 130
    started = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    workspace = Path(tempfile.mkdtemp(prefix="agent-image-eval-")).resolve()
    state = workspace.parent / f"{workspace.name}.state.json"
    state.write_text(json.dumps(asdict(tools)))
    results: list[CaseResult] = []
    interrupted = False
    current = ""
    try:
        for name, skip in plan:
            if skip is not None:
                results.append(CaseResult(name, "skipped", skip, 0.0, None))
                continue
            current = name
            print(f"image-eval: running {name}...", flush=True)
            limit = None if name == "pull" and opts.download else opts.case_timeout
            results.append(_spawn(name, workspace, state, limit))
    except KeyboardInterrupt:
        print("image-eval: interrupted", file=sys.stderr)
        results.append(CaseResult(current, "fail", "interrupted", 0.0, None))
        interrupted = True
    finally:
        state.unlink(missing_ok=True)
        if opts.keep:
            print(f"workspace kept: {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)
    path = results_dir / f"images-{started}.json"
    write_json(path, to_json(results, tools, started))
    print(table(results))
    print(f"results: {path}")
    if interrupted:
        return 130
    return 1 if any(c.result == "fail" for c in results) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
