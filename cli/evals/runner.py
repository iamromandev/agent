"""Run the eval tasks headlessly: isolation, one run, all runs, and the command-line entry."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from src.agent import Agent
from src.agent.commands.loader import load_commands
from src.agent.mcp.trust import TrustStore, trust_store_path
from src.agent.permissions import setup as permission_setup
from src.agent.permissions.config import global_settings_path
from src.agent.provider.base import Provider
from src.agent.provider.ollama import OllamaProvider
from src.agent.session import NullRecorder
from src.agent.tools import default_tools, read_only_tools
from src.agent.tools.base import ToolContext
from src.client import OllamaClient, OllamaError
from src.config.settings import Settings

from evals.approver import AutoApprover
from evals.record import RunRecord, sink
from evals.report import REPO, Meta, commit_hash, summarize, table, to_json, write_json
from evals.tasks import RESERVED, TASKS, Context, Task

FIXTURE = Path(__file__).parent / "fixtures" / "sample"
_COPY_IGNORE = shutil.ignore_patterns("__pycache__", ".pytest_cache")


@dataclass(frozen=True)
class Options:
    runs: int = 3
    tasks: tuple[str, ...] = ()  # empty: every task
    timeout: float = 300.0
    keep: bool = False


def options_from_env(env: Mapping[str, str]) -> Options:
    """EVAL_RUNS, EVAL_TASKS, EVAL_TIMEOUT, EVAL_KEEP. A bad value raises ValueError."""
    try:
        runs = int(env.get("EVAL_RUNS", "3"))
    except ValueError:
        raise ValueError(f"EVAL_RUNS must be a whole number, got {env['EVAL_RUNS']!r}") from None
    if runs < 1:
        raise ValueError(f"EVAL_RUNS must be at least 1, got {runs}")
    try:
        timeout = float(env.get("EVAL_TIMEOUT", "300"))
    except ValueError:
        raise ValueError(f"EVAL_TIMEOUT must be a number of seconds, got {env['EVAL_TIMEOUT']!r}") from None
    if not timeout > 0:
        raise ValueError(f"EVAL_TIMEOUT must be above 0, got {timeout}")
    tasks = tuple(name.strip() for name in env.get("EVAL_TASKS", "").split(",") if name.strip())
    keep = env.get("EVAL_KEEP", "").strip().lower() in ("1", "true", "yes")
    return Options(runs, tasks, timeout, keep)


class _Models(Protocol):
    async def models(self) -> list[str]: ...


async def reachable(client: _Models, model: str, url: str) -> str | None:
    """None when Ollama answers and has the model, else the line to print instead of running."""
    try:
        names = await client.models()
    except OllamaError:
        return f"eval skipped: Ollama not reachable at {url}"
    if model not in names and f"{model}:latest" not in names:
        return f"eval skipped: model {model} is not available in Ollama at {url}"
    return None


def _prepare(task: Task, tmp: Path, record: RunRecord) -> tuple[Path, dict[str, str]]:
    """The workspace (a fresh git repo with a copy of the fixture) and the environment every path helper reads."""
    workspace = tmp / "ws"
    shutil.copytree(FIXTURE, workspace, ignore=_COPY_IGNORE)
    home = tmp / "home"
    home.mkdir()
    env = {"HOME": str(home), "XDG_CONFIG_HOME": str(tmp / "config"), "XDG_DATA_HOME": str(tmp / "data")}
    try:  # a git root keeps the instructions loader inside
        subprocess.run(
            ["git", "init", "-q"],
            cwd=workspace,
            env={**env, "PATH": os.environ.get("PATH", ""), "GIT_CONFIG_NOSYSTEM": "1"},
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        note = f"git init failed ({type(exc).__name__}: {exc}); instructions may be read from outside the workspace"
        print(f"eval: {note}", file=sys.stderr)
        record.notes.append(note)
    if task.setup is not None:
        task.setup(workspace, tmp / "config")
    return workspace.resolve(), env


def _build_agent(task: Task, provider: Provider, settings: Settings, workspace: Path, env: Mapping[str, str], record: RunRecord) -> Agent:
    ctx = ToolContext(workspace, settings.bash_timeout)
    tools = default_tools(terminals=None, fetcher=None)
    permissions = permission_setup.prepare(
        workspace,
        global_settings_path(env),
        TrustStore(trust_store_path(env, name="permission-trust.json")),
        permission_setup.tool_keys(tools),
    )
    return Agent(
        provider,
        tools,
        AutoApprover(task.allow, record),
        ctx,
        task.max_steps,
        recorder=NullRecorder(),
        compact_at=settings.compact_at,
        compact_keep_turns=settings.compact_keep_turns,
        instructions_share=settings.instructions_share,
        task_tools=read_only_tools(None),
        task_max_steps=settings.task_max_steps,
        rules=permissions.rules,
    )


def _prompt(task: Task, workspace: Path, env: Mapping[str, str]) -> str:
    if not task.prompt.startswith("/"):
        return task.prompt
    name, _, rest = task.prompt[1:].partition(" ")
    command = load_commands(workspace, RESERVED, env).get(name)
    if command is None:
        raise ValueError(f"custom command /{name} not found")
    return command.expand(rest)


def _fill_from_history(record: RunRecord, agent: Agent) -> None:
    outputs = {m.tool_call_id: m.content for m in agent.history if m.role == "tool"}
    for message in agent.history:
        if message.role == "user" and not record.first_prompt:
            record.first_prompt = message.content
        for call in message.tool_calls:
            record.calls.append((call.name, call.arguments, outputs.get(call.id, "")))
    if record.tokens is None:  # no usage counts: a step is a model reply
        record.steps = sum(1 for m in agent.history if m.role == "assistant")


async def run_task(
    task: Task, run: int, provider: Provider, settings: Settings, *, timeout: float, keep: bool, record: RunRecord | None = None
) -> RunRecord:
    """One run. Pass `record` to keep the partial result when the run is interrupted (Ctrl+C)."""
    record = record if record is not None else RunRecord(task.name, run)
    tmp = Path(tempfile.mkdtemp(prefix=f"agent-eval-{task.name}-"))
    started = time.monotonic()
    try:
        if task.needs_python and shutil.which("python") is None:
            raise RuntimeError("harness: there is no `python` on PATH, but the task's prompt runs `python -m pytest`")
        workspace, env = _prepare(task, tmp, record)
        agent = _build_agent(task, provider, settings, workspace, env, record)
        try:
            prompt = _prompt(task, workspace, env)
            limit = asyncio.timeout(timeout)
            try:
                async with limit:
                    await agent.run_turn(prompt, sink(record))
            except TimeoutError:
                if not limit.expired():  # raised inside the turn, not by our deadline
                    raise
                record.error = "timeout"
                record.reasons.append("timeout")
        except Exception as exc:  # NFR2: one run's failure is its record, not the end of the eval
            record.error = f"{type(exc).__name__}: {exc}"
        finally:
            record.seconds = time.monotonic() - started
            _fill_from_history(record, agent)
        _run_checks(task, workspace, record)
    except (asyncio.CancelledError, KeyboardInterrupt):
        record.error = "interrupted"
        raise
    except Exception as exc:  # setup failed
        message = str(exc)
        record.error = record.error or (message if message.startswith("harness:") else f"{type(exc).__name__}: {exc}")
        record.seconds = record.seconds or time.monotonic() - started
    finally:
        record.passed = record.error is None and not record.reasons
        if keep:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    return record


def _run_checks(task: Task, workspace: Path, record: RunRecord) -> None:
    ctx = Context(workspace, record, record.final_text)
    for check in task.checks:
        try:
            reason = check(ctx)
        except Exception as exc:
            reason = f"a check crashed: {type(exc).__name__}: {exc}"
        if reason is not None:
            record.reasons.append(reason)


async def run_all(
    options: Options,
    provider_factory: Callable[[], Provider],
    settings: Settings,
    records: list[RunRecord] | None = None,
) -> list[RunRecord]:
    """Every selected task, `options.runs` times each, in order. Results go into `records` as they finish."""
    out = records if records is not None else []
    names = options.tasks or tuple(TASKS)
    for name in names:
        for run in range(1, options.runs + 1):
            provider = provider_factory()
            record = RunRecord(name, run)
            out.append(record)  # in the list while it runs, so an interrupted run is reported
            await run_task(TASKS[name], run, provider, settings, timeout=options.timeout, keep=options.keep, record=record)
    return out


class _Client(_Models, Protocol):
    async def aclose(self) -> None: ...


def main(
    argv: list[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    client: _Client | None = None,
    provider_factory: Callable[[], Provider] | None = None,
    report_dir: Path | None = None,
) -> int:
    """Run the eval. 0 when it ran (whatever the results) or was skipped, 2 for bad input, 130 after Ctrl+C.

    The keyword arguments are for tests; `argv` is unused (everything comes from the environment)."""
    env = os.environ if env is None else env
    try:
        options = options_from_env(env)
        unknown = [name for name in options.tasks if name not in TASKS]
        if unknown:
            raise ValueError(f"unknown task {', '.join(unknown)}; known: {', '.join(TASKS)}")
    except ValueError as exc:
        print(f"eval: {exc}", file=sys.stderr)
        return 2
    settings = Settings()
    ollama = client if client is not None else OllamaClient(settings.ollama_url)
    factory = provider_factory or (lambda: OllamaProvider(ollama, settings.model, settings.num_ctx))  # ty: ignore[invalid-argument-type]
    started = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    records: list[RunRecord] = []
    interrupted = False

    async def go() -> bool:
        try:
            skip = await reachable(ollama, settings.model, settings.ollama_url)
            if skip is not None:
                print(skip)
                return False
            await run_all(options, factory, settings, records)
            return True
        finally:
            await ollama.aclose()

    try:
        ran = asyncio.run(go())
    except KeyboardInterrupt:
        ran, interrupted = True, True
    if not ran:
        return 0
    meta = Meta(settings.model, settings.num_ctx, options.runs, options.tasks or tuple(TASKS), options.timeout, commit_hash(), started)
    print(table(summarize(records), f"model {meta.model} · ctx {meta.num_ctx} · {meta.runs} runs/task"))
    path = (report_dir or REPO / ".eval") / f"{started}.json"
    write_json(path, to_json(records, meta))
    print(f"report: {path}")
    return 130 if interrupted else 0
