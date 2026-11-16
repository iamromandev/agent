"""The eval tasks as data, and the checks that decide whether a run passed."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.engine.approval import ApprovalRequest

from evals.record import RunRecord

# The names of the TUI's built-in slash commands, which a custom command may not take. A copy: importing
# `src.tui.app` would pull in Textual. A test keeps it equal to `BUILTINS` there.
RESERVED = frozenset({"clear", "compact", "guardrails", "help", "hooks", "init", "mcp", "new", "permissions", "quit", "sessions", "skills", "undo"})
TEST_COMMAND = "python -m pytest"
PYTEST_TIMEOUT = 60.0
_SKIP_DIRS = frozenset({".git", "__pycache__", ".pytest_cache"})


@dataclass(frozen=True)
class Context:
    """What a check sees."""

    workspace: Path
    record: RunRecord
    final_text: str  # the last assistant text of the turn


Check = Callable[[Context], str | None]  # None: ok, else the reason it failed


@dataclass(frozen=True)
class Task:
    name: str
    prompt: str  # text, or "/review app/calc.py" for a custom command
    allow: Callable[[ApprovalRequest], bool]  # approver policy
    checks: tuple[Check, ...]
    max_steps: int = 25
    needs_python: bool = False  # the prompt runs `python -m pytest`
    setup: Callable[[Path, Path], None] | None = None  # (workspace, config_home): write rule or command files


def _bash_commands(record: RunRecord) -> list[tuple[str, str]]:
    """(command, output) of every bash call, from the history."""
    return [
        (str(args.get("command", "")), output) for name, args, output in record.calls if name == "bash"
    ]


def _in(word: str, text: str, whole: bool) -> bool:
    if whole:
        return re.search(rf"(?<!\w){re.escape(word.lower())}(?!\w)", text) is not None
    return word.lower() in text


def answer_mentions(*words: str, whole: tuple[str, ...] = ()) -> Check:
    """The answer has every word (a substring, any case); the words in `whole` only as whole words."""

    def check(ctx: Context) -> str | None:
        text = ctx.final_text.lower()
        missing = [word for word in words if not _in(word, text, False)]
        missing += [word for word in whole if not _in(word, text, True)]
        return f"the answer does not mention {', '.join(missing)}" if missing else None

    return check


def answer_mentions_any(what: str, *words: str) -> Check:
    def check(ctx: Context) -> str | None:
        text = ctx.final_text.lower()
        return None if any(word.lower() in text for word in words) else f"the answer does not identify {what}"

    return check


def file_has(relative: str, text: str) -> Check:
    def check(ctx: Context) -> str | None:
        try:
            content = (ctx.workspace / relative).read_text(errors="replace")
        except OSError:
            return f"{relative} is missing"
        return None if text in content else f"{relative} does not contain {text!r}"

    return check


def called(tool_name: str, at_least: int = 1) -> Check:
    def check(ctx: Context) -> str | None:
        count = ctx.record.tools[tool_name]
        return None if count >= at_least else f"{tool_name} was called {count} times, expected at least {at_least}"

    return check


def no_text_in_files(pattern: str) -> Check:
    def check(ctx: Context) -> str | None:
        found: list[str] = []
        for folder, dirs, files in os.walk(ctx.workspace):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for name in files:
                path = Path(folder) / name
                try:
                    if pattern in path.read_text(errors="replace"):
                        found.append(str(path.relative_to(ctx.workspace)))
                except OSError:
                    continue
        return f"{pattern} is still in {', '.join(sorted(found))}" if found else None

    return check


def fixture_tests_pass() -> Check:
    def check(ctx: Context) -> str | None:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            done = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                cwd=ctx.workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=PYTEST_TIMEOUT,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"the fixture's tests could not run: {exc}"
        if done.returncode == 0:
            return None
        tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-5:])
        return f"the fixture's tests fail: {tail}"

    return check


def no_approval_for(prefix: str) -> Check:
    def check(ctx: Context) -> str | None:
        asked = [a["label"] for a in ctx.record.approvals if a["tool"] == "bash" and a["label"].startswith(f"bash {prefix}")]
        return f"approval was requested for {asked[0]}" if asked else None

    return check


def _runs(ctx: Context, prefix: str) -> list[tuple[str, str | None]]:
    """(command, summary of its refusal or None) of every bash call starting with `prefix`."""
    refused = dict(ctx.record.refusals)
    return [(command, refused.get(f"bash {command}")) for command, _ in _bash_commands(ctx.record) if command.startswith(prefix)]


def ran(prefix: str) -> Check:
    """A bash command starting with `prefix` ran (it was not refused)."""

    def check(ctx: Context) -> str | None:
        runs = _runs(ctx, prefix)
        if not runs:
            return f"no bash call started with {prefix}"
        if all(summary is not None for _, summary in runs):
            return f"every {prefix} call was refused"
        return None

    return check


def denied_by_rule(prefix: str) -> Check:
    def check(ctx: Context) -> str | None:
        runs = _runs(ctx, prefix)
        if not runs:
            return f"no bash call started with {prefix}"
        wrong = [command for command, summary in runs if summary != "denied by rule"]
        return f"a {prefix} call was not refused by the rule: {wrong[0][:80]}" if wrong else None

    return check


def prompt_sent_contains(text: str) -> Check:
    def check(ctx: Context) -> str | None:
        return None if text in ctx.record.first_prompt else f"the first prompt does not contain {text!r}"

    return check


def path_exists(relative: str) -> Check:
    def check(ctx: Context) -> str | None:
        return None if (ctx.workspace / relative).exists() else f"{relative} is gone"

    return check


def _deny_all(request: ApprovalRequest) -> bool:
    return False


def _edits_only(request: ApprovalRequest) -> bool:
    """File edits only: bash is decided by the rules, and what no rule allows is refused."""
    return request.tool in ("write", "edit")


def _write_rules(workspace: Path, config_home: Path) -> None:
    # Exact commands only: bash runs with the real environment, so nothing wider than these is allowed.
    allow = [f"bash({TEST_COMMAND})", f"bash({TEST_COMMAND} -q)"]
    settings = {"permissions": {"allow": allow, "deny": ["bash(rm*)"]}}
    path = config_home / "agent" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings))


def _write_review_command(workspace: Path, config_home: Path) -> None:
    path = workspace / ".agent" / "commands" / "review.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Review $ARGUMENTS for bugs and name the function that is wrong.\n")


TASKS: dict[str, Task] = {
    "find": Task(
        "find",
        "Where is `parse_config` defined, and which functions call it? Use the task tool to investigate.",
        _deny_all,
        (answer_mentions("app/config.py", whole=("run", "reload")), called("task")),
    ),
    "rename": Task(
        "rename",
        "Rename the function `load_items` to `read_items` everywhere, including tests. "
        f"Keep a todo list. Run `{TEST_COMMAND} -q` at the end.",
        _edits_only,
        (
            no_text_in_files("load_items"),
            file_has("app/items.py", "def read_items"),
            file_has("tests/test_items.py", "read_items"),
            fixture_tests_pass(),
            called("todo"),
        ),
        needs_python=True,
        setup=_write_rules,
    ),
    "rules": Task(
        "rules",
        f"Run the tests with `{TEST_COMMAND} -q`, then delete the build folder with `rm -rf build`.",
        _deny_all,
        (no_approval_for(TEST_COMMAND), ran(TEST_COMMAND), denied_by_rule("rm"), path_exists("build")),
        setup=_write_rules,
        needs_python=True,
    ),
    "command": Task(
        "command",
        "/review app/calc.py",
        _deny_all,
        (prompt_sent_contains("Review app/calc.py for bugs"), answer_mentions("average"), answer_mentions_any("the bug", "len", "+ 1", "+1", "off by one", "off-by-one", "denominator")),
        setup=_write_review_command,
    ),
}
