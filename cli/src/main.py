import importlib.metadata
from pathlib import Path
from typing import Any

import typer
from typer.core import TyperGroup

from src.command.health import app as health_app

PICK = "__pick__"  # `--resume` with no value: the user picks from a list
STDIN = "\x00stdin"  # `-p` with no prompt: read it from stdin (a NUL cannot come from a command line)


class _OptionalResumeValue(TyperGroup):
    """Typer 0.29 has no option with an optional value (`is_flag`/`flag_value` are ignored on a
    `str` option), so `--resume` followed by nothing or by another option becomes `--resume=PICK`."""

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        fixed: list[str] = []
        bare_print: int | None = None  # index in `fixed` of a `-p` that has no value yet
        printing = any(a in ("-p", "--print") or a.startswith("--print=") for a in args)
        i = 0
        while i < len(args):
            arg = args[i]
            if printing and arg == "--":
                raise typer.BadParameter("put the prompt right after -p (quoted), not after '--'")
            if printing and not arg.startswith("-") and arg not in self.commands and not (
                bare_print is not None
            ):
                raise typer.BadParameter(f"unexpected extra argument {arg!r}: quote the whole prompt, e.g. agent -p \"...\"")
            if arg == "--" or not arg.startswith("-"):  # the root options end at `--` or the subcommand name
                if bare_print is not None and arg != "--":  # `-p --continue "prompt"`: the first word is the prompt
                    fixed[bare_print] = f"--print={arg}"
                    i += 1
                fixed.extend(args[i:])
                break
            nxt = args[i + 1] if i + 1 < len(args) else None
            if arg in ("--model", "--allow") and nxt is not None:  # its value is not a subcommand name either
                fixed.extend((arg, nxt))
                i += 2
            elif arg == "--resume" and nxt is not None and not nxt.startswith("-"):
                fixed.extend((arg, nxt))  # its value, not a subcommand name
                i += 2
            elif arg in ("-p", "--print") and nxt is not None and not nxt.startswith("-"):
                fixed.append(f"--print={nxt}")
                i += 2
            elif arg in ("-p", "--print"):
                bare_print = len(fixed)
                fixed.append(f"--print={STDIN}")  # until a later word turns out to be the prompt
                i += 1
            else:
                fixed.append(f"--resume={PICK}" if arg == "--resume" else arg)
                i += 1
        return super().parse_args(ctx, fixed)


app = typer.Typer(name="agent", help="Agent command-line client.", no_args_is_help=False, cls=_OptionalResumeValue)
app.add_typer(health_app, name="health")


def _version(value: bool) -> None:
    if value:
        typer.echo(importlib.metadata.version("agent-cli"))
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Print the version"),
    continue_: bool = typer.Option(False, "--continue", help="Resume the latest session for this folder"),
    resume: str | None = typer.Option(None, "--resume", help="Resume a session by id (no value: pick from a list)"),
    model: str | None = typer.Option(None, "--model", help="Ollama model for this launch (not saved)"),
    print_: str | None = typer.Option(
        None, "-p", "--print", help="Run one turn without the UI and print the reply (no value: read the prompt from stdin)"
    ),
    allow: list[str] = typer.Option([], "--allow", help="With -p: allow a call for this run (repeatable), e.g. 'bash(make test)'"),
    quiet: bool = typer.Option(False, "--quiet", help="With -p: only warnings and errors on stderr"),
    trust_project: bool = typer.Option(
        False, "--trust-project", help="With -p: trust the project's settings and MCP file for this run (dangerous)"
    ),
) -> None:
    """Agent command-line client. Run without a command to open the coding agent."""
    if ctx.invoked_subcommand is None:
        from src.runtime import StartChoice  # UI-free: Textual is not needed for `health`, `--help` or `-p`

        if model is not None and not model.strip():
            raise typer.BadParameter("give a model name", param_hint="--model")
        if continue_ and resume is not None:
            raise typer.BadParameter("use either --continue or --resume")
        if print_ is None:
            for given, flag in ((bool(allow), "--allow"), (quiet, "--quiet"), (trust_project, "--trust-project")):
                if given:
                    raise typer.BadParameter(f"{flag} needs -p", param_hint=flag)
        elif resume == PICK:
            raise typer.BadParameter("give a session id: there is no picker with -p", param_hint="--resume")
        if continue_:
            start = StartChoice("continue")
        elif resume is None:
            start = StartChoice("new")
        elif resume == PICK:
            start = StartChoice("pick")
        else:
            start = StartChoice("resume", resume)
        chosen = model.strip() if model else None
        if print_ is not None:
            from src.headless import HeadlessOptions
            from src.headless.run import run_headless

            opts = HeadlessOptions(None if print_ == STDIN else print_, tuple(allow), quiet, trust_project, chosen, start)
            raise typer.Exit(run_headless(Path.cwd(), opts))
        from src.tui import run  # lazy: Textual is not needed for `health` or `--help`

        if code := run(Path.cwd(), start, model=chosen):
            raise typer.Exit(code)  # e.g. 128 + the signal that ended it


def main() -> None:
    app()
