import importlib.metadata
from pathlib import Path
from typing import Any

import typer
from typer.core import TyperGroup

from src.command.health import app as health_app

PICK = "__pick__"  # `--resume` with no value: the user picks from a list


class _OptionalResumeValue(TyperGroup):
    """Typer 0.29 has no option with an optional value (`is_flag`/`flag_value` are ignored on a
    `str` option), so `--resume` followed by nothing or by another option becomes `--resume=PICK`."""

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        fixed: list[str] = []
        for i, arg in enumerate(args):
            if arg == "--" or not arg.startswith("-"):  # the root options end at `--` or the subcommand name
                fixed.extend(args[i:])
                break
            nxt = args[i + 1] if i + 1 < len(args) else None
            fixed.append(f"--resume={PICK}" if arg == "--resume" and (nxt is None or nxt.startswith("-")) else arg)
            if arg == "--resume" and nxt is not None and not nxt.startswith("-"):
                fixed.append(nxt)  # its value, not a subcommand name
                fixed.extend(args[i + 2 :])
                break
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
) -> None:
    """Agent command-line client. Run without a command to open the coding agent."""
    if ctx.invoked_subcommand is None:
        from src.tui import StartChoice, run  # lazy: Textual is not needed for `health` or `--help`

        if continue_ and resume is not None:
            raise typer.BadParameter("use either --continue or --resume")
        if continue_:
            start = StartChoice("continue")
        elif resume is None:
            start = StartChoice("new")
        elif resume == PICK:
            start = StartChoice("pick")
        else:
            start = StartChoice("resume", resume)
        if code := run(Path.cwd(), start):
            raise typer.Exit(code)  # e.g. 128 + the signal that ended it


def main() -> None:
    app()
