import importlib.metadata
from pathlib import Path

import typer

from src.command.health import app as health_app

app = typer.Typer(name="agent", help="Agent command-line client.", no_args_is_help=False)
app.add_typer(health_app, name="health")


def _version(value: bool) -> None:
    if value:
        typer.echo(importlib.metadata.version("agent-cli"))
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Print the version"),
) -> None:
    """Agent command-line client. Run without a command to open the coding agent."""
    if ctx.invoked_subcommand is None:
        from src.tui import run  # lazy: Textual is not needed for `health` or `--help`

        run(Path.cwd())


def main() -> None:
    app()
