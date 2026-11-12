import importlib.metadata

import typer

from src.command.health import app as health_app

app = typer.Typer(name="agent", help="Agent command-line client.", no_args_is_help=True)
app.add_typer(health_app, name="health")


def _version(value: bool) -> None:
    if value:
        typer.echo(importlib.metadata.version("agent-cli"))
        raise typer.Exit()


@app.callback()
def root(
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Print the version"),
) -> None:
    """Agent command-line client."""


def main() -> None:
    app()
