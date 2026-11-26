from __future__ import annotations

import json

import typer

from src.client import ApiError, get_api_client

app = typer.Typer(help="API health.")


@app.command("check")
def check(as_json: bool = typer.Option(False, "--json", help="Print the raw report as JSON")) -> None:
    """Report the API's version, environment, uptime and database status."""
    try:
        report = get_api_client().health()
    except ApiError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    if as_json:
        typer.echo(json.dumps(report, indent=2))
        return
    db = report.get("db") or {}
    typer.echo(f"version     {report.get('version')}")
    typer.echo(f"environment {report.get('environment')}")
    typer.echo(f"uptime      {report.get('uptime')}")
    typer.echo(f"db          {db.get('status')} ({db.get('version')})")
