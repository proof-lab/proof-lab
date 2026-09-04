"""Proof Lab CLI entry point.

Usage::

    prooflab --help
    prooflab --version
    prooflab config show
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import typer

app = typer.Typer(
    name="prooflab",
    help="Proof Lab – quantitative research and algorithmic trading platform.",
    add_completion=False,
    no_args_is_help=True,
)

config_app = typer.Typer(help="Configuration utilities.")
app.add_typer(config_app, name="config")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_version() -> str:
    try:
        return version("prooflab")
    except PackageNotFoundError:  # pragma: no cover
        return "0.1.0"


def _version_callback(value: bool) -> None:  # noqa: FBT001
    if value:
        typer.echo(f"prooflab {_get_version()}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Root callback (handles --version)
# ---------------------------------------------------------------------------


@app.callback()
def _root(
    _version: bool | None = typer.Option(  # noqa: UP007
        None,
        "--version",
        "-v",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Proof Lab – quantitative research and algorithmic trading platform."""


# ---------------------------------------------------------------------------
# config sub-commands
# ---------------------------------------------------------------------------


@config_app.command("show")
def config_show() -> None:
    """Pretty-print the fully resolved configuration.

    Useful for verifying which values are active after YAML merging and
    environment variable overrides.
    """
    import json

    from prooflab.config import get_settings
    from prooflab.logging import configure_logging

    settings = get_settings()
    configure_logging(settings)

    typer.echo(json.dumps(settings.model_dump(), indent=2, default=str))


# ---------------------------------------------------------------------------
# ui command
# ---------------------------------------------------------------------------


@app.command("ui")
def launch_ui(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address to bind to."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to."),
    reload: bool = typer.Option(False, "--reload", help="Enable live auto-reload."),
) -> None:
    """Launch the Proof Lab Graphical User Interface presentation server."""
    import uvicorn

    typer.echo(f"Starting Proof Lab UI at http://{host}:{port}/ ...")
    uvicorn.run("prooflab.api.app:app", host=host, port=port, reload=reload)

