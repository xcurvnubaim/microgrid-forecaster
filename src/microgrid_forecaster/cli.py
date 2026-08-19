"""CLI entrypoint: `forecaster train | serve | predict`."""

from __future__ import annotations

import json
import logging

import typer
import uvicorn

from .config import Settings

app = typer.Typer(add_completion=False, help="Microgrid PV forecaster (Chronos-2)")

LOGGING_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


@app.callback()
def _configure_logging() -> None:
    """Turn on INFO-level logs so forecast flow traces (e.g. runner.run_once)
    are visible on the terminal when running `forecaster evaluate|predict|serve`.
    No handler/config existed before, so only WARNING+ ever reached stderr.
    """
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, force=True)

DEFAULT_CFG = "configs/forecaster.yaml"


@app.command()
def evaluate(config: str = DEFAULT_CFG) -> None:
    """Offline: hold out the tail of the historical CSV and report accuracy.

    No model is trained or saved — Chronos-2 is zero-shot. This only tells you
    how well it forecasts and writes a metrics report.
    """
    from .model.evaluate import evaluate as _evaluate

    result = _evaluate(Settings.from_yaml(config))
    typer.echo(json.dumps(result, indent=2))


@app.command()
def predict(
    config: str = DEFAULT_CFG,
    horizon: int | None = None,
    target_frequency: float | None = None,
) -> None:
    """Production one-shot forecast, printed as JSON (no publish)."""
    from .service.runner import Service

    svc = Service(Settings.from_yaml(config))
    payload = svc.run_once(horizon_h=horizon, target_frequency_h=target_frequency)
    typer.echo(payload.model_dump_json(indent=2))


@app.command()
def serve(config: str = DEFAULT_CFG, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Production: run the live service (scheduler + Redis publisher + HTTP API)."""
    from .service.api import create_app
    from .service.runner import Service

    svc = Service(Settings.from_yaml(config))
    svc.start_scheduler()
    api = create_app(run_once=svc.run_once)
    uvicorn.run(api, host=host, port=port)


if __name__ == "__main__":
    app()
