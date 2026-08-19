"""FastAPI app: health, latest forecast, force a fresh forecast.

The scheduler and the POST handler share one `run_once` callable injected via
app.state, so a manual trigger and a cron tick do exactly the same thing.
"""

from __future__ import annotations

from typing import Protocol, cast

from fastapi import FastAPI, HTTPException

from .schemas import ForecastContext, ForecastPayload, ForecastRequest, Health


class RunOnce(Protocol):
    def __call__(
        self,
        horizon_h: int | None = None,
        issued_at: str | None = None,
        context: ForecastContext | None = None,
        target_frequency_h: float | None = None,
    ) -> ForecastPayload: ...


def create_app(run_once: RunOnce) -> FastAPI:
    app = FastAPI(title="microgrid-forecaster")
    app.state.last = None

    @app.get("/health", response_model=Health)
    def health() -> Health:
        last = cast(ForecastPayload | None, app.state.last)
        return Health(
            status="ok",
            model_version=last.model_version if last else None,
            last_run=last.issued_at if last else None,
        )

    @app.get("/forecast/latest", response_model=ForecastPayload | None)
    def latest() -> ForecastPayload | None:
        return cast(ForecastPayload | None, app.state.last)

    @app.post("/forecast", response_model=ForecastPayload)
    def forecast(request: ForecastRequest | None = None) -> ForecastPayload:
        try:
            payload = run_once(
                horizon_h=request.horizon_h if request else None,
                issued_at=(
                    request.issued_at.isoformat()
                    if request is not None and request.issued_at is not None
                    else None
                ),
                context=request.context if request is not None else None,
                target_frequency_h=request.target_frequency_h if request else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        app.state.last = payload
        return payload

    return app
