"""Wire schemas for aligned PV/demand forecast payloads and HTTP responses."""

from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ForecastContext(BaseModel):
    """Scenario-aligned history supplied by the simulator in canonical kW.

    This is intentionally a value-only context.  The service must not infer a
    source from a timestamp and replace it with its configured campus PV/load
    files; the caller owns the replay boundary.
    """

    source_id: str = Field(min_length=1)
    frequency_h: float = Field(gt=0.0)
    pv_kw: list[float] = Field(min_length=1)
    demand_kw: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_series(self) -> ForecastContext:
        if not self.pv_kw or not self.demand_kw:
            raise ValueError("context pv_kw and demand_kw must both be non-empty")
        if not all(isfinite(value) and value >= 0.0 for value in self.pv_kw):
            raise ValueError("context pv_kw must contain finite non-negative kW values")
        if not all(isfinite(value) and value >= 0.0 for value in self.demand_kw):
            raise ValueError("context demand_kw must contain finite non-negative kW values")
        return self


class ForecastRequest(BaseModel):
    """Optional per-request override of the configured production horizon."""

    horizon_h: int = Field(ge=1, le=168)
    # Spacing between forecast points in hours (1.0 hourly; 0.25 = 15-minute).
    target_frequency_h: float = Field(default=1.0, gt=0.0)
    issued_at: datetime | None = None
    context: ForecastContext | None = None

    @model_validator(mode="after")
    def _validate_frequency(self) -> ForecastRequest:
        steps = self.horizon_h / self.target_frequency_h
        if not isfinite(steps) or not steps.is_integer() or steps < 1:
            raise ValueError(
                "horizon_h / target_frequency_h must be a positive integer "
                f"({self.horizon_h} / {self.target_frequency_h})"
            )
        return self


class ForecastPayload(BaseModel):
    issued_at: str
    horizon_h: int
    frequency_h: float = 1.0
    # How often a snapshot is issued (informational; 15-minute migration uses
    # the target frequency one-to-one with controller decisions).
    issue_frequency_h: float | None = None
    model_version: str
    timestamps: list[str] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
    forecast: dict[str, list[float]]
    # The scheduler's legacy no-context path remains campus telemetry.  Every
    # simulator request carries its own explicit source id instead.
    source_id: str = "campus-telemetry-2025-2026"
    context_time: str | None = None
    context_steps: int = Field(default=0, ge=0)
    pv_context_steps: int = Field(default=0, ge=0)
    demand_context_steps: int = Field(default=0, ge=0)
    cold_start: bool = False
    covariate_mode: Literal["ecmwf", "none", "shortwave"] = "ecmwf"
    # Per-series data-quality flag: True at a context step where the value was
    # imputed (not observed). Lets consumers distinguish real from filled input.
    context_is_imputed: dict[str, list[bool]] = Field(default_factory=dict)


class Health(BaseModel):
    status: str
    model_version: str | None
    last_run: str | None
