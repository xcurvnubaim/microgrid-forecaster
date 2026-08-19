"""Typed configuration. Loads configs/forecaster.yaml, overridable by env vars.

Env vars use the prefix ``MGF_`` and ``__`` as a nested delimiter, e.g.
``MGF_REDIS__URL=redis://prod:6379/0`` or ``MGF_MODE=production``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Mode = Literal["evaluate", "production"]


class DataPaths(BaseModel):
    """Historical CSV/XLSX sources used as zero-shot model context.

    ``demand_xlsx`` is required when ``model.series_cols`` includes ``demand``.
    SOC remains optional and is not currently forecast.

    ``radiation_csv`` / ``radiation_column`` are the existing 15-minute
    interpolated shortwave-radiation trajectory used for the 15-minute forecast
    target (F2). The column is read as-is; no weather alignment is regenerated.
    """

    pv_xlsx: Path
    ecmwf_csv: Path
    soc_csv: Path | None = None
    demand_xlsx: Path | None = None
    radiation_csv: Path | None = None
    radiation_column: str = "shortwave_radiation_wm2"


class ModelCfg(BaseModel):
    # allow a field named ``model_id`` without pydantic's protected-namespace warning
    model_config = ConfigDict(protected_namespaces=())

    # --- foundation model (zero-shot; no weights are learned) ---
    model_id: str = "autogluon/chronos-2-small"
    context_length: int = 500
    # F3 may use target-specific contexts. None preserves the shared-context
    # behavior used by the existing F0/F2 configurations.
    pv_context_length: int | None = Field(default=None, ge=1)
    demand_context_length: int | None = Field(default=None, ge=1)

    # --- what to forecast ---
    series_cols: list[str] = Field(default_factory=lambda: ["pv_avg", "demand"])
    horizon_h: int = Field(default=24, ge=1, le=168)  # default production horizon
    # Spacing between predicted points in hours; 0.25 = 15-minute targets (F2).
    target_frequency_h: float = Field(default=1.0, gt=0.0)

    # --- training window + preprocessing ---
    train_start: str = "2025-12-01"
    train_end: str = "2026-03-31"
    holdout_cutoff: str = "2026-03-29 23:59:59"  # <= is train, > is holdout eval

    mae_gate: float | None = None  # fail training if holdout MAE exceeds this


class RedisCfg(BaseModel):
    enabled: bool = False
    url: str = "redis://localhost:6379/0"
    channel: str = "forecaster:load"


class ScheduleCfg(BaseModel):
    cron: str = "7 * * * *"  # hourly at :07 (avoid the :00 stampede)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MGF_", env_nested_delimiter="__", extra="ignore"
    )

    mode: Mode = "production"

    artifact_dir: Path = Path("artifacts")  # evaluation reports (no weights)
    # Optional append-only JSONL audit trail. ``None`` keeps HTTP inference
    # free of filesystem/log-format dependencies.
    forecast_log: Path | None = None
    data: DataPaths
    model: ModelCfg = ModelCfg()
    redis: RedisCfg = RedisCfg()
    schedule: ScheduleCfg = ScheduleCfg()

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**raw)
