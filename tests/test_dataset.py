"""PV and demand are prepared as aligned Chronos target series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microgrid_forecaster.config import DataPaths, ModelCfg, Settings
from microgrid_forecaster.data import dataset, loaders


def _settings(tmp_path, *, include_demand_path: bool = True) -> Settings:
    return Settings(
        data=DataPaths(
            pv_xlsx=tmp_path / "pv.xlsx",
            demand_xlsx=tmp_path / "demand.xlsx" if include_demand_path else None,
            ecmwf_csv=tmp_path / "weather.csv",
        ),
        model=ModelCfg(
            series_cols=["pv_avg", "demand"],
            train_start="2026-01-01",
            train_end="2026-01-01 23:00:00",
        ),
    )


def test_prepare_returns_aligned_pv_and_demand_series(monkeypatch, tmp_path) -> None:
    quarter_hour = pd.date_range("2026-01-01", periods=96, freq="15min")
    hourly = pd.date_range("2026-01-01", periods=24, freq="h")
    pv = pd.DataFrame(
        {
            "pv_avg": np.arange(96, dtype=float),
            "pv_max": np.arange(96, dtype=float),
            "pv_min": np.arange(96, dtype=float),
        },
        index=quarter_hour,
    )
    demand = pd.DataFrame(
        {"demand": np.arange(96, dtype=float) + 100.0},
        index=quarter_hour,
    )
    weather = pd.DataFrame(
        {column: np.arange(24, dtype=float) for column in loaders.EXOG_VARS},
        index=hourly,
    )
    monkeypatch.setattr(dataset.loaders, "load_pv", lambda _path: pv)
    monkeypatch.setattr(dataset.loaders, "load_demand", lambda _path: demand)
    monkeypatch.setattr(dataset.loaders, "load_ecmwf", lambda _path: weather)

    series, exog = dataset.prepare(_settings(tmp_path))

    counts = series.index.get_level_values("series_id").value_counts().to_dict()
    assert counts == {"pv_avg": 24, "demand": 24}
    assert exog.index.equals(hourly)


def test_prepare_carries_imputed_flag_roundtrip(monkeypatch, tmp_path) -> None:
    """`prepare` marks a gap-filled hour as imputed (True) and leaves rest False."""
    quarter_hour = pd.date_range("2026-01-01", periods=96, freq="15min")
    hourly = pd.date_range("2026-01-01", periods=24, freq="h")
    values = np.arange(96, dtype=float)
    values[1] = np.nan  # one missing sub-hour sample inside the first hour
    pv = pd.DataFrame(
        {"pv_avg": values, "pv_max": values, "pv_min": values}, index=quarter_hour
    )
    demand = pd.DataFrame(
        {"demand": np.arange(96, dtype=float) + 100.0}, index=quarter_hour
    )
    weather = pd.DataFrame(
        {column: np.arange(24, dtype=float) for column in loaders.EXOG_VARS},
        index=hourly,
    )
    monkeypatch.setattr(dataset.loaders, "load_pv", lambda _path: pv)
    monkeypatch.setattr(dataset.loaders, "load_demand", lambda _path: demand)
    monkeypatch.setattr(dataset.loaders, "load_ecmwf", lambda _path: weather)

    series, _ = dataset.prepare(_settings(tmp_path))

    assert "is_imputed" in series.columns
    pv_long = series.xs("pv_avg", level="series_id")
    # First hourly pv_avg had a sub-hour gap -> imputed; the rest observed.
    assert pv_long["is_imputed"].iloc[0] == True  # noqa: E712
    assert not pv_long["is_imputed"].iloc[1:].any()
    # Demand has no gaps -> entirely real.
    dem_long = series.xs("demand", level="series_id")
    assert not dem_long["is_imputed"].any()


def test_prepare_requires_demand_source_for_demand_target(tmp_path) -> None:
    with pytest.raises(ValueError, match="demand_xlsx"):
        dataset.prepare(_settings(tmp_path, include_demand_path=False))


def test_load_demand_reads_second_row_header(tmp_path) -> None:
    path = tmp_path / "demand.xlsx"
    source = pd.DataFrame(
        {
            "statstime": pd.date_range("2026-01-01", periods=2, freq="15min"),
            "demand": [54.4, 54.0],
        }
    )
    source.to_excel(path, index=False, startrow=1)

    loaded = loaders.load_demand(path)

    assert list(loaded.columns) == ["demand"]
    assert loaded.index.name == "datetime"
    assert loaded["demand"].tolist() == [54.4, 54.0]
