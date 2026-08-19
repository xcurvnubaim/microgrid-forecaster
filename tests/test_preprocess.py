"""Preprocessing invariants: hourly grid, NaN-aware resample, gap imputation."""

import numpy as np
import pandas as pd

from microgrid_forecaster.data import loaders, preprocess


def _raw_pv(periods=96):
    """15-min PV with a missing block in the first hour (to exercise NaN handling)."""
    idx = pd.date_range("2026-01-01", periods=periods, freq="15min")
    pv_avg = np.arange(periods, dtype=float)
    pv_avg[1] = np.nan  # one gap inside the first hour
    return pd.DataFrame(
        {"pv_avg": pv_avg, "pv_max": pv_avg, "pv_min": pv_avg}, index=idx
    )


def _weather(index):
    return pd.DataFrame({c: np.random.rand(len(index)) for c in loaders.EXOG_VARS}, index=index)


def _raw_demand(periods=96):
    index = pd.date_range("2026-01-01", periods=periods, freq="15min")
    values = np.arange(periods, dtype=float) + 100.0
    return pd.DataFrame({"demand": values}, index=index)


def test_resample_is_hourly_and_nan_aware():
    out = preprocess.resample_pv_hourly(_raw_pv())
    deltas = out.index.to_series().diff().dropna().unique()
    assert deltas == pd.Timedelta("1h")
    # the first hour had a missing sub-hour sample -> stays NaN
    assert np.isnan(out["pv_avg"].iloc[0])
    assert out["pv_avg"].iloc[1:].notna().all()


def test_demand_resample_is_hourly_mean():
    out = preprocess.resample_demand_hourly(_raw_demand())

    assert len(out) == 24
    assert out["demand"].iloc[0] == 101.5


def test_find_gaps():
    s = pd.Series(
        [False, True, True, False, True],
        index=pd.date_range("2026-01-01", periods=5, freq="h"),
    )
    gaps = preprocess.find_gaps(s)
    assert gaps[0] == (s.index[1], s.index[2])
    assert gaps[1] == (s.index[4], s.index[4])


def test_impute_removes_nan_and_reports_gaps():
    pv = preprocess.resample_pv_hourly(_raw_pv())
    weather = _weather(pv.index)
    multi = preprocess.assemble_frame(
        pv, weather, ["pv_avg"], "2026-01-01", "2026-01-02"
    )
    imputed, gaps = preprocess.impute_linear(multi, ["pv_avg"])
    assert len(gaps) >= 1
    assert not imputed["pv_avg"].isna().any()


def test_split_series_exog_shapes():
    pv = preprocess.resample_pv_hourly(_raw_pv())
    weather = _weather(pv.index)
    multi = preprocess.assemble_frame(pv, weather, ["pv_avg"], "2026-01-01", "2026-01-02")
    flags = preprocess.imputed_flags(multi, ["pv_avg"])
    multi, _ = preprocess.impute_linear(multi, ["pv_avg"])
    series_long, exog_wide = preprocess.split_series_exog(
        multi, ["pv_avg"], loaders.EXOG_VARS, imputed_mask=flags
    )
    assert series_long.index.names == ["series_id", "datetime"]
    assert list(exog_wide.columns) == loaders.EXOG_VARS
    assert len(exog_wide) == len(multi)


def test_split_series_exog_carries_is_imputed():
    pv = preprocess.resample_pv_hourly(_raw_pv())
    weather = _weather(pv.index)
    multi = preprocess.assemble_frame(pv, weather, ["pv_avg"], "2026-01-01", "2026-01-02")
    flags = preprocess.imputed_flags(multi, ["pv_avg"])
    multi, _ = preprocess.impute_linear(multi, ["pv_avg"])
    series_long, _ = preprocess.split_series_exog(
        multi, ["pv_avg"], loaders.EXOG_VARS, imputed_mask=flags
    )
    assert "is_imputed" in series_long.columns
    # First hour had a sub-hour gap -> its hourly pv_avg was NaN -> imputed flag True.
    assert series_long["is_imputed"].iloc[0] == True  # noqa: E712
    # Observed hours are flagged real.
    assert not series_long["is_imputed"].iloc[1:].any()
