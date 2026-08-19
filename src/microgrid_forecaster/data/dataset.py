"""Orchestrate loaders + preprocess into the (series_long, exog_wide) pair.

Shared by the offline `evaluate` step and the live production loop, so both see
an identical view of the data. Today it reads historical CSV/XLSX; swap the
loader calls for a DB/Redis source in production without touching callers.
"""

from __future__ import annotations

import pandas as pd

from ..config import Settings
from . import loaders, preprocess


def prepare(cfg: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (series_long, exog_wide) for the configured series + window."""
    m = cfg.model
    if "demand" in m.series_cols and cfg.data.demand_xlsx is None:
        raise ValueError("data.demand_xlsx is required to forecast demand")
    targets = preprocess.resample_pv_hourly(loaders.load_pv(cfg.data.pv_xlsx))
    if "demand" in m.series_cols:
        assert cfg.data.demand_xlsx is not None
        demand = preprocess.resample_demand_hourly(
            loaders.load_demand(cfg.data.demand_xlsx)
        )
        targets = pd.concat([targets, demand], axis=1)
    missing = sorted(set(m.series_cols) - set(targets.columns))
    if missing:
        raise ValueError(f"unsupported or unavailable forecast series: {missing}")
    weather = loaders.load_ecmwf(cfg.data.ecmwf_csv)
    multi = preprocess.assemble_frame(
        targets,
        weather,
        m.series_cols,
        m.train_start,
        m.train_end,
    )
    flags = preprocess.imputed_flags(multi, m.series_cols)  # pre-imputation mask
    multi, _gaps = preprocess.impute_linear(multi, m.series_cols)
    return preprocess.split_series_exog(multi, m.series_cols, loaders.EXOG_VARS, imputed_mask=flags)
