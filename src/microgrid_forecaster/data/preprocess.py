"""Turn raw loaders into the modelling frame, mirroring pv_forecast_ecmwf.ipynb.

Pipeline: NaN-aware hourly PV/demand resample -> concat with (already hourly)
weather -> slice to the context window -> linear-interpolate target gaps ->
split into a long-format series frame + a wide exogenous frame.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from skforecast.preprocessing import reshape_series_wide_to_long

FREQ = "1h"


def _nan_aware(agg: str) -> Callable[[pd.Series], float]:
    """Aggregator that yields NaN for an hour if any sub-hour sample is missing."""

    def f(s: pd.Series) -> float:
        if s.isna().any():
            return float("nan")
        return float(getattr(s, agg)())

    return f


# how each PV series collapses from sub-hourly raw to hourly
_PV_AGG = {"pv_avg": "mean", "pv_max": "max", "pv_min": "min"}


def resample_pv_hourly(pv: pd.DataFrame) -> pd.DataFrame:
    """Resample raw PV to an hourly grid; an hour with any gap stays NaN."""
    funcs = {c: _nan_aware(_PV_AGG[c]) for c in pv.columns if c in _PV_AGG}
    return pv.resample(FREQ).agg(funcs)


def resample_demand_hourly(demand: pd.DataFrame) -> pd.DataFrame:
    """Collapse 15-minute demand to hourly mean while preserving incomplete hours."""
    return demand.resample(FREQ).agg({"demand": _nan_aware("mean")})


def assemble_frame(
    targets_hourly: pd.DataFrame,
    weather: pd.DataFrame,
    series_cols: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Concat targets + weather, restrict to [start, end], force an hourly index."""
    multi = pd.concat([targets_hourly[series_cols], weather], axis=1)
    multi = multi.loc[start:end]
    return multi.asfreq(FREQ)


def find_gaps(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous (start, end) spans where ``mask`` is True (missing)."""
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None
    prev = None
    for ts, missing in mask.items():
        if missing and start is None:
            start = ts
        elif not missing and start is not None:
            gaps.append((start, prev))
            start = None
        prev = ts
    if start is not None:
        gaps.append((start, prev))
    return gaps


def impute_linear(
    multi: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Linear-interpolate ``cols`` in place; also report the pre-impute gaps."""
    out = multi.copy()
    gaps = find_gaps(out[cols].isna().any(axis=1))
    for col in cols:
        # interpolate interior gaps; bfill/ffill covers leading/trailing NaNs so
        # the foundation model always sees a contiguous context window.
        out[col] = out[col].interpolate(method="linear").bfill().ffill()
    return out, gaps


def imputed_flags(multi: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Boolean wide frame marking which hourly ``col`` cells are *not* observed.

    ``True`` means the value at that hour is imputed (filled by interpolation /
    bfill / ffill) rather than measured. Compute this from the **pre-imputation**
    frame (where the missing cells are still NaN) so the flag reflects the true
    gap pattern instead of the filled values.
    """
    return multi[cols].isna()


def split_series_exog(
    multi: pd.DataFrame,
    series_cols: list[str],
    exog_cols: list[str],
    imputed_mask: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the wide frame into (long series, wide exog) for skforecast.

    The long series frame gains an ``is_imputed`` boolean column (True where the
    value was not observed) so consumers can distinguish real from imputed input.
    Pass ``imputed_mask`` (from :func:`imputed_flags` on the pre-fill frame) to
    avoid losing the gap pattern after :func:`impute_linear` fills it.
    """
    series_long = reshape_series_wide_to_long(data=multi[series_cols])
    flags = multi[series_cols].isna() if imputed_mask is None else imputed_mask[series_cols]
    # Convert numpy bools to plain Python bools so the flag is JSON/payload friendly.
    series_long["is_imputed"] = [
        bool(value)
        for value in reshape_series_wide_to_long(data=flags)[["value"]].iloc[:, 0]
    ]
    exog_wide = multi[exog_cols].copy()
    return series_long, exog_wide
