"""Raw data loaders. One function per source; each returns a tidy DataFrame
indexed by a ``datetime`` DatetimeIndex.

These wrap the file reads from pv_forecast_ecmwf.ipynb. Swap the bodies for
DB/Redis reads later (production mode) without changing callers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EXOG_VARS = [
    "temperature_2m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "wind_speed_10m",
    "cloud_cover",
]


def load_pv(path: str | Path) -> pd.DataFrame:
    """PV xlsx with columns 時間 / p:avg / p:max / p:min -> indexed pv_avg/max/min."""
    df = pd.read_excel(path)
    df = df.rename(
        columns={"時間": "datetime", "p:avg": "pv_avg", "p:max": "pv_max", "p:min": "pv_min"}
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime")[["pv_avg", "pv_max", "pv_min"]]


def load_ecmwf(path: str | Path) -> pd.DataFrame:
    """ECMWF IFS exogenous weather features (already hourly).

    The file's timestamp column is ``time``; we normalise it to a ``datetime``
    index so every loader shares the same index name.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime")[EXOG_VARS]


def load_soc(path: str | Path) -> pd.DataFrame:
    """Battery state-of-charge CSV (時間 / MBMS:soc_sys(%)). Unused by pv_avg model."""
    df = pd.read_csv(path)
    df = df.rename(columns={"時間": "datetime", "MBMS:soc_sys(%)": "soc"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime")[["soc"]]


def load_demand(path: str | Path) -> pd.DataFrame:
    """Campus demand xlsx (header row 2, ``statstime`` + ``demand`` in kW)."""
    df = pd.read_excel(path, header=1)
    df = df.rename(columns={"statstime": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    return df.set_index("datetime")[["demand"]]


def load_radiation(path: str | Path, column: str = "shortwave_radiation_wm2") -> pd.DataFrame:
    """Load the existing 15-minute interpolated shortwave radiation trajectory.

    The processed PV candidate already carries ``shortwave_radiation_wm2`` on a
    regular 15-minute grid (linearly interpolated from native hourly radiation,
    flagged by ``weather_is_interpolated`` / ``weather_source_resolution_min``).
    This loader reads that column verbatim — no weather alignment is regenerated.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"timestamp": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    if column not in df.columns:
        raise ValueError(
            f"radiation column {column!r} not found in {path}; "
            f"available columns: {list(df.columns)}"
        )
    out = pd.DataFrame({"shortwave_radiation_wm2": pd.to_numeric(df[column], errors="coerce")})
    out.index = df["datetime"]
    return out.sort_index()
