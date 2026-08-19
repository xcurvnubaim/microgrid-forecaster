"""Offline evaluation for the zero-shot foundation forecaster.

Chronos-2 is pre-trained and never learns from our data, so there is no model
to *train* or persist. This step exists only to answer "is it good enough, and
with what settings?": it holds out the tail of the historical window, forecasts
it, and reports accuracy. Nothing is saved except an optional metrics report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..config import Settings
from ..data import dataset
from . import builder
from .postprocess import predictions_to_wide


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(cfg: Settings) -> dict[str, object]:
    series_long, exog_wide = dataset.prepare(cfg)

    cutoff = cfg.model.holdout_cutoff
    dt = series_long.index.get_level_values("datetime")
    series_train = series_long.loc[dt <= cutoff]
    series_test = series_long.loc[dt > cutoff]
    exog_train = exog_wide.loc[exog_wide.index <= cutoff]
    exog_test = exog_wide.loc[exog_wide.index > cutoff]
    if series_test.empty or exog_test.empty:
        raise ValueError(f"holdout_cutoff {cutoff!r} leaves no evaluation data")

    # "fit" only stores the context window; the forecast is zero-shot.
    forecaster = builder.build_forecaster(cfg.model)
    forecaster.fit(series=series_train, exog=exog_train)
    preds = predictions_to_wide(forecaster.predict(steps=len(exog_test), exog=exog_test))
    truth = series_test["value"].unstack(level="series_id")

    per_series: dict[str, dict[str, float]] = {}
    for sid in cfg.model.series_cols:
        aligned = pd.concat(
            [truth[sid].rename("y"), preds[sid].rename("yhat")], axis=1
        ).dropna()
        y, yhat = aligned["y"].to_numpy(), aligned["yhat"].to_numpy()
        per_series[sid] = {
            "mae": float(mean_absolute_error(y, yhat)),
            "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
            "mape_pct": _safe_mape(y, yhat),
            "r2": float(r2_score(y, yhat)),
            "n": int(len(aligned)),
        }

    mae = float(np.mean([m["mae"] for m in per_series.values()]))
    if cfg.model.mae_gate is not None and mae > cfg.model.mae_gate:
        raise RuntimeError(f"Holdout MAE {mae:.3f} exceeds gate {cfg.model.mae_gate}")

    report: dict[str, object] = {
        "model_id": cfg.model.model_id,
        "context_length": cfg.model.context_length,
        "series": cfg.model.series_cols,
        "holdout_cutoff": cutoff,
        "holdout_steps": len(exog_test),
        "mae_mean": mae,
        "per_series": per_series,
    }

    out = Path(cfg.artifact_dir) / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    report["report_path"] = str(out)
    return report
