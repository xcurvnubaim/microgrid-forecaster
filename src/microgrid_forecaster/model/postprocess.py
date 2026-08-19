"""Normalise skforecast forecaster output to a wide datetime x series frame."""

from __future__ import annotations

import pandas as pd


def predictions_to_wide(preds: pd.DataFrame) -> pd.DataFrame:
    """Accept skforecast's long ([level, pred]) or already-wide predictions."""
    if {"level", "pred"}.issubset(preds.columns):
        return preds.pivot(columns="level", values="pred")
    return preds.copy()
