"""Zero-shot production predictor.

The Chronos-2 pipeline is pre-trained; there is no artifact to load. We build the
forecaster once (its weights download lazily on first use) and, on every call,
refit it on the *latest* history — for a foundation model `fit` just stores the
context window, so this is cheap and keeps the context fresh — then forecast the
next `horizon_h` hours conditioned on the future weather.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import pandas as pd

from ..config import Settings
from ..model.builder import build_forecaster
from ..model.postprocess import predictions_to_wide
from .schemas import ForecastPayload

LOGGER = logging.getLogger(__name__)

class Predictor:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self._separate_targets = (
            cfg.model.pv_context_length is not None
            or cfg.model.demand_context_length is not None
        )
        if self._separate_targets:
            pv_cfg = cfg.model.model_copy(
                update={
                    "series_cols": ["pv_avg"],
                    "context_length": cfg.model.pv_context_length or cfg.model.context_length,
                }
            )
            demand_cfg = cfg.model.model_copy(
                update={
                    "series_cols": ["demand"],
                    "context_length": cfg.model.demand_context_length or cfg.model.context_length,
                }
            )
            self.forecasters = {
                "pv_avg": build_forecaster(pv_cfg),
                "demand": build_forecaster(demand_cfg),
            }
        else:
            self.forecaster = build_forecaster(cfg.model)
        self.version = cfg.model.model_id

    @staticmethod
    def _series_by_target(series_hist: pd.DataFrame, target: str) -> dict[str, pd.Series]:
        values = series_hist.xs(target, level="series_id")["value"].copy()
        index = pd.DatetimeIndex(values.index)
        if index.freq is None:
            inferred = index.inferred_freq
            if inferred is None:
                raise ValueError(f"forecast context for {target} has no regular frequency")
            index = pd.DatetimeIndex(index, freq=inferred)
        values.index = index
        return {target: values}

    def predict(
        self,
        issued_at: pd.Timestamp,
        series_hist: pd.DataFrame,
        exog_hist: pd.DataFrame | None,
        exog_future: pd.DataFrame | None,
        horizon_h: int | None = None,
        *,
        target_frequency_h: float | None = None,
        source_id: str = "campus-telemetry-2025-2026",
        context_time: str | None = None,
        context_steps: int = 0,
        cold_start: bool = False,
        covariate_mode: Literal["ecmwf", "none", "shortwave"] = "ecmwf",
    ) -> ForecastPayload:
        horizon = self.cfg.model.horizon_h if horizon_h is None else horizon_h
        frequency = (
            self.cfg.model.target_frequency_h if target_frequency_h is None else target_frequency_h
        )
        if not 1 <= horizon <= 168:
            raise ValueError("forecast horizon must be between 1 and 168 hours")
        steps = int(round(horizon / frequency))
        if not math.isclose(horizon / frequency, steps) or steps < 1:
            raise ValueError(
                f"horizon/target_frequency must be a positive integer: "
                f"{horizon} / {frequency}"
            )
        if exog_future is not None and len(exog_future) < steps:
            raise ValueError(
                f"future exogenous data has {len(exog_future)} rows; {steps} are required"
            )

        # A dict avoids skforecast's repeated long-DataFrame transformation on
        # every rolling simulator step.
        series_by_id = {
            str(series_id): series_hist.xs(series_id, level="series_id")["value"]
            for series_id in series_hist.index.get_level_values("series_id").unique()
        }
        # Data-quality flag: which context steps were imputed vs observed (only
        # present when `prepare` carried the column, e.g. the offline/live paths).
        context_is_imputed: dict[str, list[bool]] = {}
        if "is_imputed" in series_hist.columns:
            for series_id in series_hist.index.get_level_values("series_id").unique():
                context_is_imputed[str(series_id)] = [
                    bool(value)
                    for value in series_hist.xs(series_id, level="series_id")["is_imputed"]
                ]
        if self._separate_targets:
            outputs: list[pd.DataFrame] = []
            for target, forecaster in self.forecasters.items():
                target_series = self._series_by_target(series_hist, target)
                if exog_hist is None or exog_future is None:
                    forecaster.fit(series=target_series)
                    prediction = forecaster.predict(steps=steps)
                else:
                    forecaster.fit(series=target_series, exog=exog_hist)
                    prediction = forecaster.predict(steps=steps, exog=exog_future.iloc[:steps])
                wide_target = predictions_to_wide(prediction)
                if target not in wide_target.columns:
                    if len(wide_target.columns) != 1:
                        raise ValueError(f"forecast for {target} did not return one series")
                    wide_target = wide_target.rename(columns={wide_target.columns[0]: target})
                outputs.append(wide_target[[target]])
            wide = pd.concat(outputs, axis=1)
        else:
            # Refit only stores the fresh context; it does not train Chronos.  The
        # external pymgrid benchmark deliberately has no campus weather match,
        # so its context-only path omits exogenous variables altogether.
            if exog_hist is None or exog_future is None:
                self.forecaster.fit(series=series_by_id)
                preds = self.forecaster.predict(steps=steps)
            else:
                self.forecaster.fit(series=series_by_id, exog=exog_hist)
                preds = self.forecaster.predict(steps=steps, exog=exog_future.iloc[:steps])
            wide = predictions_to_wide(preds)
        return ForecastPayload(
            issued_at=issued_at.isoformat(),
            horizon_h=horizon,
            frequency_h=frequency,
            issue_frequency_h=frequency,
            model_version=self.version,
            timestamps=[pd.Timestamp(ts).isoformat() for ts in wide.index],
            units={col: "kw" for col in wide.columns},
            forecast={col: wide[col].tolist() for col in wide.columns},
            source_id=source_id,
            context_time=context_time,
            context_steps=context_steps,
            pv_context_steps=(
                len(series_hist.xs("pv_avg", level="series_id"))
                if "pv_avg" in series_hist.index.get_level_values("series_id")
                else 0
            ),
            demand_context_steps=(
                len(series_hist.xs("demand", level="series_id"))
                if "demand" in series_hist.index.get_level_values("series_id")
                else 0
            ),
            cold_start=cold_start,
            covariate_mode=covariate_mode,
            context_is_imputed=context_is_imputed,
        )
