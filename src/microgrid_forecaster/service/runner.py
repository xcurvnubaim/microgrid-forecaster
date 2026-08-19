"""Wires predictor + publisher + scheduler into one long-running service.

`run_once()` is the single unit of work shared by the hourly cron job and the
POST /forecast endpoint: gather recent history + the weather over the horizon ->
predict -> optionally publish -> log.

Data source note: history + exog are read from the historical CSV today. In a
real deployment these become live reads (recent measurements + the latest ECMWF
forecast); only the two `_history`/`_exog_future` helpers need to change.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ..config import Settings
from ..data import dataset, loaders
from .predict import Predictor
from .publisher import RedisPublisher
from .schemas import ForecastContext, ForecastPayload

LOGGER = logging.getLogger(__name__)


class Service:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.predictor = Predictor(cfg)
        self.publisher = RedisPublisher(cfg.redis) if cfg.redis.enabled else None

    def _history(
        self, issued_at: pd.Timestamp | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Recent context ending at ``issued_at`` when replaying a historical step."""
        series, exog = dataset.prepare(self.cfg)
        if issued_at is None:
            return series, exog
        datetimes = series.index.get_level_values("datetime")
        series = series.loc[datetimes <= issued_at]
        exog = exog.loc[exog.index <= issued_at]
        if series.empty or exog.empty:
            raise ValueError(f"no forecast history is available at or before {issued_at}")
        return series, exog

    def _exog_future(
        self, issued_at: pd.Timestamp, horizon_h: int, frequency_h: float = 1.0
    ) -> pd.DataFrame:
        """ECMWF weather over the next `horizon_h` hours. Stub: slice the CSV forward."""
        exog = loaders.load_ecmwf(self.cfg.data.ecmwf_csv)  # already hourly
        steps = int(round(horizon_h / frequency_h))
        start = (
            issued_at.floor("h") + pd.Timedelta(hours=1)
            if frequency_h == 1.0
            else issued_at + pd.Timedelta(hours=frequency_h)
        )
        idx = pd.date_range(
            start, periods=steps, freq=pd.to_timedelta(frequency_h, unit="h")
        )
        future = exog.reindex(idx, method="ffill")
        if future.isna().any().any():
            missing = int(future.isna().any(axis=1).sum())
            raise ValueError(
                f"future weather does not cover {missing} of {horizon_h} hours "
                f"after {issued_at}"
            )
        return future

    @staticmethod
    def _request_time(issued_at: str | None) -> pd.Timestamp:
        """Use a caller timestamp when supplied, otherwise a fresh service time.

        Positional benchmark replays intentionally omit ``issued_at``.  Giving
        them a service timestamp creates an independent forecast index without
        ever using the configured campus target history.
        """
        return pd.Timestamp(issued_at) if issued_at is not None else pd.Timestamp.now()

    def _context_history(
        self, context: ForecastContext, forecast_time: pd.Timestamp
    ) -> pd.DataFrame:
        """Build model history solely from caller-provided PV/demand kW values."""
        interval = pd.to_timedelta(context.frequency_h, unit="h")
        pv_index = pd.date_range(end=forecast_time, periods=len(context.pv_kw), freq=interval)
        demand_index = pd.date_range(
            end=forecast_time, periods=len(context.demand_kw), freq=interval
        )
        pv = pd.Series(context.pv_kw, index=pv_index, name="pv_avg")
        demand = pd.Series(context.demand_kw, index=demand_index, name="demand")
        # Keep target histories independent: F3 intentionally supplies different
        # context lengths, so aligning them into one wide frame would create NaNs.
        series = pd.concat(
            [pv.rename_axis("datetime"), demand.rename_axis("datetime")],
            keys=["pv_avg", "demand"],
            names=["series_id"],
        )
        series.index.names = ["series_id", "datetime"]
        series = series.rename("value").to_frame()
        # The HTTP contract targets the configured frequency. A context sampled
        # at the same frequency is consumed as-is; no later replay sample is
        # introduced by resampling.
        if series.empty:
            raise ValueError("forecast context has no usable history")
        unsupported = sorted(
            set(self.cfg.model.series_cols) - set(series.index.get_level_values("series_id"))
        )
        if unsupported:
            raise ValueError(
                "simulator forecast context does not provide configured series: "
                f"{unsupported}"
            )
        series = series.sort_index()
        # Caller-supplied context is entirely observed (no imputation happened here),
        # so every step is flagged real rather than silently absent.
        series["is_imputed"] = False
        return series

    def _context_exog_history(self, series_hist: pd.DataFrame) -> pd.DataFrame:
        """Align historical ECMWF covariates to context without loading PV/load files."""
        history_times = pd.DatetimeIndex(
            series_hist.index.get_level_values("datetime").unique()
        ).sort_values()
        weather = loaders.load_ecmwf(self.cfg.data.ecmwf_csv)
        exog = weather.reindex(history_times, method="ffill")
        if exog.isna().any().any():
            raise ValueError("ECMWF weather does not cover the supplied campus context")
        return exog

    def _radiation_exog(
        self, series_hist: pd.DataFrame, forecast_time: pd.Timestamp, steps: int, frequency_h: float
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build historical/future shortwave exog from the interpolated 15-min file.

        Reads the processed ``shortwave_radiation_wm2`` trajectory verbatim (the
        F2 weather decision). History is the radiation aligned to the context
        index (through ``forecast_time``); the future is the trajectory strictly
        after it, taken as an *assumed-known* weather forecast. The complete
        future trajectory is permitted as exogenous information, never actual
        future ``pv_kw``.
        """
        if self.cfg.data.radiation_csv is None:
            raise ValueError("F2 shortwave mode requires data.radiation_csv")
        radiation = loaders.load_radiation(
            self.cfg.data.radiation_csv, self.cfg.data.radiation_column
        ).rename(columns={"shortwave_radiation_wm2": "shortwave_radiation"})
        context_times = pd.DatetimeIndex(
            series_hist.index.get_level_values("datetime").unique()
        ).sort_values()
        exog_hist = radiation.reindex(context_times)
        if exog_hist["shortwave_radiation"].isna().any():
            raise ValueError(
                "shortwave radiation does not cover the supplied 15-minute context"
            )
        interval = pd.to_timedelta(frequency_h, unit="h")
        future_idx = pd.date_range(
            start=forecast_time + interval, periods=steps, freq=interval
        )
        exog_future = radiation.reindex(future_idx)
        if exog_future["shortwave_radiation"].isna().any():
            raise ValueError(
                f"shortwave radiation does not cover {steps} future steps after "
                f"{forecast_time}"
            )
        return exog_hist, exog_future

    def run_once(
        self,
        horizon_h: int | None = None,
        issued_at: str | None = None,
        context: ForecastContext | None = None,
        target_frequency_h: float | None = None,
    ) -> ForecastPayload:
        horizon = self.cfg.model.horizon_h if horizon_h is None else horizon_h
        frequency = (
            self.cfg.model.target_frequency_h if target_frequency_h is None else target_frequency_h
        )
        covariate_mode: Literal["ecmwf", "none", "shortwave"]
        if not 1 <= horizon <= 168:
            raise ValueError("forecast horizon must be between 1 and 168 hours")
        if frequency <= 0.0:
            raise ValueError("target_frequency_h must be positive")
        steps = int(round(horizon / frequency))
        if not math.isclose(horizon / frequency, steps) or steps < 1:
            raise ValueError(
                f"horizon/target_frequency must be a positive integer: {horizon}/{frequency}"
            )
        # 15-minute targets use the interpolated shortwave-radiation trajectory
        # (F2); hourly targets keep the ECMWF covariate set (F0).
        use_shortwave = frequency < 1.0 - 1e-9 and self.cfg.data.radiation_csv is not None
        if context is None and issued_at is None:
            series_hist, exog_hist = self._history()
            forecast_time = pd.Timestamp(
                series_hist.index.get_level_values("datetime").max()
            )
            exog_future: pd.DataFrame | None = (
                self._exog_future(forecast_time, horizon)
                if frequency == 1.0
                else self._exog_future(forecast_time, horizon, frequency)
            )
            source_id = "campus-telemetry-2025-2026"
            context_time = None
            context_steps = 0
            cold_start = False
            covariate_mode = "ecmwf"
        elif context is None:
            forecast_time = pd.Timestamp(issued_at)
            series_hist, exog_hist = self._history(forecast_time)
            latest_context = pd.Timestamp(
                series_hist.index.get_level_values("datetime").max()
            )
            expected_context = forecast_time.floor("h")
            if latest_context < expected_context:
                raise ValueError(
                    f"target history ends at {latest_context}; cannot issue forecast "
                    f"at {forecast_time}"
                )
            exog_future = (
                self._exog_future(forecast_time, horizon)
                if frequency == 1.0
                else self._exog_future(forecast_time, horizon, frequency)
            )
            source_id = "campus-telemetry-2025-2026"
            context_time = None
            context_steps = 0
            cold_start = False
            covariate_mode = "ecmwf"
        else:
            forecast_time = self._request_time(issued_at)
            series_hist = self._context_history(context, forecast_time)
            source_id = context.source_id
            context_time = issued_at
            context_steps = len(context.pv_kw)
            # Keep producing an explicitly labelled early forecast so reset
            # can use its one preceding measurement without look-ahead.
            cold_start = context_steps < min(96, self.cfg.model.context_length)
            if source_id.startswith("pymgrid"):
                exog_hist = None
                exog_future = None
                covariate_mode = "none"
            elif use_shortwave:
                exog_hist, exog_future = self._radiation_exog(
                    series_hist, forecast_time, steps, frequency
                )
                covariate_mode = "shortwave"
            else:
                exog_hist = self._context_exog_history(series_hist)
                exog_future = (
                    self._exog_future(forecast_time, horizon)
                    if frequency == 1.0
                    else self._exog_future(forecast_time, horizon, frequency)
                )
                covariate_mode = "ecmwf"

        LOGGER.info(
            "Issuing forecast at %s: %d steps @ %.3fh (%s)",
            forecast_time,
            steps,
            frequency,
            covariate_mode,
        )
        payload = self.predictor.predict(
            forecast_time,
            series_hist,
            exog_hist,
            exog_future,
            horizon_h=horizon,
            target_frequency_h=frequency,
            source_id=source_id,
            context_time=context_time,
            context_steps=context_steps,
            cold_start=cold_start,
            covariate_mode=covariate_mode,
        )
        if self.publisher is not None:
            self.publisher.publish(payload)
        try:
            self._append_log(payload)
        except OSError as exc:
            # Forecast delivery is the primary operation. A best-effort audit
            # trail must never turn a valid prediction into an HTTP 500.
            LOGGER.warning("Could not append forecast log: %s", exc)
        return payload

    def _append_log(self, payload: ForecastPayload) -> None:
        if self.cfg.forecast_log is None:
            return
        path = Path(self.cfg.forecast_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload.model_dump_json())
            handle.write("\n")

    def start_scheduler(self) -> BackgroundScheduler:
        sched = BackgroundScheduler()
        sched.add_job(self.run_once, CronTrigger.from_crontab(self.cfg.schedule.cron))
        sched.start()
        return sched
