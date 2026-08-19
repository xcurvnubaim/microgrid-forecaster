"""Service can run in HTTP-only mode without constructing or calling Redis."""

from __future__ import annotations

import json

import pandas as pd

from microgrid_forecaster.config import DataPaths, RedisCfg, Settings
from microgrid_forecaster.service.runner import Service
from microgrid_forecaster.service.schemas import ForecastContext, ForecastPayload


class _FakePredictor:
    def __init__(self, _cfg: Settings):
        pass

    def predict(self, *args, horizon_h: int, **kwargs) -> ForecastPayload:
        return ForecastPayload(
            issued_at="2026-07-23T12:00:00",
            horizon_h=horizon_h,
            model_version="chronos-test",
            forecast={
                "pv_avg": [10.0] * horizon_h,
                "demand": [20.0] * horizon_h,
            },
        )


def _settings(tmp_path, *, redis_enabled: bool) -> Settings:
    return Settings(
        data=DataPaths(
            pv_xlsx=tmp_path / "pv.xlsx",
            ecmwf_csv=tmp_path / "weather.csv",
        ),
        redis=RedisCfg(enabled=redis_enabled),
        forecast_log=tmp_path / "forecasts.jsonl",
    )


def test_http_only_run_does_not_construct_or_call_redis(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("microgrid_forecaster.service.runner.Predictor", _FakePredictor)

    def unexpected_redis(_cfg):
        raise AssertionError("RedisPublisher must not be constructed when Redis is disabled")

    monkeypatch.setattr(
        "microgrid_forecaster.service.runner.RedisPublisher",
        unexpected_redis,
    )
    service = Service(_settings(tmp_path, redis_enabled=False))
    history_index = pd.MultiIndex.from_product(
        [
            ["pv_avg", "demand"],
            [pd.Timestamp("2026-07-23 12:00:00")],
        ],
        names=["series_id", "datetime"],
    )
    monkeypatch.setattr(
        service,
        "_history",
        lambda _issued_at=None: (
            pd.DataFrame({"value": [1.0, 2.0]}, index=history_index),
            pd.DataFrame({"weather": [1.0]}),
        ),
    )
    monkeypatch.setattr(
        service,
        "_exog_future",
        lambda _issued_at, horizon_h: pd.DataFrame({"weather": range(horizon_h)}),
    )

    def failed_log(_payload):
        raise OSError("read-only audit directory")

    monkeypatch.setattr(service, "_append_log", failed_log)

    payload = service.run_once(horizon_h=3)

    assert service.publisher is None
    assert payload.horizon_h == 3
    assert payload.forecast["pv_avg"] == [10.0, 10.0, 10.0]
    assert payload.forecast["demand"] == [20.0, 20.0, 20.0]


def test_run_once_emits_issuing_forecast_info(caplog, monkeypatch, tmp_path) -> None:
    """The flow-trace INFO log in run_once actually fires (visible once the CLI
    enables INFO logging)."""
    import logging

    monkeypatch.setattr("microgrid_forecaster.service.runner.Predictor", _FakePredictor)
    service = Service(_settings(tmp_path, redis_enabled=False))
    history_index = pd.MultiIndex.from_product(
        [
            ["pv_avg", "demand"],
            [pd.Timestamp("2026-07-23 12:00:00")],
        ],
        names=["series_id", "datetime"],
    )
    monkeypatch.setattr(
        service,
        "_history",
        lambda _issued_at=None: (
            pd.DataFrame({"value": [1.0, 2.0]}, index=history_index),
            pd.DataFrame({"weather": [1.0]}),
        ),
    )
    monkeypatch.setattr(
        service,
        "_exog_future",
        lambda _issued_at, horizon_h: pd.DataFrame({"weather": range(horizon_h)}),
    )
    monkeypatch.setattr(service, "_append_log", lambda _payload: None)

    with caplog.at_level(logging.INFO, logger="microgrid_forecaster.service.runner"):
        service.run_once(horizon_h=3)

    messages = [r.getMessage() for r in caplog.records
                if r.name == "microgrid_forecaster.service.runner"]
    assert any("Issuing" in m or "issuing" in m for m in messages), messages


def test_forecast_log_is_append_only_jsonl(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("microgrid_forecaster.service.runner.Predictor", _FakePredictor)
    service = Service(_settings(tmp_path, redis_enabled=False))
    payload = ForecastPayload(
        issued_at="2026-07-23T12:00:00",
        horizon_h=2,
        model_version="chronos-test",
        forecast={"pv_avg": [10.0, 20.0], "demand": [30.0, 40.0]},
    )

    service._append_log(payload)
    service._append_log(payload)

    path = tmp_path / "forecasts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["forecast"]["pv_avg"] == [10.0, 20.0]
    assert rows[0]["forecast"]["demand"] == [30.0, 40.0]


def test_historical_request_cuts_context_and_starts_after_issue_hour(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr("microgrid_forecaster.service.runner.Predictor", _FakePredictor)
    service = Service(_settings(tmp_path, redis_enabled=False))
    datetimes = pd.date_range("2026-01-15 00:00:00", periods=6, freq="h")
    series_index = pd.MultiIndex.from_product(
        [["pv_avg"], datetimes],
        names=["series_id", "datetime"],
    )
    series = pd.DataFrame({"value": range(6)}, index=series_index)
    exog = pd.DataFrame({"weather": range(6)}, index=datetimes)
    monkeypatch.setattr(
        "microgrid_forecaster.service.runner.dataset.prepare",
        lambda _cfg: (series, exog),
    )
    weather = pd.DataFrame(
        {"weather": range(8)},
        index=pd.date_range("2026-01-15 00:00:00", periods=8, freq="h"),
    )
    monkeypatch.setattr(
        "microgrid_forecaster.service.runner.loaders.load_ecmwf",
        lambda _path: weather,
    )

    history, history_exog = service._history(pd.Timestamp("2026-01-15 02:15:00"))
    future = service._exog_future(pd.Timestamp("2026-01-15 02:15:00"), 3)

    assert history.index.get_level_values("datetime").max() == pd.Timestamp(
        "2026-01-15 02:00:00"
    )
    assert history_exog.index.max() == pd.Timestamp("2026-01-15 02:00:00")
    assert list(future.index) == list(
        pd.date_range("2026-01-15 03:00:00", periods=3, freq="h")
    )


def test_pymgrid_context_never_loads_campus_target_history(monkeypatch, tmp_path) -> None:
    class ContextPredictor:
        def __init__(self, _cfg: Settings):
            self.calls: list[tuple] = []

        def predict(self, *args, **kwargs) -> ForecastPayload:
            self.calls.append((args, kwargs))
            return ForecastPayload(
                issued_at=args[0].isoformat(),
                horizon_h=kwargs["horizon_h"],
                model_version="chronos-test",
                forecast={
                    "pv_avg": [10.0] * kwargs["horizon_h"],
                    "demand": [20.0] * kwargs["horizon_h"],
                },
                source_id=kwargs["source_id"],
                context_time=kwargs["context_time"],
                context_steps=kwargs["context_steps"],
                cold_start=kwargs["cold_start"],
                covariate_mode=kwargs["covariate_mode"],
            )

    monkeypatch.setattr("microgrid_forecaster.service.runner.Predictor", ContextPredictor)
    monkeypatch.setattr(
        "microgrid_forecaster.service.runner.dataset.prepare",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("campus targets must not be loaded")),
    )
    service = Service(_settings(tmp_path, redis_enabled=False))
    context = ForecastContext(
        source_id="pymgrid25-scenario-2",
        frequency_h=1.0,
        pv_kw=[10.0, 20.0],
        demand_kw=[30.0, 40.0],
    )

    payload = service.run_once(horizon_h=3, context=context)

    history = service.predictor.calls[0][0][1]
    assert history.loc[("pv_avg", slice(None)), "value"].tolist() == [10.0, 20.0]
    assert history.loc[("demand", slice(None)), "value"].tolist() == [30.0, 40.0]
    assert service.predictor.calls[0][0][2] is None
    assert service.predictor.calls[0][0][3] is None
    assert payload.source_id == "pymgrid25-scenario-2"
    assert payload.covariate_mode == "none"


def test_campus_context_uses_ecmwf_but_not_configured_target_history(monkeypatch, tmp_path) -> None:
    class ContextPredictor:
        def __init__(self, _cfg: Settings):
            self.calls: list[tuple] = []

        def predict(self, *args, **kwargs) -> ForecastPayload:
            self.calls.append((args, kwargs))
            return ForecastPayload(
                issued_at=args[0].isoformat(),
                horizon_h=kwargs["horizon_h"],
                model_version="chronos-test",
                forecast={
                    "pv_avg": [10.0] * kwargs["horizon_h"],
                    "demand": [20.0] * kwargs["horizon_h"],
                },
                source_id=kwargs["source_id"],
                covariate_mode=kwargs["covariate_mode"],
            )

    monkeypatch.setattr("microgrid_forecaster.service.runner.Predictor", ContextPredictor)
    monkeypatch.setattr(
        "microgrid_forecaster.service.runner.dataset.prepare",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("campus targets must not be loaded")),
    )
    weather = pd.DataFrame(
        {"weather": range(5)},
        index=pd.date_range("2026-01-15 00:00:00", periods=5, freq="h"),
    )
    monkeypatch.setattr(
        "microgrid_forecaster.service.runner.loaders.load_ecmwf", lambda _path: weather
    )
    service = Service(_settings(tmp_path, redis_enabled=False))
    payload = service.run_once(
        horizon_h=2,
        issued_at="2026-01-15T00:00:00",
        context=ForecastContext(
            source_id="campus-telemetry-2025-2026",
            frequency_h=1.0,
            pv_kw=[10.0],
            demand_kw=[20.0],
        ),
    )

    assert service.predictor.calls[0][0][2] is not None
    assert service.predictor.calls[0][0][3] is not None
    assert payload.covariate_mode == "ecmwf"
