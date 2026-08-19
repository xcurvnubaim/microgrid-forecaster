"""Per-request horizon reaches the foundation forecaster and wire payload."""

from __future__ import annotations

import pandas as pd

from microgrid_forecaster.config import DataPaths, Settings
from microgrid_forecaster.service.predict import Predictor


class _FakeForecaster:
    def __init__(self) -> None:
        self.steps: int | None = None
        self.exog_rows: int | None = None

    def fit(self, series: dict[str, pd.Series], exog: pd.DataFrame) -> None:
        assert set(series) == {"pv_avg", "demand"}
        assert not exog.empty

    def predict(self, steps: int, exog: pd.DataFrame) -> pd.DataFrame:
        self.steps = steps
        self.exog_rows = len(exog)
        return pd.DataFrame(
            {
                "pv_avg": range(steps),
                "demand": range(100, 100 + steps),
            },
            index=exog.index[:steps],
        )


def test_predict_uses_request_horizon_and_describes_payload(monkeypatch, tmp_path) -> None:
    fake = _FakeForecaster()
    monkeypatch.setattr(
        "microgrid_forecaster.service.predict.build_forecaster",
        lambda _cfg: fake,
    )
    settings = Settings(
        data=DataPaths(
            pv_xlsx=tmp_path / "pv.xlsx",
            ecmwf_csv=tmp_path / "weather.csv",
        )
    )
    predictor = Predictor(settings)
    index = pd.date_range("2026-07-23 12:00:00", periods=6, freq="h")
    history_index = pd.MultiIndex.from_product(
        [["pv_avg", "demand"], [index[0]]],
        names=["series_id", "datetime"],
    )
    history = pd.DataFrame({"value": [1.0, 2.0]}, index=history_index)
    exog_history = pd.DataFrame({"weather": [1.0]}, index=[index[0]])
    exog_future = pd.DataFrame({"weather": range(6)}, index=index)

    payload = predictor.predict(
        index[0],
        history,
        exog_history,
        exog_future,
        horizon_h=4,
    )

    assert fake.steps == 4
    assert fake.exog_rows == 4
    assert payload.horizon_h == 4
    assert payload.frequency_h == 1.0
    assert payload.units == {"pv_avg": "kw", "demand": "kw"}
    assert len(payload.timestamps) == 4
    assert payload.forecast["pv_avg"] == [0, 1, 2, 3]
    assert payload.forecast["demand"] == [100, 101, 102, 103]


class _FakeNoExogForecaster:
    def __init__(self) -> None:
        self.fit_exog = "not-called"
        self.predict_exog = "not-called"

    def fit(self, series: dict[str, pd.Series], **kwargs) -> None:
        assert set(series) == {"pv_avg", "demand"}
        self.fit_exog = kwargs.get("exog")

    def predict(self, steps: int, **kwargs) -> pd.DataFrame:
        self.predict_exog = kwargs.get("exog")
        return pd.DataFrame(
            {"pv_avg": range(steps), "demand": range(100, 100 + steps)},
            index=pd.date_range("2026-07-23 13:00:00", periods=steps, freq="h"),
        )


def test_predict_supports_context_only_chronos_without_weather(monkeypatch, tmp_path) -> None:
    fake = _FakeNoExogForecaster()
    monkeypatch.setattr(
        "microgrid_forecaster.service.predict.build_forecaster", lambda _cfg: fake
    )
    predictor = Predictor(
        Settings(data=DataPaths(pv_xlsx=tmp_path / "pv.xlsx", ecmwf_csv=tmp_path / "weather.csv"))
    )
    time = pd.Timestamp("2026-07-23 12:00:00")
    history_index = pd.MultiIndex.from_product(
        [["pv_avg", "demand"], [time]], names=["series_id", "datetime"]
    )
    history = pd.DataFrame({"value": [1.0, 2.0]}, index=history_index)

    payload = predictor.predict(
        time,
        history,
        exog_hist=None,
        exog_future=None,
        horizon_h=2,
        source_id="pymgrid25-scenario-2",
        context_steps=1,
        cold_start=True,
        covariate_mode="none",
    )

    assert fake.fit_exog is None
    assert fake.predict_exog is None
    assert payload.source_id == "pymgrid25-scenario-2"
    assert payload.covariate_mode == "none"


def test_predict_uses_separate_target_context_models(monkeypatch, tmp_path) -> None:
    created: list[int] = []

    class _TargetForecaster:
        def __init__(self, context_length: int) -> None:
            created.append(context_length)

        def fit(self, series, **kwargs) -> None:
            self.target = next(iter(series))

        def predict(self, steps: int, **kwargs) -> pd.DataFrame:
            target = self.target
            return pd.DataFrame(
                {target: range(steps)},
                index=pd.date_range("2026-07-23 13:00:00", periods=steps, freq="15min"),
            )

    monkeypatch.setattr(
        "microgrid_forecaster.service.predict.build_forecaster",
        lambda cfg: _TargetForecaster(cfg.context_length),
    )
    settings = Settings(
        data=DataPaths(pv_xlsx=tmp_path / "pv.xlsx", ecmwf_csv=tmp_path / "weather.csv"),
    )
    settings.model.pv_context_length = 8192
    settings.model.demand_context_length = 500
    predictor = Predictor(settings)
    time = pd.Timestamp("2026-07-23 12:00:00")
    index = pd.date_range(time, periods=4, freq="15min")
    history_index = pd.MultiIndex.from_product(
        [["pv_avg", "demand"], index], names=["series_id", "datetime"]
    )
    history = pd.DataFrame({"value": [1.0] * len(history_index)}, index=history_index)

    payload = predictor.predict(
        time,
        history,
        exog_hist=None,
        exog_future=None,
        horizon_h=24,
        target_frequency_h=0.25,
        source_id="islanded_72h_f3_15min_corrected",
        context_steps=4,
        covariate_mode="none",
    )

    assert sorted(created) == [500, 8192]
    assert len(payload.forecast["pv_avg"]) == 96
    assert len(payload.forecast["demand"]) == 96
    assert payload.pv_context_steps == 4
    assert payload.demand_context_steps == 4
