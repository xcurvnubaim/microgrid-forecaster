"""API contract: health works before any run; POST returns a payload."""

from fastapi.testclient import TestClient

from microgrid_forecaster.service.api import create_app
from microgrid_forecaster.service.schemas import ForecastPayload


def _stub_payload(horizon_h=None, issued_at=None, context=None, target_frequency_h=None):
    horizon = horizon_h or 24
    return ForecastPayload(
        issued_at=issued_at or "2026-06-30T13:00:00",
        horizon_h=horizon,
        frequency_h=1.0,
        model_version="v1",
        timestamps=[f"2026-06-30T{hour:02d}:00:00" for hour in range(horizon)],
        units={"pv_avg": "kw", "demand": "kw"},
        forecast={
            "pv_avg": [1.0] * horizon,
            "demand": [2.0] * horizon,
        },
        source_id=context.source_id if context else "campus-telemetry-2025-2026",
        context_steps=len(context.pv_kw) if context else 0,
        covariate_mode="none" if context else "ecmwf",
    )


def test_health_before_first_run():
    app = create_app(run_once=_stub_payload)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["last_run"] is None


def test_post_forecast_updates_latest():
    app = create_app(run_once=_stub_payload)
    client = TestClient(app)
    assert client.post("/forecast").status_code == 200
    assert client.get("/forecast/latest").json()["horizon_h"] == 24


def test_post_forecast_accepts_horizon_override():
    app = create_app(run_once=_stub_payload)
    client = TestClient(app)

    response = client.post("/forecast", json={"horizon_h": 48})

    assert response.status_code == 200
    assert response.json()["horizon_h"] == 48
    assert len(response.json()["forecast"]["pv_avg"]) == 48
    assert len(response.json()["forecast"]["demand"]) == 48


def test_post_forecast_accepts_simulator_step_timestamp():
    app = create_app(run_once=_stub_payload)
    client = TestClient(app)

    response = client.post(
        "/forecast",
        json={"horizon_h": 4, "issued_at": "2026-01-15T00:15:00"},
    )

    assert response.status_code == 200
    assert response.json()["issued_at"] == "2026-01-15T00:15:00"


def test_post_forecast_accepts_scenario_aligned_context():
    app = create_app(run_once=_stub_payload)
    client = TestClient(app)

    response = client.post(
        "/forecast",
        json={
            "horizon_h": 3,
            "context": {
                "source_id": "pymgrid25-scenario-2",
                "frequency_h": 1,
                "pv_kw": [10, 12],
                "demand_kw": [20, 21],
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["source_id"] == "pymgrid25-scenario-2"
    assert response.json()["context_steps"] == 2
    assert response.json()["covariate_mode"] == "none"


def test_post_forecast_accepts_target_specific_context_lengths() -> None:
    app = create_app(run_once=_stub_payload)
    client = TestClient(app)

    response = client.post(
        "/forecast",
        json={
            "horizon_h": 3,
            "context": {
                "source_id": "pymgrid25-scenario-2",
                "frequency_h": 1,
                "pv_kw": [10],
                "demand_kw": [20, 21],
            },
        },
    )

    assert response.status_code == 200


def test_post_forecast_rejects_out_of_range_horizon():
    app = create_app(run_once=_stub_payload)
    client = TestClient(app)

    assert client.post("/forecast", json={"horizon_h": 0}).status_code == 422
    assert client.post("/forecast", json={"horizon_h": 169}).status_code == 422


def test_post_forecast_returns_clear_client_error_for_invalid_history_timestamp():
    def no_history(*_args, **_kwargs):
        raise ValueError("no forecast history is available at or before 2000-01-02")

    app = create_app(run_once=no_history)
    client = TestClient(app)

    response = client.post(
        "/forecast",
        json={"horizon_h": 24, "issued_at": "2000-01-02T21:00:00"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "no forecast history is available at or before 2000-01-02"
