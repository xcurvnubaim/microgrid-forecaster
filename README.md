# microgrid-forecaster

Live configurable-horizon, hourly PV and demand forecaster for the campus microgrid.
Forecasts **pv_avg** and **demand** together using ECMWF weather as exogenous
features and serves them over **HTTP**. Optional
Redis publishing (`forecaster:load`) is disabled by default.

The model is **Chronos-2**, a pre-trained time-series foundation model. It is
**zero-shot**: it is never trained on our data and there is no weights artifact to
build. We feed it aligned PV/demand history plus the weather forecast at inference time.

This is the productionized form of `notebook/pv_forecast_ecmwf.ipynb`.

## Stack

`skforecast` foundation API + `chronos-forecasting` (Chronos-2, zero-shot) ·
`fastapi` + `uvicorn` (API) · `apscheduler` (hourly trigger) · optional `redis` publish ·
`typer` (CLI) · managed by **uv**.

## Two modes

- **evaluate** (offline, optional) — hold out the tail of the historical CSV,
  forecast it, and report accuracy (MAE / RMSE / MAPE / R²). Use it to decide if
  Chronos-2 is good enough and to pick `context_length`. Nothing is trained or
  saved except a metrics report.
- **production** — load the pretrained model once, then feed the latest history
  + ECMWF forecast and publish the configured horizon (24h by default).

## Quickstart

```bash
# 1. install core deps (uv creates .venv and resolves from pyproject)
uv sync

# 2. install the Chronos backend (heavy: torch + model weights)
uv sync --extra foundation

# 3. evaluate accuracy on the historical CSV holdout
uv run forecaster evaluate --config configs/forecaster.yaml

# 4a. production one-shot forecast as JSON
uv run forecaster predict --config configs/forecaster.yaml

# override the configured horizon for this request
uv run forecaster predict --config configs/forecaster.yaml --horizon 48

# 4b. run the HTTP API + scheduler on :8000 (Redis is not required)
uv run forecaster serve --config configs/forecaster.yaml

# or the whole thing in containers (forecaster + redis)
docker compose -f deployments/docker-compose.yaml up --build
```

## Interfaces

- **HTTP**
  - `GET /health` — status + model version + last run
  - `GET /forecast/latest` — latest payload generated through the HTTP endpoint
  - `POST /forecast` — force a fresh forecast; optional body:
    `{"horizon_h": 48, "issued_at": "2026-01-15T00:15:00", "context": {"source_id": "campus-telemetry-2025-2026", "frequency_h": 1, "pv_kw": [..], "demand_kw": [..]}}` (1–168 hours)
- **Redis (optional)** `forecaster:load` — when `redis.enabled: true`, publish
  each JSON payload and store the `…:latest` key.

Responses include the hourly timestamps, per-series units, model version, and
the horizon-specific values. A request without a body uses `model.horizon_h`
from the YAML and wall-clock time. Supplying `issued_at` anchors a historical
simulator step: target history is cut at that timestamp and the first forecast
point is the following hourly interval.

Simulator requests must include scenario-aligned `context` in kW. The service
uses that context rather than loading its configured campus PV/demand files,
echoes its `source_id`, and reports context count/time, cold-start status, and
whether the response used ECMWF (`ecmwf`) or no exogenous covariates (`none`).
Campus scenarios use configured ECMWF; external benchmarks whose source id
starts with `pymgrid` use context-only Chronos inference.

The default YAML is HTTP-only, so `forecaster serve` does not require a Redis
process. The Docker Compose deployment explicitly enables Redis to preserve the
multi-service deployment option.

Forecast audit logging is also disabled by default. Set `forecast_log` to a
`.jsonl` path to enable a dependency-free append-only log.

## Layout

```
src/microgrid_forecaster/
  config.py            typed settings (yaml + env); mode = evaluate | production
  data/                loaders + preprocess (hourly, impute, ECMWF exog) + dataset
  model/               builder (Chronos-2), evaluate (holdout metrics), postprocess
  service/             predict (zero-shot), publisher, api, runner (scheduler), schemas
  cli.py               evaluate | serve | predict
tests/                 preprocess, publisher (fakeredis), api (TestClient)
deployments/           Dockerfile + docker-compose
```

## Dev

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run mypy src
```
