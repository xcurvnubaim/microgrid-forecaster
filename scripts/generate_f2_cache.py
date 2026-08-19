#!/usr/bin/env python3
"""Generate a leakage-safe F2 15-minute forecast cache for the simulator.

Direct 15-minute PV/demand forecasts over a 24-hour horizon (96 points) using
the existing interpolated shortwave-radiation trajectory as the exogenous
covariate (assumed-known weather). One snapshot is issued at every 15-minute
controller tick; context is the preceding quarter-hour history (up to 512 steps,
~5 days 8 hours) available at the issue time.

Outputs a JSONL cache plus a ``forecast_manifest.json`` in the format the
simulator's ``ForecastCache.load`` expects (split-specific, causal, provenance
labelled). Written for the ``microgrid-forecaster`` environment (needs Chronos).

Usage:
    uv run python scripts/generate_f2_cache.py --split train
    uv run python scripts/generate_f2_cache.py --split val
    uv run python scripts/generate_f2_cache.py --split train --limit 20 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from microgrid_forecaster.config import Settings
from microgrid_forecaster.service.runner import Service
from microgrid_forecaster.service.schemas import ForecastContext

LOGGER = logging.getLogger("generate_f2_cache")

TARGET_FREQUENCY_H = 0.25
HORIZON_HOURS = 24
FORECAST_STEPS = 96
CONTEXT_STEPS = 512

PROCESSED_PV = (
    "/home/xcurv/teep-taiwan/data/processed/pv_15min_chronos_reconstruction_candidate.csv"
)
PROCESSED_DEMAND = (
    "/home/xcurv/teep-taiwan/data/processed/demand_15min_weekly_seasonal_reconstruction_candidate.csv"
)

SPLITS = {
    "train": {"start": "2025-12-02 00:00:00", "end": "2026-01-29 23:45:00"},
    "val": {"start": "2026-02-06 00:00:00", "end": "2026-02-24 23:45:00"},
    "test": {"start": "2026-03-01 00:00:00", "end": "2026-03-31 23:45:00"},
}

EXCLUDE_GAPS = ["2026-02-04"]


def _load_series() -> tuple[pd.Series, pd.Series]:
    pv = pd.read_csv(PROCESSED_PV, parse_dates=["timestamp"])
    dem = pd.read_csv(PROCESSED_DEMAND, parse_dates=["timestamp"])
    pv_series = pd.Series(pv["pv_kw"].to_numpy(), index=pv["timestamp"]).sort_index()
    dem_series = pd.Series(
        dem["demand_kw"].to_numpy(), index=dem["timestamp"]
    ).sort_index()
    return pv_series, dem_series


def _issue_times(split: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(SPLITS[split]["start"])
    end = pd.Timestamp(SPLITS[split]["end"])
    # Include one preceding 15-minute issue so the first decision (whose
    # forecast context is the step before the first evaluated interval) is
    # served. The context sample before a split boundary is legitimate past data.
    lead = start - pd.Timedelta(minutes=15)
    idx = pd.date_range(lead, end, freq="15min")
    # Drop the excluded gap (2026-02-04) from validation issues.
    if split == "val":
        idx = idx[
            ~idx.to_series().dt.strftime("%Y-%m-%d").isin(EXCLUDE_GAPS).to_numpy()
        ]
    return idx


def _build_context(
    pv: pd.Series,
    dem: pd.Series,
    issued_at: pd.Timestamp,
) -> tuple[list[float], list[float], int]:
    """Quarter-hour context available at ``issued_at`` (inclusive), capped at 512."""
    pv_past = pv.loc[:issued_at].tail(CONTEXT_STEPS)
    dem_past = dem.loc[:issued_at].tail(CONTEXT_STEPS)
    pv_kw = [float(max(0.0, v)) for v in pv_past]
    dem_kw = [float(max(0.0, v)) for v in dem_past]
    return pv_kw, dem_kw, max(len(pv_kw), len(dem_kw))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
        help="which split to generate (default: train)",
    )
    parser.add_argument(
        "--config",
        default="configs/forecaster-f2-15min.yaml",
        help="forecaster configuration for the F2 model",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "output directory for forecasts.jsonl + manifest "
            "(default: ../microgrid-simulator/data/f2)"
        ),
    )
    parser.add_argument("--source-id", default=None, help="scenario source_id")
    parser.add_argument("--limit", type=int, default=None, help="stop after N issues (testing)")
    parser.add_argument(
        "--resume", action="store_true", help="skip already-generated issue times"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    cfg = Settings.from_yaml(args.config)
    split = args.split
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path("/home/xcurv/teep-taiwan/microgrid-simulator/data/f2") / split
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "forecasts.jsonl"
    manifest_path = out_dir / "forecast_manifest.json"
    source_id = args.source_id or cfg.data.radiation_csv and "islanded_72h_2026-01-15"

    svc = Service(cfg)
    pv, dem = _load_series()
    issues = _issue_times(split)

    existing: set[str] = set()
    if args.resume and jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            line = line.strip()
            if line:
                existing.add(json.loads(line)["issued_at"])

    LOGGER.info("F2 cache: split=%s issues=%d (resume=%s)", split, len(issues), args.resume)

    n_written = 0
    total_elapsed = 0.0
    handle = (
        jsonl_path.open("a", encoding="utf-8")
        if args.resume
        else jsonl_path.open("w", encoding="utf-8")
    )
    try:
        for i, issued_at in enumerate(issues):
            if args.limit is not None and i >= args.limit:
                break
            issued_iso = issued_at.isoformat()
            if issued_iso in existing:
                continue
            pv_kw, dem_kw, ctx_steps = _build_context(pv, dem, issued_at)
            if ctx_steps == 0:
                LOGGER.warning("no context at %s; skipping", issued_iso)
                continue
            context = ForecastContext(
                source_id=source_id,
                frequency_h=TARGET_FREQUENCY_H,
                pv_kw=pv_kw,
                demand_kw=dem_kw,
            )
            t0 = time.perf_counter()
            payload = svc.run_once(
                horizon_h=HORIZON_HOURS,
                target_frequency_h=TARGET_FREQUENCY_H,
                issued_at=issued_iso,
                context=context,
            )
            total_elapsed += time.perf_counter() - t0
            record = {
                "issued_at": issued_iso,
                "horizon_hours": HORIZON_HOURS,
                "frequency_hours": TARGET_FREQUENCY_H,
                "issue_frequency_hours": TARGET_FREQUENCY_H,
                "forecast_steps": FORECAST_STEPS,
                "target_units": "kw",
                "model_version": payload.model_version,
                "pv_target": "pv_avg",
                "demand_target": "demand",
                "timestamps": payload.timestamps,
                # Clamp PV to non-negative (matching the simulator's HTTP-client
                # boundary); demand is already non-negative.
                "pv_values_kw": [max(0.0, float(v)) for v in payload.forecast["pv_avg"]],
                "demand_values_kw": [max(0.0, float(v)) for v in payload.forecast["demand"]],
                "source_id": source_id,
                "context_time": issued_iso,
                "context_steps": ctx_steps,
                "cold_start": ctx_steps < 96,
                "covariate_mode": payload.covariate_mode,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            n_written += 1
            if n_written % 20 == 0:
                avg = total_elapsed / n_written
                LOGGER.info(
                    "wrote %d records (avg %.2fs/record)", n_written, avg
                )
    finally:
        handle.close()

    # Count total records (including resumed) for the manifest.
    total_records = 0
    if jsonl_path.exists():
        total_records = sum(1 for _ in jsonl_path.read_text().splitlines() if _.strip())

    manifest = {
        "cache_version": "1.0",
        "source_id": source_id,
        "mode": "http",
        "total_records": total_records,
        "horizon_hours": HORIZON_HOURS,
        "frequency_hours": TARGET_FREQUENCY_H,
        "issue_frequency_hours": TARGET_FREQUENCY_H,
        "forecast_steps": FORECAST_STEPS,
        "target_units": "kw",
        "pv_model": "Chronos-2 Foundation Model (shortwave exog)",
        "demand_model": "Chronos-2 Foundation Model (shortwave exog)",
        "causal_preprocessing": True,
        "splits": {
            split: {
                "start": SPLITS[split]["start"],
                "end": SPLITS[split]["end"],
                "exclude_gaps": EXCLUDE_GAPS if split == "val" else [],
            }
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    LOGGER.info(
        "done: wrote %d records to %s (%d total in file); avg %.2fs/record",
        n_written,
        jsonl_path,
        total_records,
        total_elapsed / max(1, n_written),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
