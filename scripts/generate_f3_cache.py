#!/usr/bin/env python3
"""Generate the leakage-safe F3 15-minute corrected-context cache."""

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

LOGGER = logging.getLogger("generate_f3_cache")
TARGET_FREQUENCY_H = 0.25
HORIZON_HOURS = 24
FORECAST_STEPS = 96
PV_CONTEXT_STEPS = 8192
DEMAND_CONTEXT_STEPS = 500
PROCESSED_PV = Path(
    "/home/xcurv/teep-taiwan/data/processed/pv_15min_chronos_reconstruction_candidate.csv"
)
PROCESSED_DEMAND = Path(
    "/home/xcurv/teep-taiwan/data/processed/demand_15min_weekly_seasonal_reconstruction_candidate.csv"
)
SPLITS = {
    "train": {"start": "2025-12-02 00:00:00", "end": "2026-01-29 23:45:00"},
    "val": {"start": "2026-02-06 00:00:00", "end": "2026-02-24 23:45:00"},
    "test": {"start": "2026-03-01 00:00:00", "end": "2026-03-31 23:45:00"},
}
EXCLUDE_GAPS = ["2026-02-04"]


def _load_series(
    pv_path: Path = PROCESSED_PV, demand_path: Path = PROCESSED_DEMAND
) -> tuple[pd.Series, pd.Series]:
    pv = pd.read_csv(pv_path, parse_dates=["timestamp"])
    demand = pd.read_csv(demand_path, parse_dates=["timestamp"])
    return (
        pd.Series(pv["pv_kw"].to_numpy(), index=pv["timestamp"]).sort_index(),
        pd.Series(demand["demand_kw"].to_numpy(), index=demand["timestamp"]).sort_index(),
    )


def _issue_times(split: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(SPLITS[split]["start"])
    end = pd.Timestamp(SPLITS[split]["end"])
    index = pd.date_range(start - pd.Timedelta(minutes=15), end, freq="15min")
    if split == "val":
        index = index[
            ~index.to_series().dt.strftime("%Y-%m-%d").isin(EXCLUDE_GAPS).to_numpy()
        ]
    return index


def _context(series: pd.Series, issued_at: pd.Timestamp, steps: int) -> list[float]:
    values = series.loc[:issued_at].tail(steps)
    return [float(max(0.0, value)) for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(SPLITS), default="train")
    parser.add_argument("--config", default="configs/forecaster-f3-15min-corrected.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--source-id", default="islanded_72h_f3_15min_corrected")
    parser.add_argument("--pv-path", type=Path, default=PROCESSED_PV)
    parser.add_argument("--demand-path", type=Path, default=PROCESSED_DEMAND)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Settings.from_yaml(args.config)
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        "/home/xcurv/teep-taiwan/microgrid-simulator/data/f3"
    ) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "forecasts.jsonl"
    manifest_path = out_dir / "forecast_manifest.json"
    pv, demand = _load_series(args.pv_path, args.demand_path)
    service = Service(cfg)
    existing = set()
    if args.resume and jsonl_path.exists():
        existing = {
            json.loads(line)["issued_at"]
            for line in jsonl_path.read_text().splitlines()
            if line.strip()
        }
    written = 0
    elapsed = 0.0
    with jsonl_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for index, issued_at in enumerate(_issue_times(args.split)):
            if args.limit is not None and index >= args.limit:
                break
            issued_iso = issued_at.isoformat()
            if issued_iso in existing:
                continue
            pv_context = _context(pv, issued_at, PV_CONTEXT_STEPS)
            demand_context = _context(demand, issued_at, DEMAND_CONTEXT_STEPS)
            context = ForecastContext(
                source_id=args.source_id,
                frequency_h=TARGET_FREQUENCY_H,
                pv_kw=pv_context,
                demand_kw=demand_context,
            )
            started = time.perf_counter()
            payload = service.run_once(
                horizon_h=HORIZON_HOURS,
                target_frequency_h=TARGET_FREQUENCY_H,
                issued_at=issued_iso,
                context=context,
            )
            elapsed += time.perf_counter() - started
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
                "pv_values_kw": [max(0.0, float(v)) for v in payload.forecast["pv_avg"]],
                "demand_values_kw": [
                    max(0.0, float(v)) for v in payload.forecast["demand"]
                ],
                "source_id": args.source_id,
                "context_time": issued_iso,
                "context_steps": max(len(pv_context), len(demand_context)),
                "pv_context_steps": len(pv_context),
                "demand_context_steps": len(demand_context),
                "cold_start": len(pv_context) < 96 or len(demand_context) < 96,
                "covariate_mode": payload.covariate_mode,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            written += 1
            if written % 20 == 0:
                LOGGER.info("wrote %d records (%.2fs/record)", written, elapsed / written)

    total_records = sum(1 for line in jsonl_path.read_text().splitlines() if line.strip())
    manifest_path.write_text(
        json.dumps(
            {
                "cache_version": "1.1-f3",
                "source_id": args.source_id,
                "mode": "http",
                "total_records": total_records,
                "horizon_hours": HORIZON_HOURS,
                "frequency_hours": TARGET_FREQUENCY_H,
                "issue_frequency_hours": TARGET_FREQUENCY_H,
                "forecast_steps": FORECAST_STEPS,
                "target_units": "kw",
                "pv_model": "Chronos-2 Foundation Model",
                "demand_model": "Chronos-2 Foundation Model",
                "pv_context_steps": PV_CONTEXT_STEPS,
                "demand_context_steps": DEMAND_CONTEXT_STEPS,
                "covariate_mode": "ecmwf",
                "causal_preprocessing": True,
                "splits": {
                    args.split: {
                        "start": SPLITS[args.split]["start"],
                        "end": SPLITS[args.split]["end"],
                        "exclude_gaps": EXCLUDE_GAPS if args.split == "val" else [],
                    }
                },
            },
            indent=2,
        )
    )
    LOGGER.info("wrote %d new records; %d total records", written, total_records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
