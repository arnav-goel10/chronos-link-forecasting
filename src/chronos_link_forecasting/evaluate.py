"""Run the rolling-origin backtest and write a results JSON.

    python -m chronos_link_forecasting.evaluate \\
        --data artifacts/link_usd_1h.parquet --out results/link_usd_1h_backtest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from chronos_link_forecasting.evaluation.backtest import rolling_origin_backtest
from chronos_link_forecasting.evaluation.metrics import evaluate_quantiles
from chronos_link_forecasting.models.baselines import (
    LastValueBaseline,
    SeasonalNaiveBaseline,
)
from chronos_link_forecasting.models.chronos import (
    DEFAULT_CHRONOS_MODEL,
    ChronosForecaster,
)
from chronos_link_forecasting.models.protocol import ProbabilisticForecaster

SCHEMA_VERSION = 1
QUANTILES = (0.1, 0.5, 0.9)
MODEL_NAMES = ("last-value", "seasonal-naive", "chronos")
METRIC_KEYS = (
    "weighted_quantile_loss",
    "p50_mae",
    "coverage_80",
    "interval_width_80",
)


def build_forecaster(
    name: str, season_length: int, chronos_model: str
) -> tuple[ProbabilisticForecaster, str]:
    if name == "last-value":
        return LastValueBaseline(), "Last value"
    if name == "seasonal-naive":
        return (
            SeasonalNaiveBaseline(season_length),
            f"Seasonal naive ({season_length}h)",
        )
    if name == "chronos":
        return (
            ChronosForecaster(model_id=chronos_model),
            f"Chronos zero-shot ({chronos_model})",
        )
    raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODEL_NAMES)}")


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _frame_sha256(frame: pd.DataFrame) -> str:
    payload = frame.loc[:, ["timestamp", "target"]].to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_evaluation(
    frame: pd.DataFrame,
    data_meta: dict[str, Any],
    *,
    models: Sequence[str],
    prediction_length: int,
    stride: int,
    context_length: int,
    test_hours: int,
    season_length: int,
    chronos_model: str = DEFAULT_CHRONOS_MODEL,
) -> dict[str, Any]:
    if test_hours < prediction_length or test_hours >= len(frame):
        raise ValueError("test_hours must fit inside the data and cover one horizon")
    first_origin = pd.Timestamp(frame["timestamp"].iloc[len(frame) - test_hours])

    results: dict[str, Any] = {}
    origins: tuple[pd.Timestamp, ...] = ()
    scored_points = 0
    for name in models:
        forecaster, label = build_forecaster(name, season_length, chronos_model)
        started = time.perf_counter()
        backtest = rolling_origin_backtest(
            frame,
            forecaster,
            prediction_length=prediction_length,
            quantiles=QUANTILES,
            first_origin=first_origin,
            stride=stride,
            context_length=context_length,
        )
        bundle = evaluate_quantiles(backtest.actual, backtest.predictions, QUANTILES)
        elapsed = time.perf_counter() - started
        origins = backtest.origins
        scored_points = len(backtest.actual)
        results[name] = {
            "label": label,
            "weighted_quantile_loss": bundle.weighted_quantile_loss,
            "p50_mae": bundle.p50_mae,
            "coverage_80": bundle.coverage,
            "interval_width_80": bundle.interval_width,
            "runtime_seconds": round(elapsed, 2),
        }

    ranking = sorted(results, key=lambda key: results[key]["weighted_quantile_loss"])
    in_test = frame.loc[frame["timestamp"] >= first_origin]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "provenance": {
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "github_ref": os.environ.get("GITHUB_REF"),
            "github_workflow": os.environ.get("GITHUB_WORKFLOW"),
        },
        "data": {
            **data_meta,
            "first_timestamp": pd.Timestamp(frame["timestamp"].iloc[0]).isoformat(),
            "last_timestamp": pd.Timestamp(frame["timestamp"].iloc[-1]).isoformat(),
            "rows": len(frame),
            "sha256_timestamp_target_csv": _frame_sha256(frame),
        },
        "backtest": {
            "method": "rolling origin, fixed context, zero-shot (no fitting)",
            "prediction_length_hours": prediction_length,
            "stride_hours": stride,
            "context_length_hours": context_length,
            "test_hours": test_hours,
            "test_filled_hours": int(in_test["filled"].sum())
            if "filled" in in_test.columns
            else None,
            "season_length_hours": season_length,
            "quantile_levels": list(QUANTILES),
            "first_origin": origins[0].isoformat() if origins else None,
            "last_origin": origins[-1].isoformat() if origins else None,
            "windows": len(origins),
            "scored_points": scored_points,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": _version("pandas"),
            "torch": _version("torch"),
            "chronos_forecasting": _version("chronos-forecasting"),
            "chronos_checkpoint": chronos_model if "chronos" in models else None,
        },
        "models": results,
        "ranking_by_weighted_quantile_loss": ranking,
    }


def render_markdown_table(payload: dict[str, Any]) -> str:
    """Render the results table that the README embeds verbatim."""
    lines = [
        "| Model | Weighted quantile loss | P50 MAE (USD) "
        "| 80% interval coverage | Mean 80% interval width (USD) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in payload["ranking_by_weighted_quantile_loss"]:
        row = payload["models"][name]
        lines.append(
            f"| {row['label']} | {row['weighted_quantile_loss']:.4f} "
            f"| {row['p50_mae']:.4f} | {row['coverage_80'] * 100:.1f}% "
            f"| {row['interval_width_80']:.4f} |"
        )
    return "\n".join(lines)


def _finite(payload: dict[str, Any]) -> bool:
    return all(
        math.isfinite(float(row[key]))
        for row in payload["models"].values()
        for key in METRIC_KEYS
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the rolling-origin backtest and write a results JSON"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--meta", type=Path, help="defaults to the data path with .meta.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--test-hours", type=int, default=1440)
    parser.add_argument("--season-length", type=int, default=24)
    parser.add_argument("--chronos-model", default=DEFAULT_CHRONOS_MODEL)
    args = parser.parse_args(argv)

    frame = pd.read_parquet(args.data)
    meta_path = args.meta or args.data.with_suffix(".meta.json")
    data_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    models = [name.strip() for name in args.models.split(",") if name.strip()]

    payload = run_evaluation(
        frame,
        data_meta,
        models=models,
        prediction_length=args.prediction_length,
        stride=args.stride,
        context_length=args.context_length,
        test_hours=args.test_hours,
        season_length=args.season_length,
        chronos_model=args.chronos_model,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(render_markdown_table(payload))
    if not _finite(payload):
        print("non-finite metric in results", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
