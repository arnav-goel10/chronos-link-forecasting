from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import pandas as pd
from pandas.api.types import is_numeric_dtype

_KEY_COLUMNS = ("item_id", "timestamp")
_SUMMARY_QUANTILES = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class MetricBundle:
    weighted_quantile_loss: float
    p50_mae: float
    coverage: float
    interval_width: float


def evaluate_quantiles(
    actual: pd.DataFrame,
    predictions: pd.DataFrame,
    quantiles: Sequence[float],
) -> MetricBundle:
    """Evaluate key-aligned probabilistic forecasts.

    Weighted quantile loss is ``2 * sum(pinball) / (Q * sum(abs(actual)))``.
    If complete WQL rows exist but their actual-value denominator is zero, the
    function emits ``RuntimeWarning`` and returns NaN for weighted quantile loss.
    P50 MAE, P10-P90 coverage, and P10-P90 width share one complete-row mask.
    """
    levels = _validate_quantiles(quantiles)
    _validate_schema(actual, predictions, levels)
    aligned = _align(actual, predictions, levels)
    _validate_values(aligned, levels)
    _validate_monotonic_quantiles(aligned, levels)

    summary_complete = (
        aligned.loc[:, ["target", *_SUMMARY_QUANTILES]].notna().all(axis=1)
    )
    summary = aligned.loc[summary_complete]
    if summary.empty:
        p50_mae = math.nan
        coverage = math.nan
        interval_width = math.nan
    else:
        p50_mae = float((summary["target"] - summary[0.5]).abs().mean())
        coverage = float(
            (
                (summary[0.1] <= summary["target"])
                & (summary["target"] <= summary[0.9])
            ).mean()
        )
        interval_width = float((summary[0.9] - summary[0.1]).mean())

    wql_complete = aligned.loc[:, ["target", *levels]].notna().all(axis=1)
    wql_rows = aligned.loc[wql_complete]
    weighted_quantile_loss = _weighted_quantile_loss(wql_rows, levels)
    return MetricBundle(
        weighted_quantile_loss=weighted_quantile_loss,
        p50_mae=p50_mae,
        coverage=coverage,
        interval_width=interval_width,
    )


def _weighted_quantile_loss(aligned: pd.DataFrame, levels: tuple[float, ...]) -> float:
    if aligned.empty:
        return math.nan
    denominator = float(aligned["target"].abs().sum())
    if denominator == 0:
        warnings.warn(
            "weighted quantile loss is NaN because the sum of absolute actual "
            "values is zero",
            RuntimeWarning,
            stacklevel=2,
        )
        return math.nan

    total_pinball = 0.0
    for level in levels:
        errors = aligned["target"] - aligned[level]
        losses = pd.concat(
            (level * errors, (level - 1.0) * errors), axis="columns"
        ).max(axis="columns")
        total_pinball += float(losses.sum())
    return 2.0 * total_pinball / (len(levels) * denominator)


def _validate_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    try:
        levels = tuple(float(level) for level in quantiles)
    except (TypeError, ValueError) as error:
        raise ValueError("quantiles must be finite numbers") from error
    if not levels or any(not math.isfinite(level) for level in levels):
        raise ValueError("quantiles must be finite numbers")
    if any(level <= 0 or level >= 1 for level in levels):
        raise ValueError("quantiles must be between 0 and 1")
    if len(set(levels)) != len(levels):
        raise ValueError("quantiles must be unique")
    if any(left >= right for left, right in pairwise(levels)):
        raise ValueError("quantiles must be strictly increasing")
    if not set(_SUMMARY_QUANTILES).issubset(levels):
        raise ValueError("quantiles must include 0.1, 0.5, and 0.9")
    return levels


def _validate_schema(
    actual: pd.DataFrame,
    predictions: pd.DataFrame,
    levels: tuple[float, ...],
) -> None:
    missing_actual = set((*_KEY_COLUMNS, "target")).difference(actual.columns)
    if missing_actual:
        raise ValueError(
            f"actual must include columns: {', '.join(sorted(missing_actual))}"
        )
    missing_predictions = set((*_KEY_COLUMNS, *levels)).difference(predictions.columns)
    if missing_predictions:
        missing = ", ".join(sorted(str(column) for column in missing_predictions))
        raise ValueError(f"predictions must include columns: {missing}")
    for frame_name, frame in (("actual", actual), ("predictions", predictions)):
        if frame.duplicated(subset=list(_KEY_COLUMNS)).any():
            raise ValueError(f"{frame_name} contains duplicate item_id/timestamp keys")


def _align(
    actual: pd.DataFrame,
    predictions: pd.DataFrame,
    levels: tuple[float, ...],
) -> pd.DataFrame:
    actual_values = actual.loc[:, [*_KEY_COLUMNS, "target"]].copy(deep=True)
    prediction_values = predictions.loc[:, [*_KEY_COLUMNS, *levels]].copy(deep=True)
    aligned = actual_values.merge(
        prediction_values,
        on=list(_KEY_COLUMNS),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(aligned) != len(actual_values) or not aligned["_merge"].eq("both").all():
        raise ValueError(
            "actual and predictions must have the same item_id/timestamp keys"
        )
    return aligned.drop(columns="_merge")


def _validate_values(aligned: pd.DataFrame, levels: tuple[float, ...]) -> None:
    for column in ("target", *levels):
        values = aligned[column]
        if not is_numeric_dtype(values.dtype) or values.dtype == bool:
            raise ValueError(f"{column} values must be numeric")
        nonmissing = values.dropna()
        if not nonmissing.map(lambda value: math.isfinite(float(value))).all():
            raise ValueError(f"{column} values must be finite or missing")


def _validate_monotonic_quantiles(
    aligned: pd.DataFrame, levels: tuple[float, ...]
) -> None:
    complete = aligned.loc[:, list(levels)].notna().all(axis=1)
    values = aligned.loc[complete, list(levels)]
    for lower, upper in pairwise(levels):
        if (values[lower] > values[upper]).any():
            raise ValueError("prediction quantiles must be monotonic at every row")
