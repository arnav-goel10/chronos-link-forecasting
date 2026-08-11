from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

_KEY_COLUMNS = ("item_id", "timestamp")


class LastValueBaseline:
    """Last-value forecast with context-only first-difference uncertainty."""

    def predict(
        self,
        context: pd.DataFrame,
        prediction_length: int,
        quantiles: Sequence[float],
    ) -> pd.DataFrame:
        validated = _validate_context(context, prediction_length, quantiles)
        if len(validated) < 2:
            raise ValueError(
                "last-value context must contain at least two rows for residuals"
            )

        residuals = validated["target"].diff().dropna()
        medians = [float(validated["target"].iloc[-1])] * prediction_length
        return _prediction_frame(
            validated, medians, residuals, prediction_length, quantiles
        )


class SeasonalNaiveBaseline:
    """Seasonal-naive forecast with context-only season-lag uncertainty."""

    def __init__(self, season_length: int) -> None:
        if (
            isinstance(season_length, bool)
            or not isinstance(season_length, int)
            or season_length < 1
        ):
            raise ValueError("season_length must be a positive integer")
        self._season_length = season_length

    def predict(
        self,
        context: pd.DataFrame,
        prediction_length: int,
        quantiles: Sequence[float],
    ) -> pd.DataFrame:
        if len(context) <= self._season_length:
            raise ValueError(
                "seasonal context must contain more than season_length rows"
            )
        validated = _validate_context(context, prediction_length, quantiles)

        residuals = (
            validated["target"] - validated["target"].shift(self._season_length)
        ).dropna()
        last_season = validated["target"].iloc[-self._season_length :].tolist()
        medians = [
            float(last_season[step % self._season_length])
            for step in range(prediction_length)
        ]
        return _prediction_frame(
            validated, medians, residuals, prediction_length, quantiles
        )


def _validate_context(
    context: pd.DataFrame,
    prediction_length: int,
    quantiles: Sequence[float],
) -> pd.DataFrame:
    if (
        isinstance(prediction_length, bool)
        or not isinstance(prediction_length, int)
        or prediction_length < 1
    ):
        raise ValueError("prediction_length must be a positive integer")
    _validate_quantiles(quantiles)

    missing = set((*_KEY_COLUMNS, "target")).difference(context.columns)
    if missing:
        raise ValueError(f"context must include columns: {', '.join(sorted(missing))}")
    if context.empty:
        raise ValueError("context must not be empty")
    if context["item_id"].isna().any() or context["item_id"].nunique() != 1:
        raise ValueError("context must contain exactly one item_id")
    if not is_datetime64_any_dtype(context["timestamp"].dtype):
        raise ValueError("context timestamp must be datetime-like")
    if context["timestamp"].isna().any():
        raise ValueError("context timestamp must not contain missing values")
    if not is_numeric_dtype(context["target"].dtype) or context["target"].dtype == bool:
        raise ValueError("context target must be numeric")
    finite_target = context["target"].map(lambda value: math.isfinite(float(value)))
    if not finite_target.all():
        raise ValueError("context target values must be finite")

    validated = context.loc[:, [*_KEY_COLUMNS, "target"]].copy(deep=True)
    timestamps = validated["timestamp"]
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError(
            "context timestamps must have a regular, strictly increasing frequency"
        )
    _infer_frequency(timestamps)
    return validated


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
    return levels


def _infer_frequency(timestamps: pd.Series) -> str | pd.Timedelta:
    if len(timestamps) < 3:
        raise ValueError(
            "context timestamps must have a regular, strictly increasing frequency"
        )
    inferred = pd.infer_freq(pd.DatetimeIndex(timestamps))
    if inferred is not None:
        return inferred
    differences = timestamps.diff().dropna()
    first_difference = differences.iloc[0]
    if (
        first_difference <= pd.Timedelta(0)
        or not differences.eq(first_difference).all()
    ):
        raise ValueError(
            "context timestamps must have a regular, strictly increasing frequency"
        )
    return pd.Timedelta(first_difference)


def _prediction_frame(
    context: pd.DataFrame,
    medians: list[float],
    residuals: pd.Series,
    prediction_length: int,
    quantiles: Sequence[float],
) -> pd.DataFrame:
    levels = _validate_quantiles(quantiles)
    residual_median = float(residuals.quantile(0.5))
    offsets = [float(residuals.quantile(level)) - residual_median for level in levels]
    frequency = _infer_frequency(context["timestamp"])
    forecast_timestamps = pd.date_range(
        start=context["timestamp"].iloc[-1],
        periods=prediction_length + 1,
        freq=frequency,
    )[1:]

    rows: list[dict[str | float, object]] = []
    item_id = context["item_id"].iloc[-1]
    for timestamp, median in zip(forecast_timestamps, medians, strict=True):
        ordered_values = sorted(median + offset for offset in offsets)
        row: dict[str | float, object] = {
            "item_id": item_id,
            "timestamp": timestamp,
        }
        row.update(dict(zip(levels, ordered_values, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows, columns=[*_KEY_COLUMNS, *levels])
