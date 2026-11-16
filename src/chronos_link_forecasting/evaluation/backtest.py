"""Rolling-origin backtest that only ever hands a model its past."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from chronos_link_forecasting.models.protocol import ProbabilisticForecaster


@dataclass(frozen=True)
class BacktestResult:
    actual: pd.DataFrame
    predictions: pd.DataFrame
    origins: tuple[pd.Timestamp, ...]


def rolling_origin_backtest(
    frame: pd.DataFrame,
    forecaster: ProbabilisticForecaster,
    *,
    prediction_length: int,
    quantiles: Sequence[float],
    first_origin: pd.Timestamp,
    stride: int,
    context_length: int,
) -> BacktestResult:
    """Forecast from every ``stride``-th origin at or after ``first_origin``.

    For an origin at row ``i`` the model receives a copy of rows
    ``[i - context_length, i)`` and nothing else. The next ``prediction_length``
    rows are held back as the actuals. Every window uses the same context length,
    and windows that would run past the data are skipped. Each window is scored
    under its own ``item_id`` so overlapping windows keep unique keys.
    """
    for name, value in (
        ("prediction_length", prediction_length),
        ("stride", stride),
        ("context_length", context_length),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    missing = {"item_id", "timestamp", "target"}.difference(frame.columns)
    if missing:
        raise ValueError(f"frame must include columns: {', '.join(sorted(missing))}")
    if frame["item_id"].nunique() != 1:
        raise ValueError("frame must contain exactly one item_id")
    data = frame.loc[:, ["item_id", "timestamp", "target"]].reset_index(drop=True)
    timestamps = data["timestamp"]
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("frame timestamps must be strictly increasing")

    eligible = timestamps.index[timestamps >= first_origin]
    if len(eligible) == 0:
        raise ValueError("first_origin is after the last row")
    start = int(eligible[0])
    if start < context_length:
        raise ValueError("first_origin leaves less than context_length rows of history")

    item = str(data["item_id"].iloc[0])
    actual_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    origins: list[pd.Timestamp] = []
    for position in range(start, len(data) - prediction_length + 1, stride):
        origin = pd.Timestamp(timestamps.iloc[position])
        context = data.iloc[position - context_length : position].copy(deep=True)
        future = data.iloc[position : position + prediction_length]
        if context["timestamp"].iloc[-1] >= origin:
            raise AssertionError("context reached the forecast origin")

        forecast = forecaster.predict(context, prediction_length, quantiles)
        if list(forecast["timestamp"]) != list(future["timestamp"]):
            raise ValueError(
                "forecast timestamps must equal the next prediction_length rows"
            )

        window_id = f"{item}@{origin.isoformat()}"
        window_actual = future.copy(deep=True)
        window_actual["item_id"] = window_id
        window_forecast = forecast.copy(deep=True)
        window_forecast["item_id"] = window_id
        window_forecast["timestamp"] = future["timestamp"].to_numpy()
        actual_parts.append(window_actual)
        prediction_parts.append(window_forecast)
        origins.append(origin)

    if not origins:
        raise ValueError("no complete forecast window fits after first_origin")
    return BacktestResult(
        actual=pd.concat(actual_parts, ignore_index=True),
        predictions=pd.concat(prediction_parts, ignore_index=True),
        origins=tuple(origins),
    )
