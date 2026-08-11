from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from chronos_link_forecasting.evaluation.metrics import evaluate_quantiles


def _actual(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["LINK"] * len(values),
            "timestamp": pd.date_range(
                "2026-01-01 00:00Z", periods=len(values), freq="h"
            ),
            "target": values,
        }
    )


def _predictions(
    low: list[float], median: list[float], high: list[float]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["LINK"] * len(median),
            "timestamp": pd.date_range(
                "2026-01-01 00:00Z", periods=len(median), freq="h"
            ),
            0.1: low,
            0.5: median,
            0.9: high,
        }
    )


def test_perfect_predictions_have_zero_mae_and_wql() -> None:
    actual = _actual([10.0, 20.0])
    predictions = _predictions([10.0, 20.0], [10.0, 20.0], [10.0, 20.0])

    metrics = evaluate_quantiles(actual, predictions, (0.1, 0.5, 0.9))

    assert metrics.p50_mae == 0.0
    assert metrics.weighted_quantile_loss == 0.0


def test_contained_targets_have_full_coverage_and_known_width() -> None:
    actual = _actual([2.0, 5.0])
    predictions = _predictions([1.0, 3.0], [2.0, 5.0], [3.0, 7.0])

    metrics = evaluate_quantiles(actual, predictions, (0.1, 0.5, 0.9))

    assert metrics.coverage == 1.0
    assert metrics.interval_width == 3.0


def test_wql_uses_documented_pinball_formula_and_denominator() -> None:
    actual = _actual([10.0])
    predictions = _predictions([8.0], [12.0], [15.0])

    metrics = evaluate_quantiles(actual, predictions, (0.1, 0.5, 0.9))

    # Errors actual - forecast are [2, -2, -5], so pinball losses are
    # [0.2, 1.0, 0.5]. WQL = 2 * 1.7 / (3 * 10).
    assert metrics.weighted_quantile_loss == pytest.approx(0.11333333333333333)
    assert metrics.p50_mae == 2.0
    assert metrics.coverage == 1.0
    assert metrics.interval_width == 7.0


def test_metrics_use_one_complete_row_mask_for_all_interval_summaries() -> None:
    actual = _actual([2.0, 100.0])
    predictions = _predictions([1.0, np.nan], [2.0, 999.0], [3.0, 1000.0])

    metrics = evaluate_quantiles(actual, predictions, (0.1, 0.5, 0.9))

    assert metrics.p50_mae == 0.0
    assert metrics.coverage == 1.0
    assert metrics.interval_width == 2.0


def test_metrics_align_rows_by_item_and_timestamp_not_position() -> None:
    actual = _actual([10.0, 20.0])
    predictions = _predictions([10.0, 20.0], [10.0, 20.0], [10.0, 20.0])
    predictions = predictions.iloc[::-1].reset_index(drop=True)

    metrics = evaluate_quantiles(actual, predictions, (0.1, 0.5, 0.9))

    assert metrics.p50_mae == 0.0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda frame: frame.iloc[:-1].copy(),
        lambda frame: frame.assign(item_id=["LINK", "ETH"]),
        lambda frame: frame.assign(
            timestamp=pd.to_datetime(["2026-01-01 00:00Z", "2026-01-02 00:00Z"])
        ),
    ],
)
def test_metrics_reject_length_key_or_timestamp_mismatch(mutate: object) -> None:
    actual = _actual([10.0, 20.0])
    predictions = _predictions([9.0, 19.0], [10.0, 20.0], [11.0, 21.0])
    changed = mutate(predictions)  # type: ignore[operator]

    with pytest.raises(ValueError, match="same item_id/timestamp keys"):
        evaluate_quantiles(actual, changed, (0.1, 0.5, 0.9))


def test_metrics_reject_duplicate_keys() -> None:
    actual = _actual([10.0, 20.0])
    predictions = _predictions([9.0, 19.0], [10.0, 20.0], [11.0, 21.0])
    predictions.loc[1, "timestamp"] = predictions.loc[0, "timestamp"]

    with pytest.raises(ValueError, match="duplicate item_id/timestamp"):
        evaluate_quantiles(actual, predictions, (0.1, 0.5, 0.9))


@pytest.mark.parametrize("frame_name", ["actual", "predictions"])
@pytest.mark.parametrize(
    ("invalid_key", "message"),
    [
        ("null_item_id", "item_id must not contain missing values"),
        ("nat_timestamp", "timestamp must not contain missing values"),
        ("string_timestamp", "timestamp must be datetime-like"),
    ],
)
def test_metrics_reject_invalid_alignment_keys(
    frame_name: str, invalid_key: str, message: str
) -> None:
    actual = _actual([10.0, 20.0])
    predictions = _predictions([9.0, 19.0], [10.0, 20.0], [11.0, 21.0])
    frame = actual if frame_name == "actual" else predictions
    if invalid_key == "null_item_id":
        frame.loc[0, "item_id"] = None
    elif invalid_key == "nat_timestamp":
        frame.loc[0, "timestamp"] = pd.NaT
    else:
        frame["timestamp"] = frame["timestamp"].astype(str)

    with pytest.raises(ValueError, match=rf"^{frame_name} {message}$"):
        evaluate_quantiles(actual, predictions, (0.1, 0.5, 0.9))


@pytest.mark.parametrize(
    ("quantiles", "message"),
    [
        ((0.1, 0.1, 0.9), "unique"),
        ((0.5, 0.1, 0.9), "strictly increasing"),
        ((0.0, 0.5, 0.9), "between 0 and 1"),
        ((0.1, float("nan"), 0.9), "finite"),
        ((0.2, 0.5, 0.8), "include 0.1, 0.5, and 0.9"),
    ],
)
def test_metrics_reject_invalid_quantiles(
    quantiles: tuple[float, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_quantiles(
            _actual([10.0]),
            _predictions([9.0], [10.0], [11.0]),
            quantiles,
        )


def test_metrics_reject_infinite_values() -> None:
    predictions = _predictions([9.0], [10.0], [float("inf")])

    with pytest.raises(ValueError, match="finite or missing"):
        evaluate_quantiles(_actual([10.0]), predictions, (0.1, 0.5, 0.9))


def test_metrics_reject_crossing_quantiles() -> None:
    predictions = _predictions([11.0], [10.0], [12.0])

    with pytest.raises(ValueError, match="monotonic"):
        evaluate_quantiles(_actual([10.0]), predictions, (0.1, 0.5, 0.9))


def test_wql_warns_and_returns_nan_when_actual_denominator_is_zero() -> None:
    actual = _actual([0.0, 0.0])
    predictions = _predictions([-1.0, -1.0], [0.0, 0.0], [1.0, 1.0])

    with pytest.warns(RuntimeWarning, match="sum of absolute actual values is zero"):
        metrics = evaluate_quantiles(actual, predictions, (0.1, 0.5, 0.9))

    assert math.isnan(metrics.weighted_quantile_loss)


def test_metrics_return_nan_summaries_when_no_complete_rows() -> None:
    actual = _actual([np.nan])
    predictions = _predictions([np.nan], [np.nan], [np.nan])

    metrics = evaluate_quantiles(actual, predictions, (0.1, 0.5, 0.9))

    assert math.isnan(metrics.p50_mae)
    assert math.isnan(metrics.coverage)
    assert math.isnan(metrics.interval_width)
    assert math.isnan(metrics.weighted_quantile_loss)


def test_metrics_do_not_mutate_inputs() -> None:
    actual = _actual([10.0, 20.0])
    predictions = _predictions([9.0, 19.0], [10.0, 20.0], [11.0, 21.0])
    actual_before = actual.copy(deep=True)
    predictions_before = predictions.copy(deep=True)

    evaluate_quantiles(actual, predictions, (0.1, 0.5, 0.9))

    assert_frame_equal(actual, actual_before)
    assert_frame_equal(predictions, predictions_before)
