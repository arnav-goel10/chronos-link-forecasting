from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from chronos_link_forecasting.models.baselines import (
    LastValueBaseline,
    SeasonalNaiveBaseline,
)


def _context(target: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["LINK"] * len(target),
            "timestamp": pd.date_range(
                "2026-01-01 00:00Z", periods=len(target), freq="h"
            ),
            "target": target,
        }
    )


def test_last_value_p50_repeats_final_context_target() -> None:
    context = _context([10.0, 12.0, 11.0])

    result = LastValueBaseline().predict(context, 2, (0.1, 0.5, 0.9))

    assert result[0.5].tolist() == [11.0, 11.0]
    assert result.columns.tolist() == ["item_id", "timestamp", 0.1, 0.5, 0.9]


def test_last_value_uses_centered_first_difference_quantiles() -> None:
    context = _context([10.0, 11.0, 13.0, 12.0, 12.0])

    result = LastValueBaseline().predict(context, 1, (0.0 + 0.25, 0.5, 0.75))

    # Residuals are [1, 2, -1, 0], whose linear 25/50/75% quantiles are
    # [-0.25, 0.5, 1.25]. Centering on the residual median gives offsets
    # [-0.75, 0, 0.75] around the last-value median of 12.
    assert result[[0.25, 0.5, 0.75]].iloc[0].tolist() == [11.25, 12.0, 12.75]


def test_seasonal_naive_repeats_values_from_last_season() -> None:
    context = _context([10.0, 20.0, 11.0, 21.0])

    result = SeasonalNaiveBaseline(season_length=2).predict(context, 3, (0.1, 0.5, 0.9))

    assert result[0.5].tolist() == [11.0, 21.0, 11.0]


def test_seasonal_naive_uses_only_season_lag_residuals() -> None:
    context = _context([10.0, 20.0, 11.0, 23.0, 14.0, 24.0])

    result = SeasonalNaiveBaseline(season_length=2).predict(
        context, 1, (0.25, 0.5, 0.75)
    )

    # Lag-2 residuals are [1, 3, 3, 1], giving centered offsets [-1, 0, 1].
    assert result[[0.25, 0.5, 0.75]].iloc[0].tolist() == [13.0, 14.0, 15.0]


@pytest.mark.parametrize(
    "forecaster",
    [LastValueBaseline(), SeasonalNaiveBaseline(season_length=2)],
)
def test_baselines_return_monotonic_quantiles_at_every_horizon(
    forecaster: LastValueBaseline | SeasonalNaiveBaseline,
) -> None:
    context = _context([10.0, 12.0, 9.0, 14.0, 11.0, 16.0])

    result = forecaster.predict(context, 4, (0.1, 0.5, 0.9))

    assert (result[0.1] <= result[0.5]).all()
    assert (result[0.5] <= result[0.9]).all()


@pytest.mark.parametrize(
    "forecaster",
    [LastValueBaseline(), SeasonalNaiveBaseline(season_length=2)],
)
def test_baselines_preserve_item_and_forecast_regular_tz_aware_timestamps(
    forecaster: LastValueBaseline | SeasonalNaiveBaseline,
) -> None:
    context = _context([10.0, 12.0, 9.0, 14.0])

    result = forecaster.predict(context, 3, (0.1, 0.5, 0.9))

    assert result["item_id"].tolist() == ["LINK"] * 3
    assert result["timestamp"].tolist() == list(
        pd.date_range("2026-01-01 04:00Z", periods=3, freq="h")
    )
    assert str(result["timestamp"].dtype) == "datetime64[ns, UTC]"


@pytest.mark.parametrize(
    "forecaster",
    [LastValueBaseline(), SeasonalNaiveBaseline(season_length=2)],
)
def test_baselines_do_not_mutate_context(
    forecaster: LastValueBaseline | SeasonalNaiveBaseline,
) -> None:
    context = _context([10.0, 12.0, 9.0, 14.0])
    before = context.copy(deep=True)

    forecaster.predict(context, 2, (0.1, 0.5, 0.9))

    assert_frame_equal(context, before)


@pytest.mark.parametrize(
    ("quantiles", "message"),
    [
        ((0.1, 0.1, 0.9), "unique"),
        ((0.5, 0.1, 0.9), "strictly increasing"),
        ((0.0, 0.5, 0.9), "between 0 and 1"),
        ((0.1, 0.5, 1.0), "between 0 and 1"),
        ((0.1, float("nan"), 0.9), "finite"),
    ],
)
def test_baselines_reject_invalid_quantiles(
    quantiles: tuple[float, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LastValueBaseline().predict(_context([1.0, 2.0, 3.0]), 1, quantiles)


@pytest.mark.parametrize("target", [[1.0, np.inf, 3.0], [1.0, np.nan, 3.0]])
def test_baselines_reject_non_finite_targets(target: list[float]) -> None:
    with pytest.raises(ValueError, match="target values must be finite"):
        LastValueBaseline().predict(_context(target), 1, (0.1, 0.5, 0.9))


@pytest.mark.parametrize("target", [[1.0, 2.0], [1.0, 2.0, 3.0]])
def test_seasonal_naive_rejects_context_without_a_seasonal_residual(
    target: list[float],
) -> None:
    with pytest.raises(ValueError, match="more than season_length"):
        SeasonalNaiveBaseline(season_length=3).predict(
            _context(target), 1, (0.1, 0.5, 0.9)
        )


def test_baselines_explicitly_reject_multiple_items() -> None:
    context = _context([1.0, 2.0, 3.0, 4.0])
    context.loc[3, "item_id"] = "ETH"

    with pytest.raises(ValueError, match="exactly one item_id"):
        LastValueBaseline().predict(context, 1, (0.1, 0.5, 0.9))


def test_baselines_reject_two_timestamps_as_an_ambiguous_frequency() -> None:
    with pytest.raises(ValueError, match="regular, strictly increasing frequency"):
        LastValueBaseline().predict(_context([1.0, 2.0]), 1, (0.1, 0.5, 0.9))


@pytest.mark.parametrize(
    "timestamps",
    [
        [
            "2026-01-01 00:00Z",
            "2026-01-01 01:00Z",
            "2026-01-01 03:00Z",
        ],
        [
            "2026-01-01 00:00Z",
            "2026-01-01 02:00Z",
            "2026-01-01 01:00Z",
        ],
        [
            "2026-01-01 00:00Z",
            "2026-01-01 01:00Z",
            "2026-01-01 01:00Z",
        ],
    ],
)
def test_baselines_reject_ambiguous_timestamp_frequency(
    timestamps: list[str],
) -> None:
    context = _context([1.0, 2.0, 3.0])
    context["timestamp"] = pd.to_datetime(timestamps)

    with pytest.raises(ValueError, match="regular, strictly increasing frequency"):
        LastValueBaseline().predict(context, 1, (0.1, 0.5, 0.9))


@pytest.mark.parametrize("prediction_length", [0, -1, True, 1.5])
def test_baselines_reject_invalid_prediction_length(
    prediction_length: int | float | bool,
) -> None:
    with pytest.raises(
        ValueError, match="prediction_length must be a positive integer"
    ):
        LastValueBaseline().predict(
            _context([1.0, 2.0, 3.0]),
            prediction_length,  # type: ignore[arg-type]
            (0.1, 0.5, 0.9),
        )


@pytest.mark.parametrize("season_length", [0, -1, True, 1.5])
def test_seasonal_naive_rejects_invalid_season_length(
    season_length: int | float | bool,
) -> None:
    with pytest.raises(ValueError, match="season_length must be a positive integer"):
        SeasonalNaiveBaseline(season_length=season_length)  # type: ignore[arg-type]
