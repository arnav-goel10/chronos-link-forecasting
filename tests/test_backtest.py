from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos_link_forecasting.evaluate import main, render_markdown_table
from chronos_link_forecasting.evaluation.backtest import rolling_origin_backtest
from chronos_link_forecasting.models.baselines import LastValueBaseline

QUANTILES = (0.1, 0.5, 0.9)


def _series(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "item_id": "LINK-USD",
            "timestamp": pd.date_range("2026-01-01T00:00Z", periods=rows, freq="h"),
            "target": 15.0 + rng.normal(0, 0.1, rows).cumsum(),
        }
    )


class RecordingForecaster:
    def __init__(self) -> None:
        self.contexts: list[pd.DataFrame] = []
        self._inner = LastValueBaseline()

    def predict(
        self,
        context: pd.DataFrame,
        prediction_length: int,
        quantiles: Sequence[float],
    ) -> pd.DataFrame:
        self.contexts.append(context.copy())
        return self._inner.predict(context, prediction_length, quantiles)


def test_each_window_sees_only_rows_before_its_origin() -> None:
    frame = _series(200)
    forecaster = RecordingForecaster()

    result = rolling_origin_backtest(
        frame,
        forecaster,
        prediction_length=6,
        quantiles=QUANTILES,
        first_origin=frame["timestamp"].iloc[120],
        stride=4,
        context_length=48,
    )

    assert len(result.origins) == len(forecaster.contexts) == 19
    for origin, context in zip(result.origins, forecaster.contexts, strict=True):
        assert len(context) == 48
        assert context["timestamp"].max() < origin
        assert context["timestamp"].max() == origin - pd.Timedelta(hours=1)


def test_future_values_cannot_change_any_forecast() -> None:
    frame = _series(200)
    poisoned = frame.copy()
    origin_index = 150
    poisoned.loc[origin_index:, "target"] = 1e6

    def run(data: pd.DataFrame) -> pd.DataFrame:
        return rolling_origin_backtest(
            data,
            LastValueBaseline(),
            prediction_length=6,
            quantiles=QUANTILES,
            first_origin=data["timestamp"].iloc[origin_index],
            stride=200,
            context_length=48,
        ).predictions

    pd.testing.assert_frame_equal(run(frame), run(poisoned))


def test_overlapping_windows_keep_unique_keys() -> None:
    frame = _series(120)
    result = rolling_origin_backtest(
        frame,
        LastValueBaseline(),
        prediction_length=6,
        quantiles=QUANTILES,
        first_origin=frame["timestamp"].iloc[60],
        stride=1,
        context_length=24,
    )

    assert not result.actual.duplicated(subset=["item_id", "timestamp"]).any()
    assert (
        result.actual["timestamp"].tolist() == result.predictions["timestamp"].tolist()
    )


def test_rejects_origin_without_full_context() -> None:
    frame = _series(60)
    with pytest.raises(ValueError, match="context_length"):
        rolling_origin_backtest(
            frame,
            LastValueBaseline(),
            prediction_length=6,
            quantiles=QUANTILES,
            first_origin=frame["timestamp"].iloc[10],
            stride=6,
            context_length=24,
        )


def test_evaluate_cli_writes_results_and_table(tmp_path: Path) -> None:
    data_path = tmp_path / "link.parquet"
    _series(400).to_parquet(data_path, index=False)
    (tmp_path / "link.meta.json").write_text(json.dumps({"venue": "test"}))
    out = tmp_path / "results.json"

    code = main(
        [
            "--data",
            str(data_path),
            "--out",
            str(out),
            "--models",
            "last-value,seasonal-naive",
            "--prediction-length",
            "12",
            "--stride",
            "12",
            "--context-length",
            "96",
            "--test-hours",
            "120",
        ]
    )

    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["data"]["venue"] == "test"
    assert payload["backtest"]["windows"] == 10
    assert set(payload["models"]) == {"last-value", "seasonal-naive"}
    table = render_markdown_table(payload)
    assert table.count("\n") == 3
