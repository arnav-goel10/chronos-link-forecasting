from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chronos_link_forecasting.models.chronos import ChronosForecaster


class FakePipeline:
    def __init__(self, output: Any) -> None:
        self.output = output
        self.calls: list[tuple[Any, int, list[float]]] = []

    def predict_quantiles(
        self, context: Any, prediction_length: int, quantile_levels: list[float]
    ) -> tuple[Any, Any]:
        self.calls.append((context, prediction_length, quantile_levels))
        return self.output, self.output[..., 1]


def _context(rows: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": "LINK-USD",
            "timestamp": pd.date_range("2026-01-01T00:00Z", periods=rows, freq="h"),
            "target": np.linspace(10.0, 12.0, rows),
        }
    )


def test_adapter_returns_sorted_quantiles_on_the_next_timestamps() -> None:
    raw = np.array([[[1.0, 3.0, 2.0], [4.0, 5.0, 6.0]]])
    pipeline = FakePipeline(raw)

    result = ChronosForecaster(pipeline=pipeline).predict(
        _context(), 2, (0.1, 0.5, 0.9)
    )

    assert result.columns.tolist() == ["item_id", "timestamp", 0.1, 0.5, 0.9]
    assert result[[0.1, 0.5, 0.9]].to_numpy().tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
    assert result["timestamp"].tolist() == [
        pd.Timestamp("2026-01-02T06:00Z"),
        pd.Timestamp("2026-01-02T07:00Z"),
    ]
    context, horizon, levels = pipeline.calls[0]
    assert len(context) == 30
    assert horizon == 2
    assert levels == [0.1, 0.5, 0.9]


def test_adapter_rejects_wrong_shape() -> None:
    pipeline = FakePipeline(np.zeros((1, 3, 3)))
    with pytest.raises(ValueError, match="shape"):
        ChronosForecaster(pipeline=pipeline).predict(_context(), 2, (0.1, 0.5, 0.9))


def test_missing_extra_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> Any:
        if name == "chronos":
            raise ModuleNotFoundError(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(ImportError, match=r"\[chronos\]"):
        ChronosForecaster().predict(_context(), 2, (0.1, 0.5, 0.9))
