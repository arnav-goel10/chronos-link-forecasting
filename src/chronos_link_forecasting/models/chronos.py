"""Zero-shot Chronos behind the ``ProbabilisticForecaster`` protocol.

The model stack (``chronos-forecasting`` and ``torch``) is an optional extra and
is imported only when a forecast is first requested, so the core package and
its tests stay light.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from chronos_link_forecasting.models.baselines import (
    _infer_frequency,
    _validate_context,
    _validate_quantiles,
)

DEFAULT_CHRONOS_MODEL = "amazon/chronos-bolt-tiny"
_INSTALL_HINT = (
    "ChronosForecaster needs the optional model extra: "
    "pip install 'chronos-link-forecasting[chronos]'"
)


class ChronosForecaster:
    """Zero-shot quantile forecasts from a pretrained Chronos checkpoint.

    ``pipeline`` may be injected for testing. Otherwise the checkpoint is loaded
    lazily through ``chronos.BaseChronosPipeline.from_pretrained``. No weights
    are updated: this adapter never fine-tunes.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_CHRONOS_MODEL,
        device: str = "cpu",
        pipeline: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self._device = device
        self._pipeline = pipeline

    def _load(self) -> Any:
        if self._pipeline is None:
            try:
                chronos = importlib.import_module("chronos")
            except ModuleNotFoundError as error:
                raise ImportError(_INSTALL_HINT) from error
            self._pipeline = chronos.BaseChronosPipeline.from_pretrained(
                self.model_id, device_map=self._device
            )
        return self._pipeline

    def predict(
        self,
        context: pd.DataFrame,
        prediction_length: int,
        quantiles: Sequence[float],
    ) -> pd.DataFrame:
        validated = _validate_context(context, prediction_length, quantiles)
        levels = _validate_quantiles(quantiles)
        pipeline = self._load()

        values = validated["target"].to_numpy(dtype="float64")
        series: Any
        try:
            torch = importlib.import_module("torch")
            series = torch.tensor(values, dtype=torch.float32)
        except ModuleNotFoundError:
            series = values
        raw, _mean = pipeline.predict_quantiles(series, prediction_length, list(levels))
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        array = np.asarray(raw, dtype="float64")
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.shape != (prediction_length, len(levels)):
            raise ValueError(
                "Chronos returned quantiles of shape "
                f"{array.shape}, expected {(prediction_length, len(levels))}"
            )
        if not all(math.isfinite(value) for value in array.ravel()):
            raise ValueError("Chronos returned non-finite quantiles")
        array = np.sort(array, axis=1)

        frequency = _infer_frequency(validated["timestamp"])
        timestamps = pd.date_range(
            start=validated["timestamp"].iloc[-1],
            periods=prediction_length + 1,
            freq=frequency,
        )[1:]
        frame = pd.DataFrame(array, columns=list(levels))
        frame.insert(0, "timestamp", timestamps)
        frame.insert(0, "item_id", validated["item_id"].iloc[-1])
        return frame
