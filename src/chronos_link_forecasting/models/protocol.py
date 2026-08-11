from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import pandas as pd


class ProbabilisticForecaster(Protocol):
    """Structural interface shared by probabilistic forecasting models."""

    def predict(
        self,
        context: pd.DataFrame,
        prediction_length: int,
        quantiles: Sequence[float],
    ) -> pd.DataFrame:
        """Forecast requested quantiles after the supplied historical context."""
        ...
