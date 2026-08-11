from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, init=False)
class TemporalSplit:
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp

    def __init__(
        self,
        train_end: str | pd.Timestamp,
        validation_end: str | pd.Timestamp,
        test_end: str | pd.Timestamp,
    ) -> None:
        parsed_train_end = pd.Timestamp(train_end)
        parsed_validation_end = pd.Timestamp(validation_end)
        parsed_test_end = pd.Timestamp(test_end)

        try:
            strictly_ordered = (
                parsed_train_end < parsed_validation_end < parsed_test_end
            )
        except TypeError as error:
            raise ValueError(
                "split boundaries must use compatible timezones"
            ) from error
        if not strictly_ordered:
            raise ValueError("train_end < validation_end < test_end")

        object.__setattr__(self, "train_end", parsed_train_end)
        object.__setattr__(self, "validation_end", parsed_validation_end)
        object.__setattr__(self, "test_end", parsed_test_end)


def split_frame(
    frame: pd.DataFrame, split: TemporalSplit
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "timestamp" not in frame.columns:
        raise ValueError("frame must include a timestamp column")

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("timestamp column contains missing values")

    try:
        train_mask = timestamps < split.train_end
        validation_mask = (timestamps >= split.train_end) & (
            timestamps < split.validation_end
        )
        test_mask = (timestamps >= split.validation_end) & (timestamps < split.test_end)
    except TypeError as error:
        raise ValueError(
            "timestamp column and split boundaries must use compatible timezones"
        ) from error

    return (
        frame.loc[train_mask].copy(),
        frame.loc[validation_mask].copy(),
        frame.loc[test_mask].copy(),
    )
