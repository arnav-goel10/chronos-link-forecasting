import pandas as pd
import pytest

from chronos_link_forecasting.data.splits import TemporalSplit, split_frame

SPLIT = TemporalSplit("2026-01-03", "2026-01-05", "2026-01-07")


@pytest.fixture
def hourly_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01", "2026-01-08", freq="h", inclusive="left"
            ),
            "target": range(7 * 24),
        },
        index=range(1000, 1000 + (7 * 24)),
    )


def test_split_is_strictly_ordered() -> None:
    with pytest.raises(ValueError, match="train_end < validation_end < test_end"):
        TemporalSplit("2026-01-03", "2026-01-02", "2026-01-04")


def test_split_frame_has_no_overlap(hourly_frame: pd.DataFrame) -> None:
    train, validation, test = split_frame(hourly_frame, SPLIT)
    assert train.timestamp.max() < validation.timestamp.min()
    assert validation.timestamp.max() < test.timestamp.min()
    assert set(train.index).isdisjoint(validation.index)
    assert set(validation.index).isdisjoint(test.index)


def test_split_frame_uses_half_open_boundaries(hourly_frame: pd.DataFrame) -> None:
    train, validation, test = split_frame(hourly_frame, SPLIT)

    assert train["timestamp"].min() == pd.Timestamp("2026-01-01")
    assert train["timestamp"].max() == pd.Timestamp("2026-01-02 23:00:00")
    assert validation["timestamp"].min() == pd.Timestamp("2026-01-03")
    assert validation["timestamp"].max() == pd.Timestamp("2026-01-04 23:00:00")
    assert test["timestamp"].min() == pd.Timestamp("2026-01-05")
    assert test["timestamp"].max() == pd.Timestamp("2026-01-06 23:00:00")


def test_split_frame_excludes_rows_at_or_after_test_end(
    hourly_frame: pd.DataFrame,
) -> None:
    _, _, test = split_frame(hourly_frame, SPLIT)

    assert pd.Timestamp("2026-01-07") not in test["timestamp"].array


def test_split_frame_requires_timestamp_column(hourly_frame: pd.DataFrame) -> None:
    frame_without_timestamp = hourly_frame.drop(columns="timestamp")

    with pytest.raises(ValueError, match="timestamp column"):
        split_frame(frame_without_timestamp, SPLIT)


def test_split_frame_rejects_missing_timestamps(hourly_frame: pd.DataFrame) -> None:
    frame_with_missing_timestamp = hourly_frame.copy()
    frame_with_missing_timestamp.loc[1000, "timestamp"] = pd.NaT

    with pytest.raises(ValueError, match="timestamp column contains missing values"):
        split_frame(frame_with_missing_timestamp, SPLIT)
