from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from pandas.testing import assert_frame_equal

from chronos_link_forecasting.data.alignment import causal_asof_join

REPOSITORY_ROOT = Path(__file__).parents[1]
GENERATOR = REPOSITORY_ROOT / "scripts" / "make_sample_data.py"


def test_causal_join_never_uses_future_observation() -> None:
    target = pd.DataFrame(
        {
            "item_id": ["LINK"] * 2,
            "timestamp": pd.to_datetime(["2026-01-01 10:00Z", "2026-01-01 11:00Z"]),
            "target": [10.0, 11.0],
        }
    )
    feature = pd.DataFrame(
        {
            "item_id": ["LINK"] * 2,
            "timestamp": pd.to_datetime(["2026-01-01 09:55Z", "2026-01-01 10:05Z"]),
            "eth_price": [3000.0, 999999.0],
        }
    )

    joined = causal_asof_join(
        target,
        feature,
        timestamp="timestamp",
        by="item_id",
        tolerance="2h",
    )

    assert joined.loc[0, "eth_price"] == 3000.0
    first_observed_at = cast(pd.Timestamp, joined.loc[0, "eth_price__observed_at"])
    first_target_at = cast(pd.Timestamp, joined.loc[0, "timestamp"])
    assert first_observed_at <= first_target_at
    observed = joined["eth_price__observed_at"].dropna()
    assert (observed <= joined.loc[observed.index, "timestamp"]).all()


def test_causal_join_audits_every_feature_observation() -> None:
    target = pd.DataFrame(
        {
            "item_id": ["LINK"],
            "timestamp": pd.to_datetime(["2026-01-01 10:00Z"]),
            "target": [10.0],
        }
    )
    feature = pd.DataFrame(
        {
            "item_id": ["LINK"],
            "timestamp": pd.to_datetime(["2026-01-01 09:45Z"]),
            "eth_price": [3000.0],
            "gas_gwei": [18.0],
            "oracle_deviation": [0.002],
        }
    )

    joined = causal_asof_join(
        target,
        feature,
        timestamp="timestamp",
        by="item_id",
        tolerance="30min",
    )

    expected_observed_at = pd.Timestamp("2026-01-01 09:45Z")
    for feature_name in ("eth_price", "gas_gwei", "oracle_deviation"):
        observed_at = cast(pd.Timestamp, joined.loc[0, f"{feature_name}__observed_at"])
        target_at = cast(pd.Timestamp, joined.loc[0, "timestamp"])
        assert observed_at == expected_observed_at
        assert observed_at <= target_at


def test_causal_join_rejects_duplicate_target_key_timestamp() -> None:
    target = pd.DataFrame(
        {
            "item_id": ["LINK", "LINK"],
            "timestamp": pd.to_datetime(["2026-01-01 10:00Z", "2026-01-01 10:00Z"]),
            "target": [10.0, 11.0],
        }
    )
    feature = pd.DataFrame(
        {
            "item_id": ["LINK"],
            "timestamp": pd.to_datetime(["2026-01-01 09:55Z"]),
            "eth_price": [3000.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate target key/timestamp"):
        causal_asof_join(
            target,
            feature,
            timestamp="timestamp",
            by="item_id",
            tolerance="1h",
        )


def test_causal_join_does_not_mutate_inputs() -> None:
    target = pd.DataFrame(
        {
            "item_id": ["LINK", "LINK"],
            "timestamp": pd.to_datetime(["2026-01-01 11:00Z", "2026-01-01 10:00Z"]),
            "target": [11.0, 10.0],
        },
        index=[8, 3],
    )
    feature = pd.DataFrame(
        {
            "item_id": ["LINK", "LINK"],
            "timestamp": pd.to_datetime(["2026-01-01 10:30Z", "2026-01-01 09:30Z"]),
            "eth_price": [3100.0, 3000.0],
        },
        index=[7, 2],
    )
    target_before = target.copy(deep=True)
    feature_before = feature.copy(deep=True)

    causal_asof_join(
        target,
        feature,
        timestamp="timestamp",
        by="item_id",
        tolerance="2h",
    )

    assert_frame_equal(target, target_before)
    assert_frame_equal(feature, feature_before)


@pytest.mark.parametrize("tolerance", ["not-a-duration", "-1h", "0h"])
def test_causal_join_rejects_invalid_tolerance(tolerance: str) -> None:
    target = pd.DataFrame(
        {
            "item_id": ["LINK"],
            "timestamp": pd.to_datetime(["2026-01-01 10:00Z"]),
            "target": [10.0],
        }
    )
    feature = pd.DataFrame(
        {
            "item_id": ["LINK"],
            "timestamp": pd.to_datetime(["2026-01-01 09:55Z"]),
            "eth_price": [3000.0],
        }
    )

    with pytest.raises(ValueError, match="tolerance must be a positive duration"):
        causal_asof_join(
            target,
            feature,
            timestamp="timestamp",
            by="item_id",
            tolerance=tolerance,
        )


@pytest.mark.parametrize(
    ("target", "feature", "message"),
    [
        (
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2026-01-01 10:00Z"]),
                    "target": [10.0],
                }
            ),
            pd.DataFrame(
                {
                    "item_id": ["LINK"],
                    "timestamp": pd.to_datetime(["2026-01-01 09:55Z"]),
                    "eth_price": [3000.0],
                }
            ),
            "target must include columns: item_id",
        ),
        (
            pd.DataFrame(
                {
                    "item_id": ["LINK"],
                    "timestamp": pd.to_datetime(["2026-01-01 10:00Z"]),
                    "target": [10.0],
                }
            ),
            pd.DataFrame({"item_id": ["LINK"], "eth_price": [3000.0]}),
            "feature must include columns: timestamp",
        ),
        (
            pd.DataFrame(
                {
                    "item_id": ["LINK"],
                    "timestamp": pd.to_datetime(["2026-01-01 10:00Z"]),
                    "target": [10.0],
                }
            ),
            pd.DataFrame(
                {
                    "item_id": ["LINK"],
                    "timestamp": pd.to_datetime(["2026-01-01 09:55Z"]),
                }
            ),
            "feature must include at least one value column",
        ),
    ],
)
def test_causal_join_rejects_invalid_schema(
    target: pd.DataFrame, feature: pd.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        causal_asof_join(
            target,
            feature,
            timestamp="timestamp",
            by="item_id",
            tolerance="1h",
        )


def test_causal_join_rejects_missing_or_non_datetime_timestamps() -> None:
    target = pd.DataFrame(
        {
            "item_id": ["LINK"],
            "timestamp": ["2026-01-01 10:00Z"],
            "target": [10.0],
        }
    )
    feature = pd.DataFrame(
        {
            "item_id": ["LINK"],
            "timestamp": pd.to_datetime(["2026-01-01 09:55Z"]),
            "eth_price": [3000.0],
        }
    )

    with pytest.raises(ValueError, match="target timestamp must be datetime-like"):
        causal_asof_join(
            target,
            feature,
            timestamp="timestamp",
            by="item_id",
            tolerance="1h",
        )

    target["timestamp"] = pd.to_datetime(target["timestamp"])
    feature.loc[0, "timestamp"] = pd.NaT
    with pytest.raises(ValueError, match="feature timestamp contains missing values"):
        causal_asof_join(
            target,
            feature,
            timestamp="timestamp",
            by="item_id",
            tolerance="1h",
        )


def _run_generator(output: Path, seed: int) -> None:
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(output), "--seed", str(seed)],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_sample_generator_is_seeded_and_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    other_seed = tmp_path / "other-seed.parquet"

    _run_generator(first, 42)
    _run_generator(second, 42)
    _run_generator(other_seed, 43)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() != other_seed.read_bytes()


def test_sample_generator_writes_declared_synthetic_hourly_schema(
    tmp_path: Path,
) -> None:
    output = tmp_path / "sample.parquet"

    _run_generator(output, 42)

    parquet_file = pq.ParquetFile(output)
    metadata = parquet_file.schema_arrow.metadata
    assert metadata is not None
    assert metadata[b"chronos_link_forecasting:data_origin"] == b"fully_synthetic"
    assert metadata[b"chronos_link_forecasting:synthetic"] == b"true"

    frame = pd.read_parquet(output)
    assert list(frame.columns) == [
        "item_id",
        "timestamp",
        "target",
        "eth_price",
        "gas_gwei",
        "oracle_deviation",
    ]
    assert len(frame) == 60 * 24
    assert frame["item_id"].eq("LINK").all()
    assert frame["timestamp"].iloc[0] == pd.Timestamp("2026-01-01 00:00Z")
    assert frame["timestamp"].iloc[-1] == pd.Timestamp("2026-03-01 23:00Z")
    assert frame["timestamp"].diff().dropna().eq(np.timedelta64(1, "h")).all()
