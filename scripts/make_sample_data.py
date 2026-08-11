from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

PERIODS = 60 * 24
PARQUET_METADATA = {
    b"chronos_link_forecasting:data_origin": b"fully_synthetic",
    b"chronos_link_forecasting:synthetic": b"true",
    b"chronos_link_forecasting:generator": b"scripts/make_sample_data.py",
}


def make_sample_table(seed: int) -> pa.Table:
    """Build a deterministic table of synthetic hourly forecasting signals."""
    rng = np.random.default_rng(seed)
    hour = np.arange(PERIODS, dtype=np.float64)

    eth_price = 3_000.0 + np.cumsum(rng.normal(0.0, 4.0, PERIODS))
    gas_gwei = np.clip(
        20.0
        + 5.0 * np.sin((2.0 * np.pi * hour) / 24.0)
        + rng.normal(0.0, 1.5, PERIODS),
        1.0,
        None,
    )
    oracle_deviation = rng.normal(0.0, 0.0015, PERIODS)
    target = np.clip(
        14.0
        + 0.004 * (eth_price - 3_000.0)
        + 0.35 * np.sin((2.0 * np.pi * hour) / (24.0 * 7.0))
        + rng.normal(0.0, 0.12, PERIODS),
        0.01,
        None,
    )

    timestamp_values = np.arange(
        np.datetime64("2026-01-01T00:00:00", "ns"),
        np.datetime64("2026-03-02T00:00:00", "ns"),
        np.timedelta64(1, "h"),
    )
    schema = pa.schema(
        [
            pa.field("item_id", pa.string(), nullable=False),
            pa.field("timestamp", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("target", pa.float64(), nullable=False),
            pa.field("eth_price", pa.float64(), nullable=False),
            pa.field("gas_gwei", pa.float64(), nullable=False),
            pa.field("oracle_deviation", pa.float64(), nullable=False),
        ],
        metadata=PARQUET_METADATA,
    )
    return pa.Table.from_arrays(
        [
            pa.array(["LINK"] * PERIODS, type=pa.string()),
            pa.array(timestamp_values, type=pa.timestamp("ns", tz="UTC")),
            pa.array(target, type=pa.float64()),
            pa.array(eth_price, type=pa.float64()),
            pa.array(gas_gwei, type=pa.float64()),
            pa.array(oracle_deviation, type=pa.float64()),
        ],
        schema=schema,
    )


def write_sample(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        make_sample_table(seed),
        path,
        version="2.6",
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        row_group_size=PERIODS,
        data_page_version="1.0",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fully synthetic hourly LINK forecasting features."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_sample(args.output, args.seed)


if __name__ == "__main__":
    main()
