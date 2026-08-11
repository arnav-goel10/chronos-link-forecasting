"""Verify the checked-in synthetic fixture against a fresh regeneration.

Byte-for-byte equality is not a portable check. The seeded NumPy draws are
stable, but columns derived from them through ``sin``/``cumsum`` can differ in
the final floating-point bit across NumPy builds and CPU architectures, and the
Parquet container embeds the writer version. This script therefore verifies the
schema, metadata, row count, and values within a strict numerical tolerance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from make_sample_data import PARQUET_METADATA, make_sample_table

TOLERANCE = 1e-9
NUMERIC_COLUMNS = ("target", "eth_price", "gas_gwei", "oracle_deviation")


def verify(path: Path, seed: int) -> list[str]:
    """Return a list of failure messages; empty means the fixture is valid."""
    failures: list[str] = []
    actual = pq.read_table(path)
    expected = make_sample_table(seed)

    if actual.num_rows != expected.num_rows:
        failures.append(f"row count {actual.num_rows} != {expected.num_rows}")
    if actual.column_names != expected.column_names:
        failures.append(f"columns {actual.column_names} != {expected.column_names}")

    metadata = actual.schema.metadata or {}
    for key, value in PARQUET_METADATA.items():
        if metadata.get(key) != value:
            failures.append(
                f"metadata {key!r} is {metadata.get(key)!r}, want {value!r}"
            )

    if failures:
        return failures

    for name in ("item_id", "timestamp"):
        if not actual.column(name).equals(expected.column(name)):
            failures.append(f"column {name} differs")

    for name in NUMERIC_COLUMNS:
        left = actual.column(name).to_pylist()
        right = expected.column(name).to_pylist()
        worst = max(abs(a - b) for a, b in zip(left, right, strict=True))
        if worst > TOLERANCE:
            failures.append(f"column {name} differs by {worst:.3g} (> {TOLERANCE:g})")
        else:
            print(f"  {name:<18} max difference {worst:.3g}")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    print(f"pyarrow {pa.__version__}, verifying {args.path}")
    failures = verify(args.path, args.seed)
    if failures:
        print("\nFIXTURE VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)
    print("\nFIXTURE VERIFICATION PASSED")


if __name__ == "__main__":
    main()
