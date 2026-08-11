# Data Card: Synthetic LINK Feature Sample

## Status

`data/sample/link_features.parquet` is fully synthetic and generated locally
from a seeded random process. It contains no observations from an external
dataset and is provided as a redistributable example.

## Contents

The sample contains 1,440 hourly rows (60 days) from 2026-01-01 00:00 UTC
through 2026-03-01 23:00 UTC. `item_id` is the constant label `LINK`.
`target`, `eth_price`, `gas_gwei`, and `oracle_deviation` are simulated numeric
signals created by `scripts/make_sample_data.py`; their names describe the
forecasting schema, not measurements of real assets or networks.

The Parquet schema metadata declares both
`chronos_link_forecasting:data_origin=fully_synthetic` and
`chronos_link_forecasting:synthetic=true`.

## Generation and reproducibility

The checked-in artifact is generated with NumPy's seeded random generator and
seed `42`. Within the verified Python 3.12 and PyArrow 21 environment it is
byte-identical across runs, with SHA-256
`0aac2a7dc9e8976199a4179a43e017611860b8bc3c83fa2dae9b3d060f819a51`.
PyArrow is bounded in `pyproject.toml` so the writer format does not silently
cross a major-version boundary.

Byte-identity does not hold across environments, and the repository does not
claim it does. The seeded draws (`eth_price`, `oracle_deviation`) reproduce
exactly, but `target` and `gas_gwei` are derived through `sin` and `cumsum`,
whose results can differ in the final floating-point bit between NumPy builds
and CPU architectures. A measured cross-environment difference was on the order
of `1e-15`. The Parquet container also records the writer version, so the file
bytes change with PyArrow. `scripts/verify_sample_data.py` therefore checks
schema, metadata, row count, and values within a `1e-9` tolerance, and CI runs
that check rather than a digest comparison.

## Intended use and limitations

This sample exists to exercise loading, temporal splitting, and causal feature
alignment. It is not evidence about historical or future LINK behavior, is not
representative market data, and does not support model-quality or outcome
claims.
