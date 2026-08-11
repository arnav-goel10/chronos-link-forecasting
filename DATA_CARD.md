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
seed `42`. Two runs in the verified Python 3.12 and PyArrow 21 environment were
byte-identical to the checked-in artifact, with SHA-256
`0aac2a7dc9e8976199a4179a43e017611860b8bc3c83fa2dae9b3d060f819a51`.
PyArrow is bounded in `pyproject.toml` so the writer format does not silently
cross a major-version boundary.

## Intended use and limitations

This sample exists to exercise loading, temporal splitting, and causal feature
alignment. It is not evidence about historical or future LINK behavior, is not
representative market data, and does not support model-quality or outcome
claims.
