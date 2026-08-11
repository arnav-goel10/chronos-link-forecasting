# Chronos LINK Forecasting

Leakage-safe foundations for probabilistic LINK/USDT forecasting: strict experiment
configuration, temporal splitting, causal feature alignment, probabilistic baselines,
and quantile evaluation metrics.

This repository is about *correctness of the evaluation setup* rather than headline
accuracy. Time-series pipelines fail quietly when a feature join reaches slightly into
the future or a split is drawn at random, and the resulting numbers look excellent. The
code here makes those mistakes hard to commit, and the test suite is written to prove it.

> **Evidence boundary:** the checked-in dataset is fully synthetic. This repository
> publishes no LINK market results and makes no forecasting-accuracy claim.

## What it does

- **Strict experiment configuration.** TOML configs are parsed into frozen dataclasses
  that reject unknown fields, missing fields, wrong types, non-positive horizons, and
  quantile sets that omit 0.1/0.5/0.9. GPU runs additionally require `amazon/chronos-2`,
  bf16, and runtime FlashAttention-2 detection rather than a hardcoded flag.
- **Temporal splitting only.** `TemporalSplit` enforces
  `train_end < validation_end < test_end` and rejects mixed-timezone boundaries.
  `split_frame` partitions strictly by timestamp. There is no random split anywhere.
- **Causal feature alignment.** `causal_asof_join` performs a backward-only as-of join,
  so a feature may attach to a target row only if it was observable at or before that
  row's timestamp. It rejects duplicate timestamps, unsorted keys, null or `NaT`
  tolerances, non-datetime timestamp columns, and never mutates its inputs.
- **Probabilistic baselines.** `LastValueBaseline` and `SeasonalNaiveBaseline` emit
  quantile forecasts through a shared `ProbabilisticForecaster` protocol, so a foundation
  model can be swapped in without changing the evaluation path.
- **Quantile metrics.** `evaluate_quantiles` reports weighted quantile loss, P50 MAE,
  interval coverage, and mean interval width, and refuses to score frames whose keys or
  timestamps do not align exactly.

## Scope

Implemented and tested: configuration, temporal splits, causal alignment, baselines,
metrics, and the synthetic data generator.

Not yet in this repository: the Chronos-2 model adapter, the fine-tuning entry point,
the evaluation CLI, and published result artifacts. The `ProbabilisticForecaster`
protocol and `configs/h100-full.toml` define the interfaces those will implement.

## Quick start

Requires Python 3.12 or newer.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

pytest -q
```

Confirm the checked-in fixture matches a fresh regeneration from its seed:

```bash
PYTHONPATH=scripts python scripts/verify_sample_data.py \
  --path data/sample/link_features.parquet --seed 42
```

The check compares schema, Parquet metadata, row count, and values within a `1e-9`
tolerance. It is deliberately not a byte comparison: the seeded draws are stable, but
columns derived through `sin` and `cumsum` can differ in the last floating-point bit
across NumPy builds and CPU architectures, and the Parquet container records the writer
version. Byte-identity holds only within one pinned environment, so CI verifies the
values instead.

## Configuration

`configs/cpu-smoke.toml` is a credential-free configuration over the checked-in
synthetic fixture. Its boundaries divide the 1,440-row hourly sample into 744 training,
336 validation, and 360 test rows.

`configs/h100-full.toml` is a GPU template describing a private dataset that is not
distributed here. It is the shape a real run takes, not a runnable example.

## Quality gate

```bash
ruff check .
ruff format --check .
mypy
pytest -q
```

The suite covers 53 tests across configuration validation, split ordering and timezone
handling, future-leakage rejection, input immutability, baseline quantile monotonicity,
known-answer metric values, and alignment-key validation. CI runs the same gate on
Python 3.12.

## Data

See [DATA_CARD.md](DATA_CARD.md). The sample is generated from a seeded NumPy process,
declares `data_origin=fully_synthetic` in its Parquet metadata, and exists to exercise
loading, splitting, and alignment. It is not market data and supports no claim about
LINK behaviour.

## Further reading

- [Architecture](docs/architecture.md): module boundaries and the invariants each one owns.
- [Evaluation](docs/evaluation.md): metric definitions, label timing, and claim limits.

## License and security

Released under the [MIT License](LICENSE). Please report vulnerabilities as described in
[SECURITY.md](SECURITY.md).
