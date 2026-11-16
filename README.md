# Chronos LINK Forecasting

[![CI](https://github.com/arnav-goel10/chronos-link-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/arnav-goel10/chronos-link-forecasting/actions/workflows/ci.yml)
[![Benchmark](https://github.com/arnav-goel10/chronos-link-forecasting/actions/workflows/benchmark.yml/badge.svg)](https://github.com/arnav-goel10/chronos-link-forecasting/actions/workflows/benchmark.yml)

Zero-shot Chronos against two naive quantile baselines on real hourly LINK/USD prices,
scored by a rolling-origin backtest in which no model ever sees a row at or after its
forecast origin.

On the committed run, zero-shot Chronos-Bolt Tiny has the lowest weighted quantile loss
of the three models. The last-value baseline still has the lower median error. Every
figure below is copied from
[`results/link_usd_1h_backtest.json`](results/link_usd_1h_backtest.json), which GitHub
Actions run
[43778397490](https://github.com/arnav-goel10/chronos-link-forecasting/actions/runs/43778397490)
produced and uploaded as an artifact.

## Results

| Model | Weighted quantile loss | P50 MAE (USD) | 80% interval coverage | Mean 80% interval width (USD) |
| --- | ---: | ---: | ---: | ---: |
| Chronos zero-shot (amazon/chronos-bolt-tiny) | 0.0140 | 0.2610 | 72.3% | 0.7083 |
| Last value | 0.0172 | 0.2511 | 33.1% | 0.1942 |
| Seasonal naive (24h) | 0.0184 | 0.3519 | 80.7% | 1.1548 |

Lower is better for weighted quantile loss and P50 MAE. A calibrated 80% interval covers
close to 80%, and coverage only means something next to width.

**Setup, from the same file.** Coinbase `LINK-USD` hourly closes from
2026-05-20 00:00 to 2026-11-15 23:00 UTC, 4320 rows, 0 hours carried forward. 119
forecast origins from 2026-09-17 00:00 to 2026-11-15 00:00 UTC, 12 hours apart. Each
model gets the previous 512 hours and forecasts the next 24, giving 2856 scored points.
Quantiles are P10, P50 and P90. Chronos ran on a CPU runner with `chronos-forecasting`
2.3.2 and `torch` 2.15.0+cpu.

**How to read it.**

- Chronos wins on the probabilistic score, not on the point forecast. Its P50 MAE is a
  little worse than simply repeating the last close.
- The last-value interval is far too narrow. Its spread comes from one-hour price
  changes and is applied unchanged across a 24-hour horizon, so it covers 33.1% of
  outcomes against an 80% target.
- Seasonal naive reaches its coverage by being the widest interval, and pays for it in
  both loss and median error.
- Chronos is under-covered too, at 72.3%, with a narrower interval than seasonal naive.

**What this does not show.** It is one asset, one 1440-hour test period, one small
checkpoint and no covariates. There is no fine-tuning and no significance test, and the
three models are compared on a single run. Candles are pulled at run time and are not
committed, so a new run scores a different window. The JSON records the venue, the
window and a SHA-256 of the timestamp and close columns so a run can be matched to its
data.

## What it does

- **Live public data.** `scripts/fetch_link_candles.py` pulls hourly LINK/USD candles
  from the Coinbase Exchange public API with no key. If Coinbase is unreachable it falls
  back to Kraken's public OHLC endpoint, which returns only the last 720 hours, so a
  Kraken run needs a smaller `--test-hours`. Hours with no trades carry the previous
  close forward, and the count of such hours is recorded.
- **Rolling-origin backtest.** `rolling_origin_backtest` hands each model a copy of the
  fixed-length context strictly before the origin and nothing else. Tests check that
  every context ends one hour before its origin and that overwriting all future values
  leaves every forecast unchanged.
- **Zero-shot Chronos as an optional extra.** `ChronosForecaster` implements the same
  `ProbabilisticForecaster` protocol as the baselines and loads `amazon/chronos-bolt-tiny`
  lazily, so the core package and its tests never import `torch`.
- **Evaluate command.** `python -m chronos_link_forecasting.evaluate` runs every model
  through the backtest and writes the results JSON, including provenance and package
  versions.
- **Strict experiment configuration.** TOML configs are parsed into frozen dataclasses
  that reject unknown fields, missing fields, wrong types, non-positive horizons, and
  quantile sets that omit 0.1/0.5/0.9.
- **Temporal splitting and causal alignment.** `split_frame` partitions strictly by
  timestamp, and `causal_asof_join` is a backward-only as-of join that never lets a
  feature attach to a row before it was observable.
- **Probabilistic baselines.** `LastValueBaseline` and `SeasonalNaiveBaseline` emit
  quantile forecasts from context-only residuals.
- **Quantile metrics.** `evaluate_quantiles` reports weighted quantile loss, P50 MAE,
  P10 to P90 coverage and mean interval width, and refuses to score frames whose keys or
  timestamps do not align exactly.

## Scope

Implemented and tested: data fetching, the rolling-origin backtest, the zero-shot
Chronos adapter, the evaluate command, configuration, temporal splits, causal alignment,
baselines, metrics, and the synthetic data generator.

Not in this repository: fine-tuning, a Chronos-2 run, and covariate-aware forecasting.
`configs/h100-full.toml` describes the shape a fine-tuning run would take and is not a
runnable example.

## Run the benchmark

The **Benchmark** workflow does the whole run on a GitHub-hosted CPU runner. Start it
from the Actions tab and download the `link-usd-1h-backtest` artifact. To run it
yourself:

```bash
python -m pip install -e '.[chronos]'
python scripts/fetch_link_candles.py --days 180 --out artifacts/link_usd_1h.parquet
python -m chronos_link_forecasting.evaluate \
  --data artifacts/link_usd_1h.parquet --out artifacts/link_usd_1h_backtest.json
```

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

The suite covers configuration validation, split ordering and timezone handling,
future-leakage rejection in both the feature join and the backtest, input immutability,
baseline quantile monotonicity, known-answer metric values, candle parsing and venue
fallback with no network, the Chronos adapter against a fake pipeline, and a guard that
the committed results file keeps its schema and still matches the table in this README.
CI runs the same gate on Python 3.12 without installing the model stack.

## Data

Benchmark data is pulled from Coinbase at run time and is not redistributed here. The
checked-in fixture is separate: see [DATA_CARD.md](DATA_CARD.md). The sample is generated from a seeded NumPy process,
declares `data_origin=fully_synthetic` in its Parquet metadata, and exists to exercise
loading, splitting, and alignment. It is not market data and supports no claim about
LINK behaviour.

## Further reading

- [Architecture](docs/architecture.md): module boundaries and the invariants each one owns.
- [Evaluation](docs/evaluation.md): metric definitions, label timing, and claim limits.

## License and security

Released under the [MIT License](LICENSE). Please report vulnerabilities as described in
[SECURITY.md](SECURITY.md).
