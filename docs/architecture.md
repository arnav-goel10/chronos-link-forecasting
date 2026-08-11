# Architecture

The package is organised so that every way of accidentally leaking future information
is blocked by a component that owns that specific invariant.

```text
configs/*.toml
      |
      v
ExperimentConfig  (strict parse, frozen)
      |
      v
TemporalSplit -> split_frame        train | validation | test, by timestamp only
      |
      v
causal_asof_join                    features observable at or before target time
      |
      v
ProbabilisticForecaster.predict     baselines today, foundation model later
      |
      v
evaluate_quantiles                  WQL, P50 MAE, coverage, interval width
```

## Modules and invariants

### `config.py`

Parses TOML into frozen `ExperimentConfig` and `TrainingConfig` dataclasses. Rejects
unknown and missing keys rather than silently ignoring them, because a typo in a config
key is otherwise indistinguishable from an intentional default.

Device-specific rules are enforced at construction: a `cuda` experiment must target
`amazon/chronos-2`, must supply a training block, must enable bf16, and must set
`flash_attention_2 = "auto"` so the attention backend is detected at runtime rather than
asserted by a config file that cannot know the hardware.

### `data/splits.py`

`TemporalSplit` normalises boundaries to `pd.Timestamp` and requires
`train_end < validation_end < test_end`. Comparing a timezone-aware boundary with a naive
one raises a clear `ValueError` instead of a `TypeError` from deep inside pandas.

`split_frame` partitions purely on timestamp with half-open intervals, so every row lands
in exactly one split and no row appears twice.

### `data/alignment.py`

`causal_asof_join` is the leakage boundary. It is a backward-only as-of join: a feature
row attaches to a target row only when its timestamp is at or before the target's, within
a positive tolerance.

It validates that both frames carry real datetime columns, that keys and timestamps sort
correctly, that tolerance is a positive non-null duration, and that neither input frame is
mutated. Multi-series inputs are aligned per key so residuals from one series can never
bleed into another.

### `models/`

`ProbabilisticForecaster` is a structural `Protocol` describing `predict(context,
prediction_length, quantiles) -> DataFrame`. Both baselines conform to it, which is what
allows a Chronos-2 adapter to be dropped in later without the evaluation path changing.

`LastValueBaseline` propagates the final observed value across the horizon.
`SeasonalNaiveBaseline` repeats the value one season back. Both derive quantile spreads
from in-context residuals, produce monotonically non-decreasing quantiles, and reject
mixed-series context rather than blending unrelated assets.

### `evaluation/metrics.py`

`evaluate_quantiles` returns a `MetricBundle` of weighted quantile loss, P50 MAE, interval
coverage, and mean interval width.

Alignment is strict by design. Predictions and actuals are joined on exact key and
timestamp pairs; null identifiers, `NaT` timestamps, non-datetime timestamp columns, and
partially overlapping windows are rejected. A metric computed over a silently truncated
join is worse than no metric, because it still looks like a number.
