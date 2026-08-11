# Evaluation

## Metrics

For quantile levels `q` with predictions `p_q` and actual `y`, the pinball loss is:

```text
L_q(y, p_q) = max(q * (y - p_q), (q - 1) * (y - p_q))
```

- **Weighted quantile loss (WQL).** Pinball loss summed across levels and observations,
  normalised by the total absolute actual value. Scale-free, so it is comparable across
  series with different magnitudes.
- **P50 MAE.** Mean absolute error of the median forecast. Point accuracy only.
- **Interval coverage.** Fraction of actuals falling within the P10 to P90 band. A
  calibrated 80 percent interval should cover close to 0.80.
- **Mean interval width.** Average P90 minus P10. Read together with coverage: an
  interval can always reach any coverage target by becoming uselessly wide.

Coverage and width are reported as a pair for that reason. Neither is meaningful alone.

## Label timing

Forecasts for a horizon starting at time `t` may use only information observable at or
before `t`. Two mechanisms enforce this:

1. `split_frame` divides data strictly by timestamp, so no test-period row can enter
   training.
2. `causal_asof_join` attaches covariates with a backward-only as-of join, so a covariate
   published after `t` cannot attach to a target at `t`.

Both are covered by tests that construct a future-dated feature and assert it does not
appear in the aligned output.

## What the numbers here do and do not support

The repository ships a synthetic fixture generated from a seeded NumPy process. Any metric
computed from it describes the behaviour of the code on that fixture and nothing else.

Specifically, results from this repository are **not** evidence that:

- these baselines forecast LINK prices well, or at all;
- a fine-tuned Chronos-2 model would outperform them;
- the covariate names (`eth_price`, `gas_gwei`, `oracle_deviation`) correspond to real
  measurements. They describe a schema shape, not observations.

Any future accuracy claim requires a named dataset, a stated split, a named baseline, a
unit, and a reproducible artifact. Until those exist, this repository documents an
evaluation harness rather than a result.
