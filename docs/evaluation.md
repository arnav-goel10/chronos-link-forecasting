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

## Baseline interval behaviour

Both baselines derive their quantile spread from residuals observed in the
context window, and apply that same spread at every step of the horizon. The
intervals are therefore flat: P90 minus P10 is identical at step 1 and step 12.

That is deliberate for a naive baseline, but it is not what a random walk does.
Genuine forecast uncertainty grows roughly with the square root of the horizon,
so interval coverage from these baselines should be expected to decay at longer
horizons while the reported mean interval width stays constant. Read the two
together, and treat these baselines as a floor for calibration comparisons
rather than as calibrated models.

## Backtest

`rolling_origin_backtest` places forecast origins every `stride` hours across the last
`test_hours` of the series. For an origin at row `i`, the model receives a copy of rows
`[i - context_length, i)` and the next `prediction_length` rows are held back as the
actuals. Every window uses the same context length, and each window is scored under its
own `item_id`, so overlapping windows cannot collide on keys. No model is fitted, so
there is no training or validation split to leak from. The baselines estimate their
spread from the context only, and Chronos is used zero-shot.

## What the numbers here do and do not support

`results/link_usd_1h_backtest.json` is a measurement of three models on one window of
Coinbase LINK-USD hourly closes. It supports a comparison of those models on that window
under the stated backtest. It is **not** evidence that:

- the ranking holds on another period, another asset or another horizon, since no
  significance test or repeated run is reported;
- a fine-tuned Chronos or Chronos-2 model would do better or worse;
- the synthetic fixture's covariate names (`eth_price`, `gas_gwei`, `oracle_deviation`)
  correspond to real measurements. They describe a schema shape, not observations.

Metrics computed on the checked-in synthetic fixture describe the behaviour of the code
on that fixture and nothing else.
