# Fixed trading project: changes and run order

This is a source-code update for the existing project. Extract it **over your existing project folder** so your local data and existing results are preserved. Do not delete your `data/` or `results/` folders.

## What changed

1. The trade chart marks only actual position transitions, not repeated raw BUY/SELL decisions.
2. The environment records requested and executed actions separately, plus previous/current position, decision date, and the date of the return earned.
3. Equity/drawdown plots use the P&L realization date; buy/sell markers remain on the decision date. This resolves the one-session visual offset.
4. The reconstructed price index begins at 1.0 at the first observation.
5. Added `results/action_diagnostics.csv`, with action counts, real entries/exits, redundant actions, long/flat days, and guardrail overrides for each experiment, seed, and split.
6. Sortino uses downside deviation relative to a zero-return target.
7. Corrected `momentum_signal`: it is now the trailing 20-session directional return, rather than lag-1 return autocorrelation. The old feature was not actually a directional momentum signal.
8. Default turnover penalty is `0.0008`, consistent with the values explored in your sweep. The old README recommendation of `0.05-0.2` was inconsistent with reward scaling.
9. Added deterministic environment accounting tests in `tests/test_env_accounting.py`.

## Run order (PowerShell, from project root)

Activate your existing venv first. Then rebuild the feature CSV because `momentum_signal` changed:

```powershell
python features/build_features.py --price_csv data/gold.csv --ats_csv results/ats_features.csv --out results/features_all.csv
```

If your source files use different names/paths, substitute your actual paths. Do not proceed if the build command fails or the output date range/row count unexpectedly changes.

Run a short smoke test first:

```powershell
python run_experiments.py --features_csv results/features_all.csv --train_end 2016-12-31 --valid_end 2020-12-31 --timesteps 3000 --seeds 1 --turnover_penalty 0.0008
```

Inspect `results/action_diagnostics.csv`, `results/results_table.md`, and the new `results/backtest_*_seed0.png` files. If the smoke test completes and files look sensible, run the full comparison:

```powershell
python run_experiments.py --features_csv results/features_all.csv --train_end 2016-12-31 --valid_end 2020-12-31 --timesteps 100000 --seeds 5 --turnover_penalty 0.0008
```

## Important interpretation notes

- The turnover penalty affects PPO's training reward only. It is not subtracted from reported P&L. Transaction costs are included in reported P&L.
- A flat equity section is expected while the agent is in cash. A flat line alone is not evidence of a bug.
- A guardrail override can close a position even if the raw action was HOLD; requested/executed action columns now expose this.
- This patch improves diagnostics and corrects identified inconsistencies; it does **not** guarantee positive returns or that RL will beat buy-and-hold.
- Keep the final test period out of further hyperparameter selection. Use validation or walk-forward results for model choices.
- The uploaded files did not include the upstream source that creates `results/ats_features.csv`; therefore the causal/no-look-ahead status of `regime_flag` and `garch_cond_vol` still needs to be verified in that upstream ATS pipeline.
