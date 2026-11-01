# RL trading agent with statistical (ATS) risk awareness

One codebase, three experiments, one honest evaluation.

| Experiment | What the PPO agent sees | Guardrail |
|---|---|---|
| **A** | raw price features (returns, rolling vol, RSI) | no |
| **B** | A + GARCH volatility forecast, regime flag, momentum signal | no |
| **C** | same as B | yes: high predicted vol blocks buys and forces an exit |

Baselines run through the same environment and cost model: cash, buy & hold,
a 10-day momentum rule, and a "vol filter" (long only when predicted vol is
below the guardrail threshold, no RL). The vol filter answers an important
question: does the guardrail alone explain any gain, or did RL add something?

## Leakage controls

- Chronological split: train <= `--train_end`, validation <= `--valid_end`, test after.
- GARCH parameters are estimated on the training window only, then frozen and
  filtered forward. The feature is a one-step-ahead forecast, known at the close of day t.
- Feature scaler and guardrail threshold come from training data only.
- An action at the close of day t earns the NEXT day's return (no look-ahead).
- No model selection on the test window; each run is evaluated on test once.

## Commands (run from the project folder, inside your venv)

```powershell
# 1. Data
python data/fetch_data.py --ticker GC=F --start 2010-01-01 --end 2025-01-01 --out data/gold.csv

# 2. ATS diagnostics (also your Applied Time Series deliverable)
python ats/diagnostics.py --csv data/gold.csv --seasonal_period 252 --lags 40 --train_end 2019-12-31

# 3. One feature table for all experiments
python features/build_features.py --price_csv data/gold.csv --ats_csv results/ats_features.csv --out results/features_all.csv

# 4a. Baselines only (seconds, no training)
python run_experiments.py --features_csv results/features_all.csv --baselines_only

# 4b. Smoke test of training (a few minutes)
python run_experiments.py --features_csv results/features_all.csv --timesteps 3000 --seeds 1

# 4c. The real run: A, B, C x 5 seeds x 100k steps (budget roughly 30-60 minutes on CPU)
python run_experiments.py --features_csv results/features_all.csv --train_end 2019-12-31 --valid_end 2022-12-31 --timesteps 100000 --seeds 5
```

Outputs in `results/`: `results_table.md` (paste into your report),
`summary_test.csv`, `summary_valid.csv`, `runs.csv` (every individual run),
`test_equity_curves.png`, plus the ATS plots from step 2.

## Reading the results

- Compare A vs B (do ATS features help?) and B vs C (does the guardrail help?),
  and compare all of them with `vol filter (no RL)` and `buy & hold`.
- Look at mean AND spread across seeds. If the spread is as large as the gap
  between A, B and C, you cannot claim one is better.
- `p<0.05 vs B&H` counts seeds whose daily excess return over buy & hold passes
  a paired t-test. It is a rough guide (it ignores autocorrelation), not proof.
- Check `entries`, `in market %` and `cost drag %`. An agent that is 99% invested
  with one trade is just buy & hold.
- Gold rose strongly in 2020-2024. If your test window is a bull run, buy & hold
  is hard to beat and a falling-market result is missing. Try other splits
  (for example `--train_end 2016-12-31 --valid_end 2019-12-31`) and report both.

## Other tools

- `data/combine_intraday.py`, `data/resample_ohlc.py`: merge daily intraday
  downloads and resample to coarser timeframes (use one price type, BID or ASK, throughout).
- `data/synthetic_gold.csv`: synthetic data for testing the code path only. Never report it.

## Known limitations

- Long/flat only, one asset, daily bars, flat cost per trade.
- The GARCH recursion in `ats/diagnostics.py` assumes GARCH(1,1).
- The random-start training episodes (500 days) and PPO settings are
  reasonable defaults, not tuned.
- Not financial advice.

## v5: buy/sell chart, per-step log, and a Windows save-crash fix

New file `plotting.py`. `TradingEnv.to_frame()` now also logs `price` (a reconstructed,
display-only market price -- never used in the reward), the raw `action` before any
guardrail override, running `equity` and `drawdown`. `run_experiments.py` automatically
saves one chart per experiment after a normal run (not `--baselines_only`):
`results/backtest_<A|B|C>_seed0.png` -- three panels (equity vs buy & hold, drawdown,
price with green buy / red sell markers and an orange X wherever the guardrail overrode
the agent's action), the same layout used in the reference project's report. It's seed 0
only, one representative run, not all seeds -- check `results/runs.csv` for how seed 0's
numbers compare to the others before reading too much into any one chart.

This also fixes the crash at the end of your last run: saving a PNG while it's open in
another program raises `PermissionError` on Windows and used to kill the whole script
after training had already finished. Both plot-saving calls now go through
`plotting.safe_savefig()`, which catches that, warns, and saves under a fallback
filename instead -- so a locked file can no longer lose a completed run's results.

No command changes; just re-run as before and the new chart appears alongside the usual
outputs.

## v4: fixing overtrading, HAR-RV, and walk-forward

The 2021-2024 test run showed every experiment losing to buy-and-hold, with 55-60 trades
over ~1,000 days and `corr(entries, cost_drag) = 1.00` -- the agent was flipping position
far more than any edge could justify, which was eating returns regardless of which features
it saw (A, B and C all failed the same way). This points to the reward/training setup, not
to the ATS features, so three things changed:

1. **Turnover penalty.** `env/trading_env.py` and `run_experiments.py` take `--turnover_penalty`,
   an extra cost applied only to the training reward (never to the reported P&L) each time the
   agent trades. Use values on the scale of `0.0006`-`0.0012` with this reward implementation. The reward is multiplied by 100, so `0.1` is extremely large and can suppress trading. Compare `results/action_diagnostics.csv` as well as `entries`/`total trades`.
2. **HAR-RV feature option.** `features/build_features.py` takes `--har_csv`, pointing at the
   5-minute notebook's `results/ats_5m/daily_rv_features.csv` export, and uses its HAR-RV
   forecast (which beat GARCH on QLIKE with a significant Diebold-Mariano test) in place of the
   GARCH feature. It's written into the same `garch_cond_vol` column, so the environment and
   guardrail work unchanged -- only the number feeding them is better.
3. **Walk-forward evaluation.** `run_experiments.py --walkforward` trains on an expanding window
   and tests on each following period separately (`results/walkforward_table.md`), so one
   unusual window -- like the 2023-2024 gold rally the single-split test happened to land on --
   can't drive the whole conclusion.

### Commands

```powershell
# A: does the turnover penalty fix overtrading? (same split as your last run)
python run_experiments.py --features_csv results/features_all.csv --train_end 2016-12-31 --valid_end 2020-12-31 --timesteps 100000 --seeds 5 --turnover_penalty 0.0008

# B: rebuild features using HAR-RV instead of GARCH (needs the 5-min notebook's output first)
python features/build_features.py --price_csv data/gold.csv --ats_csv results/ats_features.csv --har_csv results/ats_5m/daily_rv_features.csv --out results/features_all_har.csv
python run_experiments.py --features_csv results/features_all_har.csv --train_end 2019-12-31 --valid_end 2022-12-31 --timesteps 100000 --seeds 5 --turnover_penalty 0.0008

# C: walk-forward across several market regimes instead of one fixed split
python run_experiments.py --features_csv results/features_all.csv --walkforward --wf_first_test_year 2016 --wf_fold_years 1 --wf_min_train_years 5 --timesteps 100000 --seeds 3 --turnover_penalty 0.0008
```

Run a small smoke test first, then the full experiment. The penalty is not guaranteed to improve returns; compare validation results and action diagnostics, and do not repeatedly tune against the final test period.

## v6: action accounting and diagnostic fixes

- Buy/sell markers now use actual position changes (`traded`) and executed actions, rather than every repeated raw BUY/SELL decision.
- Equity and drawdown curves are plotted on the date the next-day return is realized; price/action markers remain on the decision date.
- Reconstructed price index is normalized to 1.0 at the first observed close.
- `momentum_signal` now represents a causal 20-session directional price return (rather than lag-1 autocorrelation, which measures return persistence rather than momentum). Rebuild `results/features_all.csv` before the next run so the saved feature table matches the source code.
- Environment logs requested and executed actions, previous/current positions, decision date, and P&L date.
- `results/action_diagnostics.csv` reports requested/executed action counts, real entries/exits, redundant actions, time long/flat, and guardrail overrides for each seed and split.
- Sortino now uses downside deviation relative to a zero-return target.
- Default turnover penalty is `0.0008`, reflecting the scale explored in the actual sweep; the old `0.05-0.2` suggestion was inconsistent with the reward scaling and has been removed.
- Added a deterministic environment accounting test under `tests/`.

## Earlier changelog

- v3: train/valid/test split, train-only GARCH with one-step-ahead forecast,
  train-only scaler and guardrail threshold, baselines, multi-seed runner and
  automatic results table. `train.py` and `evaluate.py` were replaced by `run_experiments.py`.
- v2: fixed look-ahead in the environment (an action now earns the next day's return).
