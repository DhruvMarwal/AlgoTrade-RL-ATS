# Results

Split: train <= 2019-12-31, validation <= 2022-12-31, test after.
Cost per position change: 5 bps. Guardrail threshold: 0.06782 (train 85% quantile).
PPO timesteps: 3000, seeds: 1. Values are mean ± std across seeds.

## TEST (final exam, evaluated once)

| strategy | runs | return % | Sharpe | max DD % | Sortino | entries | total trades | in market % | cost drag % | p<0.05 vs B&H |
|---|---|---|---|---|---|---|---|---|---|---|
| flat (cash) | 1 | 0.0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 0 | 0.0 | - |
| buy & hold | 1 | 177.9 | 0.76 | -45.3 | 1.15 | 1.0 | 1.0 | 100 | 0.0 | - |
| momentum (10d>0) | 1 | 116.4 | 0.79 | -33.9 | 0.98 | 48.0 | 96.0 | 53 | 4.7 | - |
| vol filter (no RL) | 1 | 177.9 | 0.76 | -45.3 | 1.15 | 1.0 | 1.0 | 100 | 0.0 | - |
| A | 1 | -3.5 | -0.60 | -3.5 | -0.03 | 1.0 | 2.0 | 0 | 0.1 | 0/1 |
| B | 1 | 130.2 | 0.95 | -29.1 | 1.11 | 16.0 | 32.0 | 58 | 1.6 | 0/1 |
| C | 1 | 139.7 | 0.87 | -24.7 | 1.13 | 43.0 | 86.0 | 65 | 4.2 | 0/1 |

## VALIDATION

| strategy | runs | return % | Sharpe | max DD % | Sortino | entries | total trades | in market % | cost drag % | p<0.05 vs B&H |
|---|---|---|---|---|---|---|---|---|---|---|
| flat (cash) | 1 | 0.0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 0 | 0.0 | - |
| buy & hold | 1 | 814.2 | 0.61 | -79.3 | 0.74 | 1.0 | 1.0 | 100 | 0.0 | - |
| momentum (10d>0) | 1 | 1117.0 | 0.99 | -60.2 | 1.17 | 70.0 | 140.0 | 59 | 6.8 | - |
| vol filter (no RL) | 1 | 572.3 | 0.58 | -79.3 | 0.66 | 6.0 | 11.0 | 90 | 0.5 | - |
| A | 1 | 7.4 | 0.48 | -0.1 | 0.00 | 1.0 | 2.0 | 0 | 0.1 | 0/1 |
| B | 1 | 642.7 | 1.32 | -50.7 | 1.25 | 31.0 | 61.0 | 33 | 3.0 | 0/1 |
| C | 1 | 552.4 | 1.03 | -56.4 | 0.97 | 66.0 | 131.0 | 37 | 6.3 | 0/1 |
