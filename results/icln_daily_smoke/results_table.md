# Results

Split: train <= 2019-12-31, validation <= 2022-12-31, test after.
Cost per position change: 5 bps. Guardrail threshold: 0.02291 (train 85% quantile).
PPO timesteps: 1024, seeds: 1. Values are mean ± std across seeds.

## TEST (final exam, evaluated once)

| strategy | runs | return % | Sharpe | max DD % | Sortino | entries | total trades | in market % | cost drag % | p<0.05 vs B&H |
|---|---|---|---|---|---|---|---|---|---|---|
| flat (cash) | 1 | 0.0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 0 | 0.0 | - |
| buy & hold | 1 | -17.3 | -0.19 | -49.0 | -0.29 | 1.0 | 1.0 | 100 | 0.0 | - |
| momentum (10d>0) | 1 | -22.9 | -0.36 | -38.7 | -0.39 | 67.0 | 134.0 | 51 | 6.5 | - |
| vol filter (no RL) | 1 | -4.9 | -0.05 | -47.9 | -0.08 | 8.0 | 15.0 | 95 | 0.7 | - |
| A | 1 | 14.7 | 0.17 | -40.6 | 0.21 | 36.0 | 71.0 | 74 | 3.5 | 0/1 |
| B | 1 | -17.3 | -0.19 | -49.0 | -0.29 | 1.0 | 1.0 | 100 | 0.0 | 0/1 |
| C | 1 | -20.3 | -0.25 | -49.4 | -0.35 | 7.0 | 13.0 | 88 | 0.6 | 0/1 |

## VALIDATION

| strategy | runs | return % | Sharpe | max DD % | Sortino | entries | total trades | in market % | cost drag % | p<0.05 vs B&H |
|---|---|---|---|---|---|---|---|---|---|---|
| flat (cash) | 1 | 0.0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 0 | 0.0 | - |
| buy & hold | 1 | 73.3 | 0.48 | -49.1 | 0.65 | 1.0 | 1.0 | 100 | 0.0 | - |
| momentum (10d>0) | 1 | 144.3 | 1.22 | -28.5 | 1.50 | 46.0 | 92.0 | 57 | 4.5 | - |
| vol filter (no RL) | 1 | 46.7 | 0.50 | -50.6 | 0.66 | 22.0 | 43.0 | 68 | 2.1 | - |
| A | 1 | -26.9 | -0.37 | -46.1 | -0.32 | 26.0 | 51.0 | 48 | 2.5 | 0/1 |
| B | 1 | 66.1 | 0.46 | -46.8 | 0.59 | 4.0 | 7.0 | 93 | 0.3 | 0/1 |
| C | 1 | -23.7 | -0.44 | -45.3 | -0.47 | 14.0 | 27.0 | 44 | 1.3 | 0/1 |
