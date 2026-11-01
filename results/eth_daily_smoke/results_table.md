# Results

Split: train <= 2021-12-31, validation <= 2023-12-31, test after.
Cost per position change: 5 bps. Guardrail threshold: 0.06338 (train 85% quantile).
PPO timesteps: 1024, seeds: 1. Values are mean ± std across seeds.

## TEST (final exam, evaluated once)

| strategy | runs | return % | Sharpe | max DD % | Sortino | entries | total trades | in market % | cost drag % | p<0.05 vs B&H |
|---|---|---|---|---|---|---|---|---|---|---|
| flat (cash) | 1 | 0.0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 0 | 0.0 | - |
| buy & hold | 1 | -2.4 | -0.01 | -67.6 | -0.02 | 1.0 | 1.0 | 100 | 0.0 | - |
| momentum (10d>0) | 1 | 24.3 | 0.14 | -63.6 | 0.16 | 83.0 | 166.0 | 48 | 8.0 | - |
| vol filter (no RL) | 1 | -1.9 | -0.01 | -67.6 | -0.01 | 2.0 | 3.0 | 100 | 0.1 | - |
| A | 1 | 56.5 | 0.20 | -65.9 | 0.30 | 16.0 | 31.0 | 97 | 1.5 | 1/1 |
| B | 1 | 30.8 | 0.13 | -63.5 | 0.18 | 14.0 | 27.0 | 85 | 1.3 | 0/1 |
| C | 1 | 48.8 | 0.20 | -58.7 | 0.27 | 9.0 | 17.0 | 87 | 0.8 | 0/1 |

## VALIDATION

| strategy | runs | return % | Sharpe | max DD % | Sortino | entries | total trades | in market % | cost drag % | p<0.05 vs B&H |
|---|---|---|---|---|---|---|---|---|---|---|
| flat (cash) | 1 | 0.0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 0 | 0.0 | - |
| buy & hold | 1 | -39.5 | -0.30 | -74.1 | -0.38 | 1.0 | 1.0 | 100 | 0.0 | - |
| momentum (10d>0) | 1 | -31.7 | -0.34 | -57.4 | -0.36 | 51.0 | 101.0 | 50 | 4.9 | - |
| vol filter (no RL) | 1 | -45.1 | -0.37 | -73.1 | -0.45 | 4.0 | 7.0 | 95 | 0.3 | - |
| A | 1 | -16.0 | -0.11 | -67.5 | -0.13 | 11.0 | 21.0 | 98 | 1.0 | 0/1 |
| B | 1 | -45.1 | -0.41 | -71.9 | -0.47 | 13.0 | 25.0 | 84 | 1.2 | 0/1 |
| C | 1 | -22.2 | -0.18 | -69.2 | -0.20 | 9.0 | 17.0 | 76 | 0.8 | 0/1 |
