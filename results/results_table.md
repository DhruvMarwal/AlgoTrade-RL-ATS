# Results

Split: train <= 2019-12-31, validation <= 2022-12-31, test after.
Cost per position change: 5 bps. Guardrail threshold: 0.01225 (train 85% quantile).
PPO timesteps: 3000, seeds: 1. Values are mean ± std across seeds.

## TEST (final exam, evaluated once)

| strategy | runs | return % | Sharpe | max DD % | Sortino | entries | total trades | in market % | cost drag % | p<0.05 vs B&H |
|---|---|---|---|---|---|---|---|---|---|---|
| flat (cash) | 1 | 0.0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 0 | 0.0 | - |
| buy & hold | 1 | 43.0 | 1.25 | -10.9 | 1.74 | 1.0 | 1.0 | 100 | 0.0 | - |
| momentum (10d>0) | 1 | 6.5 | 0.27 | -11.0 | 0.30 | 41.0 | 82.0 | 58 | 4.0 | - |
| vol filter (no RL) | 1 | 43.0 | 1.25 | -10.9 | 1.74 | 1.0 | 1.0 | 100 | 0.0 | - |
| A | 1 | 43.4 | 1.26 | -10.9 | 1.75 | 1.0 | 1.0 | 100 | 0.0 | 0/1 |
| B | 1 | 40.2 | 1.28 | -10.9 | 1.69 | 2.0 | 4.0 | 85 | 0.2 | 0/1 |
| C | 1 | 43.0 | 1.25 | -10.9 | 1.74 | 1.0 | 1.0 | 100 | 0.0 | 0/1 |

## VALIDATION

| strategy | runs | return % | Sharpe | max DD % | Sortino | entries | total trades | in market % | cost drag % | p<0.05 vs B&H |
|---|---|---|---|---|---|---|---|---|---|---|
| flat (cash) | 1 | 0.0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 0 | 0.0 | - |
| buy & hold | 1 | 19.7 | 0.34 | -21.2 | 0.45 | 1.0 | 1.0 | 100 | 0.0 | - |
| momentum (10d>0) | 1 | -12.6 | -0.35 | -26.4 | -0.34 | 69.0 | 137.0 | 54 | 6.6 | - |
| vol filter (no RL) | 1 | 1.7 | 0.04 | -24.6 | 0.04 | 8.0 | 15.0 | 82 | 0.7 | - |
| A | 1 | 12.0 | 0.25 | -19.9 | 0.29 | 9.0 | 17.0 | 88 | 0.8 | 0/1 |
| B | 1 | 9.7 | 0.24 | -20.2 | 0.24 | 6.0 | 11.0 | 64 | 0.5 | 0/1 |
| C | 1 | 3.2 | 0.07 | -23.4 | 0.08 | 7.0 | 13.0 | 82 | 0.6 | 0/1 |
