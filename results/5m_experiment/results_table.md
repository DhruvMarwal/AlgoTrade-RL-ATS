# Results

Split: train <= 2023-12-31, validation <= 2024-12-31, test after.
Cost per position change: 5 bps. Guardrail threshold: 0.00060 (train 85% quantile).
PPO timesteps: 60000, seeds: 3. Values are mean ± std across seeds.

## TEST (final exam, evaluated once)

| strategy | runs | return % | Sharpe | max DD % | Sortino | entries | total trades | in market % | cost drag % | p<0.05 vs B&H |
|---|---|---|---|---|---|---|---|---|---|---|
| flat (cash) | 1 | 0.0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 0 | 0.0 | - |
| buy & hold | 1 | 47.4 | 0.19 | -9.0 | 0.39 | 1.0 | 1.0 | 100 | 0.0 | - |
| momentum (10d>0) | 1 | -80.1 | -0.85 | -82.6 | -1.76 | 2020.0 | 4039.0 | 53 | 86.7 | - |
| vol filter (no RL) | 1 | -7.4 | -0.05 | -21.4 | -0.09 | 359.0 | 718.0 | 72 | 30.2 | - |
| A | 3 | 31.6 ± 27.3 | 0.13 ± 0.11 | -6.0 ± 5.2 | 0.26 ± 0.23 | 5.0 ± 7.8 | 9.3 ± 15.3 | 66 ± 57 | 0.5 ± 0.8 | 1/3 |
| B | 3 | 27.1 ± 24.3 | 0.11 ± 0.10 | -6.9 ± 3.7 | 0.22 ± 0.20 | 6.0 ± 8.7 | 11.3 ± 17.9 | 59 ± 49 | 0.6 ± 0.9 | 2/3 |

## VALIDATION

| strategy | runs | return % | Sharpe | max DD % | Sortino | entries | total trades | in market % | cost drag % | p<0.05 vs B&H |
|---|---|---|---|---|---|---|---|---|---|---|
| flat (cash) | 1 | 0.0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 0 | 0.0 | - |
| buy & hold | 1 | 26.1 | 0.10 | -9.0 | 0.13 | 1.0 | 1.0 | 100 | 0.0 | - |
| momentum (10d>0) | 1 | -99.5 | -2.91 | -99.5 | -3.42 | 5430.0 | 10860.0 | 52 | 99.6 | - |
| vol filter (no RL) | 1 | -27.6 | -0.19 | -28.5 | -0.23 | 658.0 | 1315.0 | 80 | 48.2 | - |
| A | 3 | 17.3 ± 15.0 | 0.07 ± 0.06 | -6.0 ± 5.2 | 0.08 ± 0.07 | 8.7 ± 14.2 | 16.7 ± 28.0 | 66 ± 58 | 0.8 ± 1.4 | 0/3 |
| B | 3 | 17.3 ± 15.5 | 0.06 ± 0.06 | -7.5 ± 2.5 | 0.08 ± 0.08 | 8.7 ± 13.3 | 16.7 ± 27.1 | 68 ± 55 | 0.8 ± 1.3 | 0/3 |
