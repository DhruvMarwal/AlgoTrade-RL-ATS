# Results

Split: train <= 2016-12-31, validation <= 2020-12-31, test after.
Cost per position change: 5 bps. Guardrail threshold: 0.01306 (train 85% quantile).
PPO timesteps: 15000, seeds: 3. Values are mean ± std across seeds.

## TEST (final exam, evaluated once)

| strategy | runs | return % | Sharpe | max DD % | Sortino | entries | total trades | in market % | cost drag % | p<0.05 vs B&H |
|---|---|---|---|---|---|---|---|---|---|---|
| flat (cash) | 1 | 0.0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 0 | 0.0 | - |
| buy & hold | 1 | 35.6 | 0.51 | -20.2 | 0.70 | 1.0 | 1.0 | 100 | 0.0 | - |
| momentum (10d>0) | 1 | -7.9 | -0.18 | -15.8 | -0.19 | 82.0 | 164.0 | 54 | 7.9 | - |
| vol filter (no RL) | 1 | 34.8 | 0.50 | -20.2 | 0.69 | 2.0 | 3.0 | 100 | 0.1 | - |
| A | 3 | 22.9 ± 21.7 | 0.41 ± 0.39 | -15.8 ± 2.0 | 0.47 ± 0.45 | 22.7 ± 10.2 | 44.3 ± 20.4 | 57 ± 20 | 2.2 ± 1.0 | 0/3 |

## VALIDATION

| strategy | runs | return % | Sharpe | max DD % | Sortino | entries | total trades | in market % | cost drag % | p<0.05 vs B&H |
|---|---|---|---|---|---|---|---|---|---|---|
| flat (cash) | 1 | 0.0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 0 | 0.0 | - |
| buy & hold | 1 | 63.2 | 0.86 | -14.2 | 1.14 | 1.0 | 1.0 | 100 | 0.0 | - |
| momentum (10d>0) | 1 | -5.0 | -0.12 | -20.0 | -0.12 | 81.0 | 161.0 | 58 | 7.7 | - |
| vol filter (no RL) | 1 | 36.8 | 0.64 | -15.4 | 0.76 | 4.0 | 7.0 | 91 | 0.3 | - |
| A | 3 | 31.4 ± 21.0 | 0.59 ± 0.26 | -13.4 ± 1.8 | 0.62 ± 0.42 | 13.0 ± 8.9 | 26.0 ± 17.8 | 58 ± 35 | 1.3 ± 0.9 | 0/3 |
