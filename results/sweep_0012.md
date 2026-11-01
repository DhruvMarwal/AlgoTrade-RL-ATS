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
| A | 3 | 17.6 ± 22.4 | 0.37 ± 0.35 | -9.2 ± 9.5 | 0.34 ± 0.44 | 4.0 ± 2.6 | 7.7 ± 5.7 | 36 ± 53 | 0.4 ± 0.3 | 0/3 |

## VALIDATION

| strategy | runs | return % | Sharpe | max DD % | Sortino | entries | total trades | in market % | cost drag % | p<0.05 vs B&H |
|---|---|---|---|---|---|---|---|---|---|---|
| flat (cash) | 1 | 0.0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 0 | 0.0 | - |
| buy & hold | 1 | 63.2 | 0.86 | -14.2 | 1.14 | 1.0 | 1.0 | 100 | 0.0 | - |
| momentum (10d>0) | 1 | -5.0 | -0.12 | -20.0 | -0.12 | 81.0 | 161.0 | 58 | 7.7 | - |
| vol filter (no RL) | 1 | 36.8 | 0.64 | -15.4 | 0.76 | 4.0 | 7.0 | 91 | 0.3 | - |
| A | 3 | 16.9 ± 13.8 | 0.59 ± 0.27 | -6.3 ± 6.4 | 0.33 ± 0.34 | 5.3 ± 3.2 | 10.3 ± 6.7 | 27 ± 45 | 0.5 ± 0.3 | 0/3 |
