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
| A | 3 | 17.6 ± 16.5 | 0.44 ± 0.07 | -11.5 ± 10.0 | 0.36 ± 0.33 | 20.3 ± 17.0 | 40.0 ± 33.5 | 47 ± 43 | 2.0 ± 1.6 | 0/3 |

## VALIDATION

| strategy | runs | return % | Sharpe | max DD % | Sortino | entries | total trades | in market % | cost drag % | p<0.05 vs B&H |
|---|---|---|---|---|---|---|---|---|---|---|
| flat (cash) | 1 | 0.0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 0 | 0.0 | - |
| buy & hold | 1 | 63.2 | 0.86 | -14.2 | 1.14 | 1.0 | 1.0 | 100 | 0.0 | - |
| momentum (10d>0) | 1 | -5.0 | -0.12 | -20.0 | -0.12 | 81.0 | 161.0 | 58 | 7.7 | - |
| vol filter (no RL) | 1 | 36.8 | 0.64 | -15.4 | 0.76 | 4.0 | 7.0 | 91 | 0.3 | - |
| A | 3 | 40.7 ± 26.7 | 0.82 ± 0.19 | -11.1 ± 5.8 | 0.83 ± 0.56 | 16.3 ± 13.6 | 32.3 ± 26.7 | 57 ± 48 | 1.6 ± 1.3 | 0/3 |
