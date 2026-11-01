# RL Trading Project update

## Files
- `run_experiments.py`: updated PPO A/B/C runner with additional diagnostics.
- `env/trading_env.py`: logs every environment reset's episode start/end date and index.
- `run_q_learning.py`: syllabus baseline using tabular Q-learning + epsilon-greedy.
- `evaluation.py`, `plotting.py`: supporting files from the latest project version available in this conversation.

## Apply update
Back up your current `run_experiments.py` and `env/trading_env.py`. Copy the files in this bundle into the matching locations in your existing project. Keep your own `features/` directory intact; the bundle's `features/__init__.py` is only a placeholder and is not a replacement for your existing feature modules.

## First, run a quick PPO diagnostic pass
```powershell
python run_experiments.py --features_csv results/features_all.csv --train_end 2016-12-31 --valid_end 2020-12-31 --timesteps 10000 --seeds 2 --turnover_penalty 0.0008 --experiments B
```
This writes `results/episode_starts.csv`, `results/training_progress.csv`, `results/seed_action_history.csv`, and `results/seed_summary.csv` along with the usual reports. Diagnostic files are cleared at each new run to avoid mixing different configurations; copy prior files to a dated folder before rerunning.

## Then test Q-learning
```powershell
python run_q_learning.py --features_csv results/features_all.csv --train_end 2016-12-31 --valid_end 2020-12-31 --seeds 3 --episodes 150 --episode_len 500
```
This writes `q_learning_runs.csv`, `q_learning_summary.csv`, `q_learning_episode_starts.csv`, `q_learning_training_progress.csv`, and `q_learning_action_history.csv` under `results/`.

Q-learning uses quantile-discretized `return_5d`, `rolling_vol_10d`, `rsi_14`, and `momentum_signal`; bin edges are fitted on training data only. Evaluation is greedy (no exploration). It is a baseline, not guaranteed to outperform PPO.

## Research caution
The 2021–2024 test window has already been inspected. Use validation for tuning, and treat further results on that test window as exploratory. Use a fresh chronological holdout for a stronger final assessment if data is available.

This update intentionally leaves PPO's current three-action semantics unchanged. Test action-space changes as a separate experiment so we can attribute performance changes correctly.
