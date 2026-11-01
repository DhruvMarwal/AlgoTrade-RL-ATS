RL Trading Project — targeted methodology update

Files in this patch:
- run_experiments.py: walk-forward mode now refits GARCH parameters for each fold using returns only through that fold's training cutoff, then rebuilds the causal GARCH volatility and trailing regime features. Fixed-split A/B/C runs are otherwise unchanged.
- evaluation.py: paired_pvalue now uses a Newey-West/HAC standard error for daily excess log returns when statsmodels HAC is available. This is a more robust diagnostic than the previous plain t-test, but it is not definitive proof and does not correct for multiple-seed testing/model selection.
- env/trading_env.py, features/build_features.py, plotting.py: included as matching copies of the files supplied for review; no intentional changes to these three files.

How to install safely:
1. Back up your existing run_experiments.py and evaluation.py.
2. Extract this ZIP into the ROOT of your existing project folder, preserving the folder structure and allowing replacement of these files.
3. Keep your existing data/, ats/, features/scaling.py, results/, models/ and other project files. This ZIP is a targeted code patch, not a complete project with your data.

Recommended next test (B only, walk-forward):
python run_experiments.py --features_csv results/features_all.csv --walkforward --wf_first_test_year 2016 --wf_fold_years 1 --wf_min_train_years 5 --timesteps 60000 --seeds 3 --turnover_penalty 0.0008 --experiments B

This will train B on expanding windows and test each following year. It can take a while: many folds x 3 seeds. For a quick smoke test first, use --timesteps 3000 --seeds 1.

Important:
- Walk-forward mode requires the `arch` package (used by your ATS/GARCH workflow). If missing, install it in your existing .venv with: pip install arch
- The walk-forward output is saved under results/walkforward_runs.csv, results/walkforward_per_fold.csv, and results/walkforward_table.md.
- Fixed-split A/B/C outputs and existing results are not automatically deleted by walk-forward mode.
- Keep results_table.md from A/B/C backed up. Walk-forward uses separate filenames.
