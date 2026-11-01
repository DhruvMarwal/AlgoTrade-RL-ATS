"""Preflight checks for the RL trading seed-selection pipeline.
Run from project root after the short smoke test, before any long run.
This does not train a model. It validates source syntax, data schema, saved artifacts,
checkpoint presence, model/scaler compatibility, and reload evaluation.
"""
from __future__ import annotations
import json
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
FEATURES = ROOT / 'results' / 'features_all.csv'
RUNNER = ROOT / 'run_seed_selection.py'
EVALUATOR = ROOT / 'evaluate_saved_model.py'
SELECTED = ROOT / 'models' / 'seed_selection' / 'selected' / 'B'
FAILURES: list[str] = []
PASSES: list[str] = []

def check(ok: bool, passed: str, failed: str) -> None:
    (PASSES if ok else FAILURES).append(('PASS: ' if ok else 'FAIL: ') + (passed if ok else failed))

def main() -> int:
    print('=== RL TRADING PIPELINE PREFLIGHT (no training) ===')
    # Source syntax and imports
    import py_compile
    for path in (RUNNER, EVALUATOR):
        if path.exists():
            try:
                py_compile.compile(str(path), doraise=True)
                check(True, f'{path.name} compiles', f'{path.name} does not compile')
            except Exception as exc:
                check(False, '', f'{path.name} compile error: {exc}')
        else:
            check(False, '', f'missing {path}')
    if RUNNER.exists():
        src = RUNNER.read_text(encoding='utf-8')
        check('import pickle' in src, 'runner imports pickle', 'runner is missing import pickle')
        check('import shutil' in src, 'runner imports shutil', 'runner is missing import shutil')
        check('CheckpointCallback' in src, 'runner implements periodic checkpoints', 'runner has no CheckpointCallback')

    # Required Python packages
    try:
        import numpy as np
        import pandas as pd
        import gymnasium
        import sklearn
        from stable_baselines3 import PPO
        check(True, 'core Python packages import', 'core package imports failed')
    except Exception as exc:
        check(False, '', f'core dependency import failed: {type(exc).__name__}: {exc}')
        print_results()
        return 1

    # Feature CSV schema / chronology
    if not FEATURES.exists():
        check(False, '', f'missing feature CSV: {FEATURES}')
    else:
        try:
            df = pd.read_csv(FEATURES, parse_dates=['Date'])
            required = {'Date', 'log_return', 'return_5d', 'return_10d', 'rolling_vol_10d',
                        'rsi_14', 'garch_cond_vol', 'regime_flag', 'momentum_signal'}
            missing = sorted(required - set(df.columns))
            check(not missing, 'feature CSV has all required columns', f'feature CSV missing columns: {missing}')
            check(len(df) >= 1000, f'feature CSV has {len(df)} rows', f'feature CSV unexpectedly short: {len(df)} rows')
            check(df['Date'].is_monotonic_increasing, 'feature dates are sorted', 'feature dates are not sorted')
            check(not df['Date'].duplicated().any(), 'feature dates are unique', 'feature CSV has duplicate dates')
            numeric = [c for c in required if c != 'Date' and c in df.columns]
            check(not df[numeric].isna().any().any(), 'feature columns have no missing values', 'feature CSV contains missing numeric feature values')
            check(df[numeric].apply(lambda col: np.isfinite(col.to_numpy(dtype=float)).all()).all(),
                  'feature columns contain finite values', 'feature CSV contains inf/-inf values')
        except Exception as exc:
            check(False, '', f'feature CSV read/validation failed: {type(exc).__name__}: {exc}')

    # Artifacts from the already completed B smoke test
    required_files = [SELECTED / 'model.zip', SELECTED / 'scaler.pkl', SELECTED / 'metadata.json']
    for p in required_files:
        check(p.is_file() and p.stat().st_size > 0, f'artifact exists: {p.relative_to(ROOT)}', f'missing/empty artifact: {p.relative_to(ROOT)}')
    checkpoint_dir = SELECTED / 'checkpoints'
    checkpoints = list(checkpoint_dir.glob('*.zip')) if checkpoint_dir.exists() else []
    check(bool(checkpoints), f'{len(checkpoints)} PPO checkpoint(s) found',
          'no PPO checkpoint files found in selected/B/checkpoints; interruption recovery is not verified')

    if all(p.is_file() for p in required_files):
        try:
            meta = json.loads((SELECTED / 'metadata.json').read_text(encoding='utf-8'))
            with open(SELECTED / 'scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            model = PPO.load(str(SELECTED / 'model.zip'), device='cpu')
            feature_count = len(meta['feature_cols'])
            # Validate the custom project scaler by actually transforming a correctly
            # shaped one-row DataFrame; do not assume sklearn-only attributes exist.
            probe = pd.DataFrame([[0.0] * feature_count], columns=meta['feature_cols'])
            transformed = scaler.transform(probe)
            transformed_shape = getattr(transformed, 'shape', None)
            obs_shape = tuple(model.observation_space.shape)
            check(transformed_shape == (1, feature_count),
                  f'scaler transforms {feature_count} features correctly',
                  f'scaler output shape mismatch: {transformed_shape}, expected {(1, feature_count)}')
            check(obs_shape == (feature_count + 1,),
                  f'model observation shape {obs_shape} matches {feature_count} features + position',
                  f'model observation shape mismatch: shape={obs_shape}, expected={(feature_count+1,)}')
            check(meta.get('experiment') == 'B', 'metadata identifies Experiment B', f'unexpected model experiment: {meta.get("experiment")}')
            check(meta.get('train_end') and meta.get('valid_end'), 'metadata has chronological split dates', 'metadata missing train_end/valid_end')
        except Exception as exc:
            check(False, '', f'model/scaler/metadata compatibility failed: {type(exc).__name__}: {exc}')

    # Ensure output artifacts created by the smoke test are present
    expected_results = [ROOT/'results'/'best_seed_selection.csv', ROOT/'results'/'backtest_B_seed0.png',
                        ROOT/'results'/'summary_test.csv', ROOT/'results'/'summary_valid.csv']
    for p in expected_results:
        check(p.is_file() and p.stat().st_size > 0, f'output exists: {p.relative_to(ROOT)}', f'missing/empty output: {p.relative_to(ROOT)}')

    # Re-run the evaluator on the saved model; this is evaluation only, no training.
    if EVALUATOR.exists() and (SELECTED/'model.zip').exists() and (SELECTED/'metadata.json').exists() and (SELECTED/'scaler.pkl').exists():
        cmd = [sys.executable, str(EVALUATOR), '--model_dir', str(SELECTED), '--features_csv', str(FEATURES), '--split', 'test', '--out_dir', str(ROOT/'results'/'preflight_saved_model_evaluation')]
        try:
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=180)
            good = proc.returncode == 0 and 'Loaded saved model; no training performed.' in proc.stdout
            check(good, 'saved-model evaluator reloads and runs successfully',
                  f'evaluator failed (exit {proc.returncode}); stdout={proc.stdout[-1200:]}; stderr={proc.stderr[-1200:]}')
        except Exception as exc:
            check(False, '', f'evaluator smoke test failed: {type(exc).__name__}: {exc}')
    else:
        check(False, '', 'cannot run evaluator smoke test because required files are missing')

    print_results()
    return 1 if FAILURES else 0

def print_results() -> None:
    print('\n'.join(PASSES))
    print('\n'.join(FAILURES))
    print(f'\nSummary: {len(PASSES)} passed, {len(FAILURES)} failed.')
    if FAILURES:
        print('DO NOT start the long training run until every FAIL is resolved.')
    else:
        print('Smoke-test artifacts passed preflight. This verifies the current artifacts and reload path, not every future seed run.')

if __name__ == '__main__':
    raise SystemExit(main())
