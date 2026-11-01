"""Evaluate an already-trained PPO model without retraining it.

Example:
  python evaluate_saved_model.py --model_dir models/seed_selection/selected/B --features_csv results/features_all.csv --split test
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from run_seed_selection import make_env, metrics, run_policy



def run_policy_with_action_probs(env, model):
    """Run saved PPO policy and log action probabilities at each decision date."""
    env.reset()
    probability_rows = []
    done = False
    while not done:
        obs = env._obs()
        decision_date = pd.Timestamp(env.dates[env.t])
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        probs = model.policy.get_distribution(obs_tensor).distribution.probs
        probs = probs.detach().cpu().numpy().reshape(-1)
        action, _ = model.predict(obs, deterministic=True)
        action = int(np.asarray(action).reshape(-1)[0])
        probability_rows.append({
            "date": decision_date,
            "policy_prob_hold": float(probs[0]),
            "policy_prob_buy": float(probs[1]),
            "policy_prob_sell": float(probs[2]),
            "policy_action": action,
        })
        _, _, done, _, _ = env.step(action)
    frame = env.to_frame()
    probability_frame = pd.DataFrame(probability_rows).set_index("date")
    probability_frame.index.name = frame.index.name or "Date"
    return frame.join(probability_frame, how="left")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--features_csv', default='results/features_all.csv')
    ap.add_argument('--split', choices=['valid', 'test', 'all_after_train'], default='test')
    ap.add_argument('--out_dir', default='results/saved_model_evaluation')
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    metadata_path = model_dir / 'metadata.json'
    model_path = model_dir / 'model.zip'
    scaler_path = model_dir / 'scaler.pkl'
    for p in (metadata_path, model_path, scaler_path):
        if not p.exists():
            raise SystemExit(f'Missing required artifact: {p}')

    meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    model = PPO.load(str(model_path), device='cpu')
    df = pd.read_csv(args.features_csv, index_col='Date', parse_dates=True).sort_index()

    train_end = pd.Timestamp(meta['train_end'])
    valid_end = pd.Timestamp(meta['valid_end'])
    if args.split == 'valid':
        frame = df[(df.index > train_end) & (df.index <= valid_end)]
    elif args.split == 'test':
        frame = df[df.index > valid_end]
    else:
        frame = df[df.index > train_end]
    if len(frame) < 2:
        raise SystemExit(f'Not enough rows for split={args.split}: {len(frame)}')

    env = make_env(frame, scaler, meta['feature_cols'], bool(meta['guardrail']),
                   float(meta['guardrail_vol_threshold']), float(meta['transaction_cost']))
    result = run_policy_with_action_probs(env, model)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{meta['experiment']}_seed{meta['seed']}_{args.split}"
    result.to_csv(out / f'{tag}_daily_log.csv', index=True, index_label='Date')
    m = metrics(result)
    pd.DataFrame([{'experiment': meta['experiment'], 'seed': meta['seed'], 'split': args.split, **m}]).to_csv(
        out / f'{tag}_metrics.csv', index=False)
    equity = np.exp(np.cumsum(result['strategy_logret'].values))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(result.index, equity, label=f"{meta['experiment']} seed {meta['seed']}")
    ax.set_title(f"Saved PPO model — {args.split} period")
    ax.set_ylabel('Equity (start = 1.0)')
    ax.tick_params(axis='x', rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f'{tag}_equity.png', dpi=140)
    plt.close(fig)
    print('Loaded saved model; no training performed.')
    print(pd.DataFrame([m]).to_string(index=False))
    print(f'Saved logs and plot to: {out.resolve()}')

if __name__ == '__main__':
    main()
