"""Standalone PPO training + policy/action logger for a historical OHLCV CSV.

This is a self-contained teaching/smoke-test environment, not a drop-in replacement
for a project's existing custom trading environment. It logs real PPO training
metrics and real actions produced by the trained policy on the supplied CSV.

Expected input: CSV with Date and Close; Open/High/Low/Volume are optional.
Outputs: outputs_ppo_logger/training_updates.csv, action_log.csv, checkpoints/...
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def load_market(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path.resolve()}")
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    date_col = next((lower[k] for k in ("date", "datetime", "timestamp", "time") if k in lower), None)
    close_col = lower.get("close")
    if date_col is None or close_col is None:
        raise ValueError(f"CSV must contain Date/DateTime and Close columns. Found: {list(df.columns)}")
    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    out["Close"] = pd.to_numeric(df[close_col], errors="coerce")
    for name in ("Open", "High", "Low", "Volume"):
        col = lower.get(name.lower())
        out[name] = pd.to_numeric(df[col], errors="coerce") if col else np.nan
    out = out.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    if len(out) < 80:
        raise ValueError(f"Need at least 80 valid rows; found {len(out)}")
    # Fill optional OHLCV with safe values so observations are finite.
    out["Open"] = out["Open"].fillna(out["Close"])
    out["High"] = out["High"].fillna(out[["Open", "Close"]].max(axis=1))
    out["Low"] = out["Low"].fillna(out[["Open", "Close"]].min(axis=1))
    out["Volume"] = out["Volume"].fillna(0)
    close = out["Close"].astype(float)
    out["log_return"] = np.log(close).diff()
    out["return_5d"] = close.pct_change(5)
    out["return_10d"] = close.pct_change(10)
    out["volatility_10d"] = out["log_return"].rolling(10).std()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    # Scale OHLCV-free features only; keep them bounded and finite.
    out = out.replace([np.inf, -np.inf], np.nan)
    out["log_return"] = out["log_return"].fillna(0).clip(-0.2, 0.2)
    out["return_5d"] = out["return_5d"].fillna(0).clip(-0.5, 0.5)
    out["return_10d"] = out["return_10d"].fillna(0).clip(-0.7, 0.7)
    out["volatility_10d"] = out["volatility_10d"].fillna(0).clip(0, 0.2)
    out["rsi_14"] = out["rsi_14"].fillna(50).clip(0, 100)
    out = out.iloc[15:].reset_index(drop=True)
    return out


def check_dependencies():
    try:
        import gymnasium  # noqa: F401
        import stable_baselines3  # noqa: F401
    except ImportError as e:
        print("Missing RL dependency:", e)
        print("Install in your project's active venv with:")
        print("  python -m pip install stable-baselines3 gymnasium")
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Train PPO and log training updates + per-candle actions.")
    ap.add_argument("--data", default="gold(1).csv", help="Input market CSV path (default: gold(1).csv)")
    ap.add_argument("--timesteps", type=int, default=5000, help="PPO training timesteps; smoke test uses 1024")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="outputs_ppo_logger", help="Output folder")
    ap.add_argument("--smoke-test", action="store_true", help="Run a tiny real PPO training + action-log test")
    ap.add_argument("--check-only", action="store_true", help="Validate CSV and output paths without RL training")
    args = ap.parse_args()
    data_path = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    df = load_market(data_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"CSV OK: {data_path}")
    print(f"Rows usable: {len(df)} | Date range: {df.Date.min().date()} -> {df.Date.max().date()}")
    print(f"Close range: {df.Close.min():.4f} -> {df.Close.max():.4f}")
    print(f"Output folder: {out_dir}")
    if args.check_only:
        print("CHECK-ONLY PASS: data validation succeeded; no model was trained.")
        return 0
    if not check_dependencies():
        return 2

    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    features = ["log_return", "return_5d", "return_10d", "volatility_10d", "rsi_14"]
    class GoldLongFlatEnv(gym.Env):
        """Long/flat paper-trading environment: 0 HOLD, 1 BUY, 2 SELL (close long)."""
        metadata = {"render_modes": []}
        def __init__(self, frame):
            super().__init__()
            self.frame = frame.reset_index(drop=True)
            self.X = self.frame[features].to_numpy(dtype=np.float32)
            self.returns = self.frame["log_return"].to_numpy(dtype=np.float32)
            self.prices = self.frame["Close"].to_numpy(dtype=np.float64)
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(features)+1,), dtype=np.float32)
            self.reset()
        def _obs(self):
            return np.concatenate([self.X[self.i], np.array([self.position], dtype=np.float32)]).astype(np.float32)
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.i = 0
            self.position = 0
            self.equity = 1.0
            self.prev_equity = 1.0
            return self._obs(), {}
        def step(self, action):
            old_pos = self.position
            if action == 1: self.position = 1
            elif action == 2: self.position = 0
            # Exposure earns the next bar's return; a small turnover cost applies when position changes.
            next_i = min(self.i + 1, len(self.frame)-1)
            market_ret = float(self.returns[next_i])
            cost = 0.0005 if self.position != old_pos else 0.0
            reward = float(self.position * market_ret - cost)
            self.equity *= float(np.exp(np.clip(reward, -1, 1)))
            self.i = next_i
            terminated = self.i >= len(self.frame)-1
            truncated = False
            info = {"reward": reward, "position_before": old_pos, "position_after": self.position,
                    "market_return_next": market_ret, "transaction_cost": cost, "equity": self.equity}
            return self._obs(), reward, terminated, truncated, info

    class LoggingPPO(PPO):
        """PPO subclass that writes metrics after each actual optimizer update."""
        def __init__(self, *args, update_csv: Path, **kwargs):
            super().__init__(*args, **kwargs)
            self.update_csv = Path(update_csv)
            self.update_rows = []
        def train(self):
            super().train()
            vals = self.logger.name_to_value
            def get(k):
                v = vals.get(k)
                try: return float(v) if v is not None and np.isfinite(float(v)) else None
                except (TypeError, ValueError): return None
            row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "timesteps": int(self.num_timesteps),
                   "train/entropy_loss": get("train/entropy_loss"),
                   "train/policy_gradient_loss": get("train/policy_gradient_loss"),
                   "train/value_loss": get("train/value_loss"),
                   "train/approx_kl": get("train/approx_kl"),
                   "train/clip_fraction": get("train/clip_fraction"),
                   "train/explained_variance": get("train/explained_variance"),
                   "train/learning_rate": get("train/learning_rate")}
            self.update_rows.append(row)
            pd.DataFrame(self.update_rows).to_csv(self.update_csv, index=False)


    timesteps = 1024 if args.smoke_test else max(1024, args.timesteps)
    env = DummyVecEnv([lambda: Monitor(GoldLongFlatEnv(df))])
    model = LoggingPPO("MlpPolicy", env, seed=args.seed, verbose=1, device="cpu",
                       n_steps=256, batch_size=64, learning_rate=3e-4, gamma=0.99,
                       ent_coef=0.01, policy_kwargs={"net_arch": [64, 64]},
                       update_csv=out_dir / "training_updates.csv")
    print(f"Starting PPO: timesteps={timesteps}, seed={args.seed}")
    model.learn(total_timesteps=timesteps)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    model.save(str(checkpoint_dir / "ppo_gold_logger"))

    # Deterministic replay: log policy probabilities AND selected action for every usable candle.
    obs = env.reset()
    action_rows = []
    replay_env = GoldLongFlatEnv(df)
    obs, _ = replay_env.reset()
    for i in range(len(df)-1):
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        with __import__("torch").no_grad():
            dist = model.policy.get_distribution(obs_tensor).distribution
            probs = dist.probs.detach().cpu().numpy()[0]
        action, _ = model.predict(obs, deterministic=True)
        action = int(np.asarray(action).reshape(-1)[0])
        before = replay_env.position
        obs, reward, done, trunc, info = replay_env.step(action)
        action_rows.append({"date": df.loc[i, "Date"].isoformat(), "close": float(df.loc[i, "Close"]),
            "p_hold": float(probs[0]), "p_buy": float(probs[1]), "p_sell": float(probs[2]),
            "selected_action_id": action, "selected_action": ["HOLD", "BUY", "SELL"][action],
            "position_before": before, "position_after": replay_env.position, "reward": float(reward),
            "market_return_next": float(info["market_return_next"]), "transaction_cost": float(info["transaction_cost"]),
            "equity": float(info["equity"]), "checkpoint": "ppo_gold_logger_deterministic_replay"})
        if done: break
    pd.DataFrame(action_rows).to_csv(out_dir / "action_log.csv", index=False)
    metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "input_csv": str(data_path),
        "rows_used": int(len(df)), "features": features, "actions": {"0":"HOLD","1":"BUY/open long","2":"SELL/close long"},
        "training_timesteps_requested": timesteps, "seed": args.seed,
        "environment_note": "Standalone long/flat teaching environment; reward = position * next-row log return - 5 bps on position changes. Not the project's existing custom environment unless its reward/observation design matches.",
        "replay_note": "Action probabilities are from the saved trained policy; replay uses deterministic argmax action selection."}
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nTRAINING PASS")
    print(f"Training updates CSV: {out_dir / 'training_updates.csv'}")
    print(f"Action/probability CSV: {out_dir / 'action_log.csv'}")
    print(f"Saved model: {checkpoint_dir / 'ppo_gold_logger.zip'}")
    print(f"Rows logged: {len(action_rows)}")
    print("NOTE: this is a standalone demonstration environment, not automatically identical to the main project backtest.")
    return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
