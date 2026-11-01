
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from env.trading_env import TradingEnv
from prepare_external_features import main as _unused  # existing project module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text())
    cols = metadata["feature_cols"]

    df = pd.read_csv(
        args.features_csv, index_col="Date", parse_dates=True
    ).sort_index()

    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    df = df.loc[df.index >= pd.Timestamp(args.start_date)].copy()

    if len(df) < 2:
        raise ValueError("Need at least two unseen daily feature rows.")

    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    model = PPO.load(str(model_dir / "model.zip"), device="cpu")

    scaled = scaler.transform(df[cols])

    guardrail = bool(metadata.get("guardrail", False))
    env = TradingEnv(
        obs_features=scaled,
        returns=df["log_return"].values,
        guard_vol=df["garch_cond_vol"].values if guardrail else None,
        vol_threshold=metadata["guardrail_vol_threshold"] if guardrail else None,
        transaction_cost=float(metadata["transaction_cost"]),
        turnover_penalty=0.0,
    )

    obs, _ = env.reset()
    records = []
    done = False

    while not done:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=model.device
            ).unsqueeze(0)
            distribution = model.policy.get_distribution(obs_tensor)
            probabilities = (
                distribution.distribution.probs
                .cpu().numpy()[0]
            )

        action, _ = model.predict(obs, deterministic=True)
        action = int(np.asarray(action).item())

        decision_date = env.dates[env.t]
        obs, reward, done, truncated, info = env.step(action)

        records.append({
            "date": decision_date,
            "policy_prob_hold": float(probabilities[0]),
            "policy_prob_buy": float(probabilities[1]),
            "policy_prob_sell": float(probabilities[2]),
            "policy_action": action,
        })

    result = env.to_frame().reset_index()
    policy_df = pd.DataFrame(records)
    result = result.merge(policy_df, on="date", how="left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    print(f"Saved: {out}")
    print(f"Decision rows: {len(result)}")
    print(f"First decision: {result['date'].iloc[0]}")
    print(f"Last decision: {result['date'].iloc[-1]}")
    print("No model training performed.")


if __name__ == "__main__":
    main()
