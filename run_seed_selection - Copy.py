"""
Seed-selection runner: trains A/B/C, saves checkpoints and final weights, artifacts, logs, plots, and selects seeds by VALIDATION only.

  A  PPO on raw price features
  B  PPO on raw + ATS features (GARCH vol forecast, regime flag, momentum)
  C  same as B, with a volatility guardrail inside the environment

Everything (scaler, guardrail threshold, GARCH parameters) is derived from the
TRAINING window only. The test window is evaluated once per run, and no model
selection is done on it.

Usage:
    python run_experiments.py --features_csv results/features_all.csv \
        --train_end 2019-12-31 --valid_end 2022-12-31 --timesteps 100000 --seeds 5

Quick check (a minute or two):
    python run_experiments.py --features_csv results/features_all.csv --timesteps 3000 --seeds 1
Baselines only (no torch needed):
    python run_experiments.py --features_csv results/features_all.csv --baselines_only

TURNOVER PENALTY
    Earlier runs showed 50-100+ trades over a ~1000-day test window, with cost_drag of
    5-9% of equity -- the agent was flipping position far more than any edge could justify
    (corr(entries, cost_drag) = 1.00 in that run). --turnover_penalty adds an extra cost to
    the TRAINING reward only (never to the reported P&L) each time the agent trades, to push
    PPO away from that behaviour. Try 0.05 to 0.2 first; compare "entries" and "total_trades"
    in the results table against a run with --turnover_penalty 0.

WALK-FORWARD (robustness across market regimes, not just one test window)
    python run_experiments.py --features_csv results/features_all.csv --walkforward \
        --wf_first_test_year 2016 --wf_fold_years 1 --wf_min_train_years 5 --timesteps 60000 --seeds 3
    Trains on an expanding window and tests on the next --wf_fold_years, one fold at a time,
    until the data runs out. Reports each fold separately (results/walkforward_table.md) so a
    strong or weak result on one window (e.g. the 2023-2024 gold rally) cannot drive the
    conclusion by itself. No separate validation split per fold; this mode is for an honest
    read of consistency across time, not for hyperparameter tuning.
"""
import os
import json
import pickle
import argparse
import shutil
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.trading_env import TradingEnv
from features.build_features import RAW_COLS, ATS_COLS
from features.scaling import RobustScaler
from evaluation import run_policy, metrics, paired_pvalue, baseline_policies
from plotting import plot_backtest, safe_savefig

EXPERIMENTS = {
    "A": dict(cols=RAW_COLS, guardrail=False),
    "B": dict(cols=RAW_COLS + ATS_COLS, guardrail=False),
    "C": dict(cols=RAW_COLS + ATS_COLS, guardrail=True),
}


def split_by_date(df, train_end, valid_end):
    train_end, valid_end = pd.Timestamp(train_end), pd.Timestamp(valid_end)
    train = df[df.index <= train_end]
    valid = df[(df.index > train_end) & (df.index <= valid_end)]
    test = df[df.index > valid_end]
    return {"train": train, "valid": valid, "test": test}


def make_env(raw, scaler, cols, guardrail, vol_thr, cost, turnover_penalty=0.0,
             random_start=False, episode_len=None):
    X = scaler.transform(raw[cols])
    return TradingEnv(
        X, raw["log_return"].values,
        guard_vol=raw["garch_cond_vol"].values if guardrail else None,
        vol_threshold=vol_thr if guardrail else None,
        transaction_cost=cost, turnover_penalty=turnover_penalty,
        random_start=random_start, episode_len=episode_len,
    )


def fmt(mean, std, digits=1):
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def build_table(runs: pd.DataFrame, split: str) -> pd.DataFrame:
    sub = runs[runs["split"] == split]
    rows = []
    order = list(dict.fromkeys(sub["name"]))
    for name in order:
        g = sub[sub["name"] == name]
        m, s = g.mean(numeric_only=True), g.std(numeric_only=True, ddof=1)
        multi = len(g) > 1
        sd = lambda k: s[k] if multi else np.nan
        rows.append({
            "strategy": name,
            "runs": len(g),
            "return %": fmt(m["total_return_pct"], sd("total_return_pct")),
            "Sharpe": fmt(m["sharpe"], sd("sharpe"), 2),
            "max DD %": fmt(m["max_drawdown_pct"], sd("max_drawdown_pct")),
            "Sortino": fmt(m["sortino"], sd("sortino"), 2),
            "entries": fmt(m["entries"], sd("entries"), 1),
            "total trades": fmt(m["total_trades"], sd("total_trades"), 1),
            "in market %": fmt(m["time_in_market_pct"], sd("time_in_market_pct"), 0),
            "cost drag %": fmt(m["cost_drag_pct"], sd("cost_drag_pct")),
            "p<0.05 vs B&H": (f"{int((g['paired_p_vs_bh'] < 0.05).sum())}/{len(g)}"
                              if name in EXPERIMENTS else "-"),
        })
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def plot_equity(curves, baseline_curves, out_path, title):
    exps = [e for e in EXPERIMENTS if e in curves]
    fig, axes = plt.subplots(1, max(len(exps), 1), figsize=(5.5 * max(len(exps), 1), 4.5), sharey=True, squeeze=False)
    for ax, e in zip(axes[0], exps):
        for eq in curves[e]:
            ax.plot(eq.index, eq.values, lw=0.8, alpha=0.6)
        for name, style in [("buy & hold", "k--"), ("vol filter (no RL)", "g-."), ("flat (cash)", "0.5")]:
            if name in baseline_curves:
                ax.plot(baseline_curves[name].index, baseline_curves[name].values, style, lw=1.4, label=name)
        ax.set_title(f"Experiment {e} (one line per seed)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("equity (start = 1.0)")
    fig.suptitle(title)
    fig.tight_layout()
    safe_savefig(fig, out_path, dpi=120)


def equity_series(frame):
    eq = np.exp(np.cumsum(frame["strategy_logret"].values))
    return pd.Series(eq, index=frame.index)



def run_policy_with_action_probs(env, model):
    """Evaluate a PPO model and attach its state-wise HOLD/BUY/SELL probabilities.

    Probabilities are captured from the same observation used to choose each action.
    This is evaluation logging only; it does not change training or action selection.
    """
    env.reset()
    probability_rows = []
    done = False
    while not done:
        obs = env._obs()
        decision_date = pd.Timestamp(env.dates[env.t])
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        distribution = model.policy.get_distribution(obs_tensor).distribution
        probs = distribution.probs.detach().cpu().numpy().reshape(-1)
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

def run_walkforward(df: pd.DataFrame, args, out: Path):
    """Expanding-window folds: train on everything up to year Y-1, test on year Y (or
    --wf_fold_years years), then move Y forward. Reports each fold separately so one
    unusual window (e.g. a strong trend) cannot drive the overall conclusion."""
    first_year, last_year = df.index.year.min(), df.index.year.max()
    folds = []
    y = args.wf_first_test_year
    while y + args.wf_fold_years - 1 <= last_year:
        train_end = pd.Timestamp(y - 1, 12, 31)
        test_end = pd.Timestamp(y + args.wf_fold_years - 1, 12, 31)
        if (train_end.year - first_year + 1) >= args.wf_min_train_years:
            folds.append((train_end, test_end))
        y += args.wf_fold_years
    if not folds:
        raise SystemExit("No folds fit in this date range; lower --wf_min_train_years or --wf_first_test_year.")

    print(f"Walk-forward: {len(folds)} fold(s)")
    for te, tend in folds:
        print(f"  train <= {te.date()}  |  test {(te + pd.Timedelta(days=1)).date()} -> {tend.date()}")

    rows = []
    for fold_i, (train_end, test_end) in enumerate(folds):
        train = df[df.index <= train_end]
        test = df[(df.index > train_end) & (df.index <= test_end)]
        if len(train) < 300 or len(test) < 30:
            print(f"Skipping fold {fold_i} (train={len(train)}, test={len(test)} rows)")
            continue
        vol_thr = float(train["garch_cond_vol"].quantile(args.vol_quantile))
        scaler_bh = RobustScaler().fit(train[EXPERIMENTS["B"]["cols"]])
        bh_env = make_env(test, scaler_bh, EXPERIMENTS["B"]["cols"], False, vol_thr, args.cost)
        bh_frame = run_policy(bh_env, lambda e: _to_action_bh(e))
        rows.append({"fold": fold_i, "name": "buy & hold", "seed": 0,
                     "test_start": test.index.min().date(), "test_end": test.index.max().date(),
                     **metrics(bh_frame), "paired_p_vs_bh": np.nan})

        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        for exp in args.experiments:
            cfg = EXPERIMENTS[exp]
            scaler = RobustScaler().fit(train[cfg["cols"]])
            for seed in range(args.seeds):
                t0 = time.time()
                train_env = make_env(train, scaler, cfg["cols"], cfg["guardrail"], vol_thr, args.cost,
                                     turnover_penalty=args.turnover_penalty, random_start=True,
                                     episode_len=min(args.episode_len, max(len(train) - 2, 50)))
                model = PPO("MlpPolicy", train_env, seed=seed, verbose=0, device="cpu",
                            n_steps=1024, batch_size=64, learning_rate=3e-4, gamma=0.99,
                            ent_coef=0.01, policy_kwargs=dict(net_arch=[64, 64]))
                model.learn(total_timesteps=args.timesteps)
                env = make_env(test, scaler, cfg["cols"], cfg["guardrail"], vol_thr, args.cost)
                fr = run_policy_with_action_probs(env, model)
                rows.append({"fold": fold_i, "name": exp, "seed": seed,
                             "test_start": test.index.min().date(), "test_end": test.index.max().date(),
                             **metrics(fr), "paired_p_vs_bh": paired_pvalue(fr, bh_frame)})
                print(f"  fold {fold_i} [{exp}] seed {seed}  {time.time()-t0:4.0f}s  "
                      f"return {rows[-1]['total_return_pct']:+6.1f}%  Sharpe {rows[-1]['sharpe']:+.2f}  "
                      f"trades {rows[-1]['total_trades']}", flush=True)

    wf_df = pd.DataFrame(rows)
    wf_df.to_csv(out / "walkforward_runs.csv", index=False)

    per_fold = (wf_df.groupby(["fold", "name"], as_index=False)
                .agg(test_start=("test_start", "first"), test_end=("test_end", "first"),
                     return_pct=("total_return_pct", "mean"), sharpe=("sharpe", "mean"),
                     max_dd_pct=("max_drawdown_pct", "mean"), total_trades=("total_trades", "mean")))
    per_fold.to_csv(out / "walkforward_per_fold.csv", index=False)

    overall = []
    for name in wf_df.name.unique():
        g = wf_df[wf_df.name == name]
        overall.append({
            "strategy": name, "folds": g.fold.nunique(), "runs": len(g),
            "return %": fmt(g.total_return_pct.mean(), g.total_return_pct.std(ddof=1)),
            "Sharpe": fmt(g.sharpe.mean(), g.sharpe.std(ddof=1), 2),
            "max DD %": fmt(g.max_drawdown_pct.mean(), g.max_drawdown_pct.std(ddof=1)),
            "folds beating B&H": (f"{sum((wf_df[(wf_df.fold==f)&(wf_df.name==name)].total_return_pct.mean() > wf_df[(wf_df.fold==f)&(wf_df.name=='buy & hold')].total_return_pct.mean()) for f in g.fold.unique())}/{g.fold.nunique()}"
                                  if name != "buy & hold" else "-"),
        })
    overall_df = pd.DataFrame(overall)

    md = ["# Walk-forward results", "",
          f"{len(folds)} fold(s), expanding training window, {args.wf_fold_years}-year test per fold.",
          f"Cost: {args.cost*1e4:.0f} bps/trade. Turnover penalty (training only): {args.turnover_penalty}. Seeds/fold: {args.seeds}.", "",
          "## Per fold (mean across seeds)", "", to_markdown(per_fold.round(2)), "",
          "## Overall (pooled across all folds and seeds)", "", to_markdown(overall_df), ""]
    (out / "walkforward_table.md").write_text("\n".join(md), encoding="utf-8")
    print("\n===== WALK-FORWARD: per fold =====")
    print(per_fold.round(2).to_string(index=False))
    print("\n===== WALK-FORWARD: overall =====")
    print(overall_df.to_string(index=False))
    print(f"\nSaved: {out/'walkforward_runs.csv'}, walkforward_per_fold.csv, walkforward_table.md")


def _to_action_bh(env):
    from env.trading_env import BUY, SELL, HOLD
    return BUY if env.position == 0 else HOLD



class TrainingDiagnosticsCallback:
    """Log rewards during rollout and PPO metrics after each training update."""

    def __new__(cls, csv_path, run_name, seed, check_freq=2048, verbose=0):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__(verbose)
                self.csv_path = Path(csv_path)
                self.run_name = run_name
                self.seed_id = seed
                self.recent_rewards = deque(maxlen=2048)
                self.rows = []

            def _on_step(self):
                rewards = self.locals.get("rewards")
                if rewards is not None:
                    self.recent_rewards.extend(
                        np.asarray(rewards).reshape(-1).tolist()
                    )
                return True

            def record_training_metrics(self, logger_values, timesteps):
                row = {
                    "experiment": self.run_name,
                    "seed": self.seed_id,
                    "timesteps": int(timesteps),
                    "recent_reward_mean": (
                        float(np.mean(self.recent_rewards))
                        if self.recent_rewards else np.nan
                    ),
                    "recent_reward_std": (
                        float(np.std(self.recent_rewards))
                        if self.recent_rewards else np.nan
                    ),
                }

                for key in (
                    "train/entropy_loss",
                    "train/policy_gradient_loss",
                    "train/value_loss",
                    "train/approx_kl",
                    "train/clip_fraction",
                ):
                    value = logger_values.get(key)
                    row[key.replace("/", "_")] = (
                        float(value) if value is not None else np.nan
                    )

                self.rows.append(row)

            def _on_training_end(self):
                if self.rows:
                    self.csv_path.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    pd.DataFrame(self.rows).to_csv(
                        self.csv_path,
                        mode="a",
                        header=not self.csv_path.exists(),
                        index=False,
                    )

        return _Callback()

def save_episode_log(env, out_path, experiment, seed):
    rows = getattr(env, "episode_log", [])
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.insert(0, "seed", seed)
    frame.insert(0, "experiment", experiment)
    p = Path(out_path)
    frame.to_csv(p, mode="a", header=not p.exists(), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", default="results/features_all.csv")
    ap.add_argument("--train_end", default="2019-12-31")
    ap.add_argument("--valid_end", default="2022-12-31")
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--experiments", nargs="+", default=["A", "B", "C"], choices=list(EXPERIMENTS))
    ap.add_argument("--cost", type=float, default=0.0005, help="cost per position change (0.0005 = 5 bps)")
    ap.add_argument("--turnover_penalty", type=float, default=0.0,
                    help="extra TRAINING-reward cost per position change; 0.0008 means 0.08 after reward scaling by 100")
    ap.add_argument("--vol_quantile", type=float, default=0.85, help="guardrail threshold = this quantile of TRAIN vol")
    ap.add_argument("--episode_len", type=int, default=500)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--model_dir", default="models/seed_selection")
    ap.add_argument("--checkpoint_freq", type=int, default=10000, help="Save PPO checkpoints every N environment steps")
    ap.add_argument("--baselines_only", action="store_true")
    ap.add_argument("--walkforward", action="store_true", help="run expanding-window folds instead of one split; see module docstring")
    ap.add_argument("--wf_first_test_year", type=int, default=2016)
    ap.add_argument("--wf_fold_years", type=int, default=1)
    ap.add_argument("--wf_min_train_years", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    df = pd.read_csv(args.features_csv, index_col="Date", parse_dates=True)

    if args.walkforward:
        run_walkforward(df, args, out)
        return

    splits = split_by_date(df, args.train_end, args.valid_end)
    for k, v in splits.items():
        if len(v) < 100:
            raise SystemExit(f"Split '{k}' has only {len(v)} rows; adjust --train_end/--valid_end.")
        print(f"{k:>5}: {len(v):5d} rows  {v.index.min().date()} -> {v.index.max().date()}")

    vol_thr = float(splits["train"]["garch_cond_vol"].quantile(args.vol_quantile))
    print(f"\nGuardrail threshold (train {args.vol_quantile:.0%} quantile of GARCH vol): {vol_thr:.5f}")

    runs, test_curves, base_curves = [], {e: [] for e in EXPERIMENTS}, {}
    episode_log_path = out / "episode_starts.csv"
    training_log_path = out / "training_progress.csv"
    action_log_path = out / "seed_action_history.csv"
    seed_summary_path = out / "seed_summary.csv"
    for diagnostic_path in (episode_log_path, training_log_path, action_log_path, seed_summary_path):
        if diagnostic_path.exists(): diagnostic_path.unlink()
    seed0_test_frames = {}   # one representative per-step frame per experiment, for the buy/sell chart

    # ---------------------------------------------------------------- baselines
    scaler_all = RobustScaler().fit(splits["train"][EXPERIMENTS["B"]["cols"]])
    bh_frames = {}
    for split in ("valid", "test"):
        raw = splits[split]
        pols = baseline_policies(raw, vol_thr)
        for name, pol in pols.items():
            env = make_env(raw, scaler_all, EXPERIMENTS["B"]["cols"], False, vol_thr, args.cost)
            fr = run_policy(env, pol)
            if name == "buy & hold":
                bh_frames[split] = fr
            if split == "test":
                base_curves[name] = equity_series(fr)
            runs.append({"name": name, "seed": 0, "split": split, **metrics(fr), "paired_p_vs_bh": np.nan})

    if not args.baselines_only:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        torch.set_num_threads(1)
        model_root = Path(args.model_dir)
        model_root.mkdir(parents=True, exist_ok=True)

        for exp in args.experiments:
            cfg = EXPERIMENTS[exp]
            scaler = RobustScaler().fit(splits["train"][cfg["cols"]])
            for seed in range(args.seeds):
                t0 = time.time()
                train_env = make_env(splits["train"], scaler, cfg["cols"], cfg["guardrail"], vol_thr,
                                     args.cost, turnover_penalty=args.turnover_penalty,
                                     random_start=True, episode_len=args.episode_len)
                model = PPO("MlpPolicy", train_env, seed=seed, verbose=0, device="cpu",
                            n_steps=1024, batch_size=64, learning_rate=3e-4, gamma=0.99,
                            ent_coef=0.01, policy_kwargs=dict(net_arch=[64, 64]))
                run_dir = model_root / exp / f"seed_{seed:03d}"
                run_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_callback = CheckpointCallback(
                    save_freq=max(1, args.checkpoint_freq),
                    save_path=str(run_dir / "checkpoints"),
                    name_prefix="ppo",
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                )


                diag_callback = TrainingDiagnosticsCallback(
                    training_log_path, exp, seed,
                    check_freq=max(1024, min(5000, args.timesteps // 20))
                )

                # Capture metrics immediately after each PPO training update.
                original_train = model.train

                def train_with_diagnostics():
                    original_train()
                    diag_callback.record_training_metrics(
                        model.logger.name_to_value,
                        model.num_timesteps,
                    )

                model.train = train_with_diagnostics

                from stable_baselines3.common.callbacks import CallbackList

                model.learn(
                    total_timesteps=args.timesteps,
                    callback=CallbackList([checkpoint_callback, diag_callback]),
                )



                
                # Save final model + all preprocessing needed to reproduce its observations.
                model.save(str(run_dir / "model"))
                with open(run_dir / "scaler.pkl", "wb") as f:
                    pickle.dump(scaler, f)
                metadata = {
                    "experiment": exp, "seed": int(seed), "timesteps": int(args.timesteps),
                    "train_start": str(splits["train"].index.min().date()),
                    "train_end": str(splits["train"].index.max().date()),
                    "valid_start": str(splits["valid"].index.min().date()),
                    "valid_end": str(splits["valid"].index.max().date()),
                    "test_start": str(splits["test"].index.min().date()),
                    "test_end": str(splits["test"].index.max().date()),
                    "feature_cols": list(cfg["cols"]), "guardrail": bool(cfg["guardrail"]),
                    "guardrail_vol_threshold": float(vol_thr), "vol_quantile": float(args.vol_quantile),
                    "transaction_cost": float(args.cost), "turnover_penalty": float(args.turnover_penalty),
                    "features_csv": str(args.features_csv), "selection_metric": "validation composite; test not used for selection"
                }
                (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                save_episode_log(train_env, episode_log_path, exp, seed)

                for split in ("valid", "test"):
                    env = make_env(splits[split], scaler, cfg["cols"], cfg["guardrail"], vol_thr, args.cost)
                    fr = run_policy_with_action_probs(env, model)
                    action_dump = fr.copy()
                    action_dump.insert(0, "split", split)
                    action_dump.insert(0, "seed", seed)
                    action_dump.insert(0, "experiment", exp)
                    action_dump.to_csv(action_log_path, mode="a", header=not action_log_path.exists(), index=True)
                    # Save a separate per-run daily log for easy replay/inspection.
                    fr.to_csv(run_dir / f"{split}_daily_log.csv", index=True, index_label="Date")
                    eq = equity_series(fr)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(eq.index, eq.values, label=f"{exp} seed {seed}")
                    ax.set_title(f"{exp} seed {seed} - {split} equity")
                    ax.set_ylabel("Equity (start=1.0)")
                    ax.tick_params(axis="x", rotation=30)
                    ax.legend()
                    fig.tight_layout()
                    safe_savefig(fig, run_dir / f"{split}_equity.png", dpi=120)
                    runs.append({"name": exp, "seed": seed, "split": split, **metrics(fr),
                                 "paired_p_vs_bh": paired_pvalue(fr, bh_frames[split])})
                    if split == "test":
                        test_curves[exp].append(eq)
                        if seed == 0:
                            seed0_test_frames[exp] = fr
                last = [r for r in runs if r["name"] == exp and r["seed"] == seed and r["split"] == "valid"][0]
                print(f"[{exp}] seed {seed}  {time.time() - t0:5.0f}s  "
                      f"VALID return {last['total_return_pct']:+7.1f}%  Sharpe {last['sharpe']:+.2f}  "
                      f"drawdown {last['max_drawdown_pct']:+.1f}%  (model saved)", flush=True)

    # ------------------------------------------------------------------ reports
    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(out / "runs.csv", index=False)
    runs_df[runs_df["name"].isin(EXPERIMENTS)].to_csv(seed_summary_path, index=False)

    md = ["# Results", "",
          f"Split: train <= {args.train_end}, validation <= {args.valid_end}, test after.",
          f"Cost per position change: {args.cost * 1e4:.0f} bps. Guardrail threshold: {vol_thr:.5f} (train {args.vol_quantile:.0%} quantile).",
          f"PPO timesteps: {args.timesteps}, seeds: {args.seeds}. Values are mean ± std across seeds.", ""]
    for split, label in (("test", "TEST (final exam, evaluated once)"), ("valid", "VALIDATION")):
        table = build_table(runs_df, split)
        table.to_csv(out / f"summary_{split}.csv", index=False)
        md += [f"## {label}", "", to_markdown(table), ""]
        print(f"\n===== {label} =====")
        print(table.to_string(index=False))
    (out / "results_table.md").write_text("\n".join(md), encoding="utf-8")

    # Select the best seed per experiment using VALIDATION metrics only.
    # Composite ranks Sharpe first, then return and drawdown as tie-breakers.
    valid_rl = runs_df[(runs_df["split"] == "valid") & (runs_df["name"].isin(EXPERIMENTS))].copy()
    if not valid_rl.empty:
        valid_rl["selection_score"] = (
            valid_rl["sharpe"].fillna(-999) * 1.0
            + valid_rl["total_return_pct"].fillna(-999) * 0.01
            + valid_rl["max_drawdown_pct"].fillna(-999) * 0.005
        )
        selected_rows = []
        for exp in args.experiments:
            g = valid_rl[valid_rl["name"] == exp].sort_values(
                ["selection_score", "sharpe", "total_return_pct"], ascending=False)
            if g.empty:
                continue
            best = g.iloc[0]
            source = model_root / exp / f"seed_{int(best['seed']):03d}"
            dest = model_root / "selected" / exp
            dest.mkdir(parents=True, exist_ok=True)
            # Copy full run folder, including checkpoints/logs/scaler/metadata.
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest)
            selected_rows.append({"experiment": exp, "selected_seed": int(best["seed"]),
                                  "validation_return_pct": float(best["total_return_pct"]),
                                  "validation_sharpe": float(best["sharpe"]),
                                  "validation_max_drawdown_pct": float(best["max_drawdown_pct"]),
                                  "selection_score": float(best["selection_score"]),
                                  "model_path": str(dest / "model.zip")})
        pd.DataFrame(selected_rows).to_csv(out / "best_seed_selection.csv", index=False)
        print("\n===== SELECTED SEEDS (validation only) =====")
        print(pd.DataFrame(selected_rows).to_string(index=False))

    if not args.baselines_only:
        plot_equity(test_curves, base_curves, out / "test_equity_curves.png", "Test period equity curves")
        for exp, fr in seed0_test_frames.items():
            # seed 0 only: one representative run per experiment, not all seeds (would be visual noise).
            # Check results/runs.csv if you want to see how seed 0 compares to the other seeds for this experiment.
            saved = plot_backtest(fr, bh_frames["test"], f"Experiment {exp} (seed 0), test period",
                                  out / f"backtest_{exp}_seed0.png")
            if saved:
                print(f"Saved buy/sell chart: {saved}")
    print(f"\nSaved: {out/'runs.csv'}, seed_summary.csv, episode_starts.csv, training_progress.csv, seed_action_history.csv, summary_test.csv, summary_valid.csv, results_table.md, test_equity_curves.png"
          + (f", backtest_<experiment>_seed0.png" if not args.baselines_only else ""))


if __name__ == "__main__":
    main()
