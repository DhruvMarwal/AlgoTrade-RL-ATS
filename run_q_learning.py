"""Tabular Q-learning + epsilon-greedy baseline for the gold trading project.
State bin edges are fitted on training data only. Evaluation is greedy (no exploration).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from env.trading_env import TradingEnv
from features.scaling import RobustScaler
from evaluation import metrics, run_policy

STATE_COLS = ["return_5d", "rolling_vol_10d", "rsi_14", "momentum_signal"]

def split_by_date(df, train_end, valid_end):
    a, b = pd.Timestamp(train_end), pd.Timestamp(valid_end)
    return {"train": df[df.index <= a], "valid": df[(df.index > a) & (df.index <= b)], "test": df[df.index > b]}

def fit_bins(train, cols, n_bins=4):
    edges = {}
    for c in cols:
        edges[c] = np.unique(train[c].quantile(np.linspace(0, 1, n_bins + 1)[1:-1]).to_numpy(dtype=float))
    return edges

def state_for(row, cols, edges, position):
    return tuple([int(np.digitize(float(row[c]), edges[c])) for c in cols] + [int(position)])

def train_one(train, seed, episodes, episode_len, alpha, gamma, eps_start, eps_end, penalty, cost):
    rng = np.random.default_rng(seed)
    cols = [c for c in STATE_COLS if c in train.columns]
    if len(cols) < 2:
        raise ValueError(f"Need at least two state columns from {STATE_COLS}; found {cols}")
    edges = fit_bins(train, cols)
    scaler = RobustScaler().fit(train[cols])
    X = scaler.transform(train[cols])
    env = TradingEnv(pd.DataFrame(X, index=train.index, columns=cols), train["log_return"].to_numpy(),
                     transaction_cost=cost, turnover_penalty=penalty, random_start=True,
                     episode_len=min(episode_len, len(train)-2))
    q = {}; starts=[]; progress=[]; steps=0
    max_steps = max(1, episodes * min(episode_len, len(train)-2))
    for ep in range(episodes):
        env.reset(seed=int(rng.integers(0, 2**31-1)))
        start = env.t
        starts.append({"algorithm":"Q-learning-epsilon-greedy", "seed":seed, "episode":ep+1,
                       "start_index":int(start), "end_index_exclusive":int(env.t_end),
                       "start_date":str(pd.Timestamp(train.index[start]).date()),
                       "end_date":str(pd.Timestamp(train.index[min(env.t_end,len(train)-1)]).date()),
                       "episode_length_steps":int(env.t_end-start)})
        total=0.0
        while True:
            st=state_for(train.iloc[env.t], cols, edges, env.position)
            q.setdefault(st, np.zeros(3, dtype=float))
            eps=eps_end+(eps_start-eps_end)*max(0.0, 1-steps/max_steps)
            if rng.random() < eps: action=int(rng.integers(0,3))
            else: action=int(rng.choice(np.flatnonzero(np.isclose(q[st], q[st].max()))))
            _, reward, done, _, _ = env.step(action)
            next_i=min(env.t, len(train)-1)
            nxt=state_for(train.iloc[next_i], cols, edges, env.position)
            q.setdefault(nxt, np.zeros(3, dtype=float))
            target=reward if done else reward+gamma*float(np.max(q[nxt]))
            q[st][action]+=alpha*(target-q[st][action])
            total+=reward; steps+=1
            if done: break
        progress.append({"algorithm":"Q-learning-epsilon-greedy", "seed":seed, "episode":ep+1,
                         "global_steps":steps, "epsilon_end":eps, "episode_reward":total, "q_states_seen":len(q)})
    return q, cols, edges, starts, progress

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features_csv", default="results/features_all.csv")
    ap.add_argument("--train_end", default="2016-12-31")
    ap.add_argument("--valid_end", default="2020-12-31")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--episode_len", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--turnover_penalty", type=float, default=0.0008)
    ap.add_argument("--cost", type=float, default=0.0005)
    ap.add_argument("--out_dir", default="results")
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.features_csv, index_col="Date", parse_dates=True).dropna()
    splits=split_by_date(df,args.train_end,args.valid_end)
    if min(len(v) for v in splits.values()) < 100: raise SystemExit("A split has fewer than 100 rows; check date boundaries.")
    rows=[]; starts_all=[]; progress_all=[]; actions_all=[]
    for seed in range(args.seeds):
        q,cols,edges,starts,progress=train_one(splits["train"],seed,args.episodes,args.episode_len,args.alpha,args.gamma,args.eps_start,args.eps_end,args.turnover_penalty,args.cost)
        starts_all.extend(starts); progress_all.extend(progress)
        for split in ("valid","test"):
            raw=splits[split]; scaler=RobustScaler().fit(splits["train"][cols]); X=scaler.transform(raw[cols])
            env=TradingEnv(pd.DataFrame(X,index=raw.index,columns=cols),raw["log_return"].to_numpy(),transaction_cost=args.cost)
            def policy(e):
                st=state_for(raw.iloc[e.t],cols,edges,e.position)
                return int(np.argmax(q.get(st,np.zeros(3))))
            fr=run_policy(env,policy)
            rows.append({"algorithm":"Q-learning-epsilon-greedy","seed":seed,"split":split,**metrics(fr)})
            h=fr.copy(); h.insert(0,"split",split); h.insert(0,"seed",seed); h.insert(0,"algorithm","Q-learning-epsilon-greedy"); actions_all.append(h.reset_index())
    results=pd.DataFrame(rows); results.to_csv(out/"q_learning_runs.csv",index=False)
    pd.DataFrame(starts_all).to_csv(out/"q_learning_episode_starts.csv",index=False)
    pd.DataFrame(progress_all).to_csv(out/"q_learning_training_progress.csv",index=False)
    pd.concat(actions_all,ignore_index=True).to_csv(out/"q_learning_action_history.csv",index=False)
    summary=results.groupby("split").agg(return_mean=("total_return_pct","mean"),return_std=("total_return_pct","std"),sharpe_mean=("sharpe","mean"),sharpe_std=("sharpe","std"),max_drawdown_mean=("max_drawdown_pct","mean"),exposure_mean=("time_in_market_pct","mean"),trades_mean=("total_trades","mean")).reset_index()
    summary.to_csv(out/"q_learning_summary.csv",index=False)
    print(summary.to_string(index=False)); print(f"\nSaved Q-learning results under {out.resolve()}")
if __name__=="__main__": main()
