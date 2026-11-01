"""
Metrics, baseline policies and a paired significance test.
All policies are run through the SAME environment (same costs, same timing),
so every number in the results table is directly comparable.
"""

import numpy as np
import pandas as pd
from scipy import stats

from env.trading_env import BUY, SELL, HOLD

TRADING_DAYS = 252  # annualisation; use 365 for crypto


def run_policy(env, policy) -> pd.DataFrame:
    """Run a policy (callable env -> action) over the environment's full sequence."""
    env.reset()
    done = False
    while not done:
        _, _, done, _, _ = env.step(policy(env))
    return env.to_frame()


def metrics(frame: pd.DataFrame) -> dict:
    r = frame["strategy_logret"].values
    equity = np.exp(np.cumsum(r))
    equity = np.concatenate([[1.0], equity])
    peak = np.maximum.accumulate(equity)
    max_dd = float(((equity - peak) / peak).min())

    pos = frame["position"].values
    entries = int(np.sum(np.diff(np.concatenate([[0], pos])) == 1))
    total_trades = int(frame["traded"].sum()) if "traded" in frame else entries
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    sharpe = float(np.sqrt(TRADING_DAYS) * r.mean() / sd) if sd > 0 else 0.0
    downside = r[r < 0]
    dsd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino = float(np.sqrt(TRADING_DAYS) * r.mean() / dsd) if dsd > 0 else 0.0
    return {
        "total_return_pct": float((np.exp(r.sum()) - 1) * 100),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd * 100,
        "entries": entries,
        "total_trades": total_trades,
        "time_in_market_pct": float(pos.mean() * 100),
        "cost_drag_pct": float((1 - np.exp(frame["cost_log"].sum())) * 100),
        "guard_overrides": int(frame["guard_override"].sum()),
    }


def paired_pvalue(agent: pd.DataFrame, benchmark: pd.DataFrame) -> float:
    """Two-sided HAC/Newey-West test of mean daily excess log return.

    This is more robust to short-range serial correlation than a plain one-sample t-test.
    It remains a rough per-run diagnostic: it does not correct for testing many seeds or
    account for all strategy-selection effects, so it must not be treated as definitive proof.
    """
    excess = np.asarray(agent["strategy_logret"], dtype=float) - np.asarray(benchmark["strategy_logret"], dtype=float)
    excess = excess[np.isfinite(excess)]
    if len(excess) < 10 or np.allclose(excess, 0) or excess.std(ddof=1) == 0:
        return 1.0
    try:
        import statsmodels.api as sm
        # Newey-West bandwidth rule, capped to leave a reasonable number of observations.
        maxlags = int(min(len(excess) // 4, max(1, round(4 * (len(excess) / 100.0) ** (2 / 9)))))
        fit = sm.OLS(excess, np.ones((len(excess), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags}
        )
        return float(fit.pvalues[0])
    except (ImportError, ValueError, ZeroDivisionError):
        # Conservative fallback for environments without statsmodels HAC support.
        return float(stats.ttest_1samp(excess, 0.0).pvalue)


def _to_action(target: int, position: int) -> int:
    if target == 1 and position == 0:
        return BUY
    if target == 0 and position == 1:
        return SELL
    return HOLD


def baseline_policies(raw: pd.DataFrame, vol_threshold: float) -> dict:
    """Rule-based policies computed from RAW (unscaled) features of the same split."""
    mom = raw["return_10d"].values
    vol = raw["garch_cond_vol"].values

    return {
        "flat (cash)":        lambda env: HOLD,
        "buy & hold":         lambda env: _to_action(1, env.position),
        "momentum (10d>0)":   lambda env: _to_action(int(mom[env.t] > 0), env.position),
        "vol filter (no RL)": lambda env: _to_action(int(vol[env.t] <= vol_threshold), env.position),
    }
