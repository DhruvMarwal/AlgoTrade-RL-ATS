"""
Per-step diagnostic chart: equity curve vs buy-and-hold, drawdown, and price with
buy/sell markers -- the same three-panel layout used in the reference project's report,
built from TradingEnv.to_frame().

Also: safe_savefig(), which never crashes a long training run just because a PNG is
open in a viewer (a real failure mode on Windows -- a locked file raises PermissionError).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_savefig(fig, path, **kw):
    """Save a figure; on failure (e.g. the file is open in a viewer), fall back to a
    timestamped name instead of losing the whole run's output. Always closes fig."""
    import time
    try:
        fig.savefig(path, **kw)
        ok_path = path
    except (PermissionError, OSError) as e:
        import os
        root, ext = os.path.splitext(str(path))
        fallback = f"{root}_{int(time.time())}{ext}"
        print(f"WARNING: could not save {path} ({e}); saved to {fallback} instead. "
              f"(Close the file if it's open in another program.)")
        try:
            fig.savefig(fallback, **kw)
            ok_path = fallback
        except Exception as e2:
            print(f"WARNING: fallback save also failed ({e2}); figure discarded.")
            ok_path = None
    finally:
        plt.close(fig)
    return ok_path


def plot_backtest(frame: pd.DataFrame, bh_frame: pd.DataFrame, title: str, out_path):
    """frame, bh_frame: outputs of TradingEnv.to_frame() for the agent and for a
    buy & hold run over the SAME period (so the equity panel is a fair comparison)."""
    bh_equity = np.exp(np.cumsum(bh_frame["strategy_logret"].values))

    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1, 1.4]})

    ax[0].plot(frame.index, frame["equity"], lw=1.4, color="tab:blue", label="RL agent")
    ax[0].plot(bh_frame.index, bh_equity, lw=1.2, color="tab:grey", ls="--", label="Buy & hold")
    ax[0].set_ylabel("Equity (start = 1.0)")
    ax[0].set_title(f"{title} -- out-of-sample equity curve")
    ax[0].legend()

    ax[1].fill_between(frame.index, frame["drawdown"] * 100, 0, color="tab:red", alpha=0.4)
    ax[1].set_ylabel("Drawdown %")

    ax[2].plot(frame.index, frame["price"], lw=0.9, color="k")
    ax[2].fill_between(frame.index, frame["price"].min(), frame["price"].max(),
                       where=frame["position"] == 1, color="tab:green", alpha=0.15, label="long")
    buys = frame[(frame["action"] == 1) & (frame["position"] == 1)]
    sells = frame[(frame["action"] == 2) & (frame["position"] == 0)]
    guarded = frame[frame["guard_override"] == 1]
    ax[2].scatter(buys.index, buys["price"], marker="^", s=32, color="green", zorder=5, label="buy")
    ax[2].scatter(sells.index, sells["price"], marker="v", s=32, color="red", zorder=5, label="sell")
    if len(guarded):
        ax[2].scatter(guarded.index, guarded["price"], marker="x", s=28, color="orange",
                      zorder=6, label="guardrail override")
    ax[2].set_ylabel("Reconstructed price (start = 1.0)")
    ax[2].legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    return safe_savefig(fig, out_path, dpi=120)
