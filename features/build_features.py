"""
Builds ONE feature table containing every column any experiment may use.
run_experiments.py then picks the subset for each experiment:

    Experiment A: RAW_COLS only                      (price-derived, no ATS)
    Experiment B: RAW_COLS + ATS_COLS                (adds GARCH vol, regime, momentum)
    Experiment C: same features as B + guardrail     (guardrail lives in the environment)

All features at row t use information up to and including the close of day t.

Usage:
    python features/build_features.py --price_csv data/gold.csv --ats_csv results/ats_features.csv --out results/features_all.csv

Optional: replace the GARCH volatility feature with the HAR-RV forecast from the 5-minute
ATS notebook (results/ats_5m/daily_rv_features.csv), which beat GARCH on QLIKE with a
significant Diebold-Mariano test in that notebook's output:
    python features/build_features.py --price_csv data/gold.csv --ats_csv results/ats_features.csv \
        --har_csv results/ats_5m/daily_rv_features.csv --out results/features_all_har.csv
"""

import argparse
import numpy as np
import pandas as pd

RAW_COLS = ["log_return", "return_5d", "return_10d", "rolling_vol_10d", "rsi_14"]
ATS_COLS = ["garch_cond_vol", "regime_flag", "momentum_signal"]


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build(price_csv: str, ats_csv: str, har_csv: str | None = None) -> pd.DataFrame:
    price = pd.read_csv(price_csv, index_col="Date", parse_dates=True)
    ats = pd.read_csv(ats_csv, index_col="Date", parse_dates=True)

    f = pd.DataFrame(index=price.index)
    f["log_return"] = price["log_return"]
    f["return_5d"] = price["Close"].pct_change(5)
    f["return_10d"] = price["Close"].pct_change(10)
    f["rolling_vol_10d"] = price["log_return"].rolling(10).std()
    f["rsi_14"] = _rsi(price["Close"], 14)

    f = f.join(ats[["garch_cond_vol", "regime_flag"]], how="inner")
    # Actual directional momentum: trailing 20-session price return, known at close t.
    # This is deliberately distinct from the raw 10-session return feature.
    f["momentum_signal"] = price["Close"].pct_change(20)

    if har_csv:
        # har_rv_fc is next-day realized-VARIANCE (from the 5-min notebook, train-only fitted,
        # filtered forward -- no leakage). Convert to volatility (%) and overwrite garch_cond_vol
        # so it slots into the SAME feature name the environment/guardrail already use.
        har = pd.read_csv(har_csv, index_col="Date", parse_dates=True)
        har_vol = np.sqrt(har["har_rv_fc"].clip(lower=0)) / 100.0   # -> same units as garch_cond_vol
        f["garch_cond_vol"] = har_vol.reindex(f.index)
        f = f.dropna(subset=["garch_cond_vol"])
        print(f"Replaced garch_cond_vol with the HAR-RV forecast from {har_csv}")

    f = f[RAW_COLS + ATS_COLS].dropna()
    f.index.name = "Date"
    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price_csv", type=str, required=True)
    parser.add_argument("--ats_csv", type=str, default="results/ats_features.csv")
    parser.add_argument("--har_csv", type=str, default=None,
                        help="optional: results/ats_5m/daily_rv_features.csv, to use HAR-RV instead of GARCH")
    parser.add_argument("--out", type=str, default="results/features_all.csv")
    args = parser.parse_args()

    feats = build(args.price_csv, args.ats_csv, args.har_csv)
    feats.to_csv(args.out)
    print(f"Saved {feats.shape[0]} rows x {feats.shape[1]} columns to {args.out}")
    print(f"Date range: {feats.index.min().date()} -> {feats.index.max().date()}")
    print(feats.columns.tolist())


if __name__ == "__main__":
    main()
