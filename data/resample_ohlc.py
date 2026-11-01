"""
Resamples a combined OHLCV dataset up to any coarser timeframe. Use this
on top of combine_intraday.py's output -- download 1-min bars (the finest
grain you need), combine them, then derive 5min/1h/4h/daily from that one
file instead of separately downloading and merging each timeframe.

Usage:
    python data/resample_ohlc.py --csv data/gold_1min_combined.csv --freq 1h --out data/gold_1h.csv
    python data/resample_ohlc.py --csv data/gold_1min_combined.csv --freq 4h --out data/gold_4h.csv
    python data/resample_ohlc.py --csv data/gold_1min_combined.csv --freq 1D --out data/gold_daily.csv

Pandas offset aliases: 5min=5min, 1h=1h, 4h=4h, 1D=1 day.
"""

import argparse
import numpy as np
import pandas as pd


OHLC_AGG = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
}


def resample(csv_path: str, freq: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    missing = [c for c in OHLC_AGG if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing expected columns: {missing}")

    agg = {k: v for k, v in OHLC_AGG.items() if k in df.columns}
    out = df.resample(freq).agg(agg)
    out = out.dropna(subset=["Open", "Close"])  # drop bars with no trades (e.g. market closed)

    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    out = out.dropna()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--freq", type=str, required=True, help="e.g. 5min, 1h, 4h, 1D")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    out = resample(args.csv, args.freq)
    out.to_csv(args.out)
    print(f"Resampled to '{args.freq}': {len(out)} rows -> saved to {args.out}")
    print(out.head())


if __name__ == "__main__":
    main()
