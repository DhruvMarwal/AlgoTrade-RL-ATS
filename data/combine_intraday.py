"""
Merges multiple daily OHLC CSV exports (the kind downloaded one-day-at-a-time
from a site like Dukascopy) into a single clean, sorted, deduplicated CSV.

Expects each file to have a timestamp column (auto-detected, e.g. "Etc/UTC")
and Open/High/Low/Close/Volume columns -- exactly the format of the files
downloaded from the historical data site.

Usage:
    python data/combine_intraday.py --input_dir data/raw_1min --pattern "XAU-USD_1Minute_*.csv" --out data/gold_1min_combined.csv
    python data/combine_intraday.py --input_dir data/raw_5min --pattern "XAU-USD_5Minute_*.csv" --out data/gold_5min_combined.csv

Typical workflow:
    1. Download each day's file from the site and drop them all into one
       folder (e.g. data/raw_1min/), without renaming them.
    2. Run this script once to merge everything into one combined CSV.
    3. Re-run it again later (pointing at the same folder) whenever you've
       downloaded more days -- it always produces a fresh, deduplicated,
       sorted combined file, so it's safe to re-run as your collection grows.
"""

import argparse
import glob
import os
import pandas as pd


TIMESTAMP_CANDIDATES = ["Etc/UTC", "Timestamp", "Date", "Datetime", "time"]


def _find_timestamp_col(df: pd.DataFrame) -> str:
    for c in TIMESTAMP_CANDIDATES:
        if c in df.columns:
            return c
    # fall back to the first column if nothing matches
    return df.columns[0]


def combine(input_dir: str, pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in '{input_dir}'. "
            "Check the folder path and pattern."
        )

    frames = []
    for f in files:
        df = pd.read_csv(f)
        ts_col = _find_timestamp_col(df)
        df = df.rename(columns={ts_col: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp", keep="first")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined = combined.set_index("timestamp")

    print(f"Merged {len(files)} files -> {len(combined)} rows "
          f"({combined.index.min()} to {combined.index.max()})")

    # sanity check: look for large time gaps (missed downloads, weekends, etc.)
    expected_freq = pd.Timedelta(
        (combined.index[1] - combined.index[0])
    ) if len(combined) > 1 else None
    if expected_freq is not None:
        gaps = combined.index.to_series().diff()
        big_gaps = gaps[gaps > expected_freq * 5]
        if len(big_gaps) > 0:
            print(f"Note: {len(big_gaps)} time gaps larger than 5x the normal "
                  "bar spacing were found (likely weekends/missing days/market closures). "
                  "This is expected for daily-downloaded forex/gold data -- just "
                  "be aware of it before you compute rolling statistics across gaps.")

    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, required=True,
                         help='e.g. "XAU-USD_1Minute_*.csv"')
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    combined = combine(args.input_dir, args.pattern)
    combined.to_csv(args.out)
    print(f"Saved combined dataset to {args.out}")


if __name__ == "__main__":
    main()
