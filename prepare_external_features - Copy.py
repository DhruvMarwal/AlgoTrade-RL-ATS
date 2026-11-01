
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features.build_features import build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frequency", choices=["daily", "15m"], required=True)
    parser.add_argument("--train_end", required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.input)

    if "time" not in source.columns:
        raise ValueError("Expected a 'time' timestamp column.")

    df = source.copy()
    df["Date"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col.title()] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .drop_duplicates("Date")
        .set_index("Date")
    )

    if args.frequency == "15m":
        raise ValueError(
            "The current RL pipeline uses daily-style features. "
            "Do not use the 15-minute dataset until the intraday pipeline is adapted."
        )

    price = pd.DataFrame(index=df.index)
    price["Close"] = df["Close"]
    price["log_return"] = np.log(price["Close"]).diff()

    # Build an asset-specific volatility proxy.
    # This is a rolling estimate, not a fitted GARCH forecast.
    vol = price["log_return"].rolling(20).std()
    vol_threshold = vol.rolling(60, min_periods=30).quantile(0.75)

    ats = pd.DataFrame(index=df.index)
    ats["garch_cond_vol"] = vol
    ats["regime_flag"] = (vol > vol_threshold).astype(float)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Temporary files are isolated by asset under the output directory.
    temp_dir = out_path.parent / "_feature_inputs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    price_path = temp_dir / "price.csv"
    ats_path = temp_dir / "ats.csv"

    price.to_csv(price_path, index_label="Date")
    ats.to_csv(ats_path, index_label="Date")

    features = build(str(price_path), str(ats_path))
    features.to_csv(out_path, index_label="Date")

    print(f"Saved features: {out_path}")
    print(f"Rows: {len(features)}")
    print(f"Date range: {features.index.min()} -> {features.index.max()}")
    print(f"Columns: {features.columns.tolist()}")


if __name__ == "__main__":
    main()
