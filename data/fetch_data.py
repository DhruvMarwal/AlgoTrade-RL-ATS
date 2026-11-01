"""
Downloads daily OHLCV data for a chosen asset and saves it as a clean CSV.

Usage:
    python data/fetch_data.py --ticker GC=F --start 2010-01-01 --end 2025-01-01 --out data/gold.csv

Common tickers:
    GC=F     Gold futures
    CL=F     Crude oil (WTI) futures
    NG=F     Natural gas futures
    SI=F     Silver futures
    XAUUSD=X Gold spot vs USD
"""

import argparse
import yfinance as yf
import pandas as pd


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            "Check the ticker symbol and your internet connection."
        )
    # yfinance sometimes returns MultiIndex columns for a single ticker; flatten if so
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df["log_return"] = (df["Close"] / df["Close"].shift(1)).apply(
        lambda x: None if pd.isna(x) else __import__("math").log(x)
    )
    df = df.dropna()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="GC=F")
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default="2025-01-01")
    parser.add_argument("--out", type=str, default="data/asset.csv")
    args = parser.parse_args()

    df = fetch(args.ticker, args.start, args.end)
    df.to_csv(args.out)
    print(f"Saved {len(df)} rows to {args.out}")
    print(df.head())


if __name__ == "__main__":
    main()
