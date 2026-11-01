
import pandas as pd
from pathlib import Path

path = Path("data/ETHUSDT_5m.csv")
df = pd.read_csv(path)

# Rebuild dates from Binance timestamps if present in the original file.
# If the CSV only contains the already-converted Date column,
# use the expected 5-minute sequence as a diagnostic, not as a final fix.
if "timestamp" in df.columns:
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    unit = "us" if ts.dropna().median() > 1e14 else "ms"
    df["Date"] = pd.to_datetime(ts, unit=unit, utc=True).dt.tz_localize(None)
else:
    print("No original timestamp column found.")
    print("Re-download after correcting the downloader's timestamp unit.")
    raise SystemExit(1)

df = df.dropna(subset=["Date"]).drop_duplicates("Date").sort_values("Date")
df.to_csv(path, index=False)

print("Saved:", path)
print("Rows:", len(df))
print("From:", df["Date"].min())
print("To:", df["Date"].max())
