import numpy as np
import pandas as pd
from pathlib import Path

src = "data/XAU_5m_data.csv"
out = "results/features_5m_2023_2025.csv"

df = pd.read_csv(src, sep=";")
df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M", errors="coerce")
for c in ["Open", "High", "Low", "Close", "Volume"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates("Date")
df = df.set_index("Date")
close = df["Close"]
lr = np.log(close).diff()

f = pd.DataFrame(index=df.index)
f["log_return"] = lr
f["return_5d"] = close.pct_change(5)
f["return_10d"] = close.pct_change(10)
f["rolling_vol_10d"] = lr.rolling(10).std()
delta = close.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
f["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
# Rolling-volatility proxy, NOT a fitted GARCH forecast.
f["garch_cond_vol"] = lr.rolling(20).std()
ref = f["garch_cond_vol"].rolling(288, min_periods=100).median()
f["regime_flag"] = (f["garch_cond_vol"] > ref).astype(float)
f["momentum_signal"] = close.pct_change(20)

f = f.replace([np.inf, -np.inf], np.nan).dropna()
f = f[f.index >= "2023-01-01"]
f.index.name = "Date"
Path("results").mkdir(exist_ok=True)
f.to_csv(out)

print("Saved:", out, "| Rows:", len(f))
for y in [2023, 2024, 2025]:
    z = f[f.index.year == y]
    print(y, len(z), z.index.min() if len(z) else "-", z.index.max() if len(z) else "-")
