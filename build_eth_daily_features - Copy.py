
import numpy as np
import pandas as pd
from pathlib import Path

src = "data/eth_daily.csv"
out = "results/eth_features_all.csv"

df = pd.read_csv(src, parse_dates=["Date"])
df = df.sort_values("Date").drop_duplicates("Date")
df = df.set_index("Date")

close = pd.to_numeric(df["Close"], errors="coerce")
log_return = np.log(close).diff()

f = pd.DataFrame(index=df.index)
f["log_return"] = log_return
f["return_5d"] = close.pct_change(5)
f["return_10d"] = close.pct_change(10)
f["rolling_vol_10d"] = log_return.rolling(10).std()

delta = close.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
f["rsi_14"] = 100 - 100 / (1 + rs)

# Causal rolling-volatility proxy; not a fitted GARCH forecast.
f["garch_cond_vol"] = log_return.rolling(20).std()
vol_median = f["garch_cond_vol"].rolling(
    252, min_periods=60
).median()
f["regime_flag"] = (
    f["garch_cond_vol"] > vol_median
).astype(float)

# Trailing 20-day momentum, known at the daily close.
f["momentum_signal"] = close.pct_change(20)

f = f.replace([np.inf, -np.inf], np.nan).dropna()
f.index.name = "Date"

Path("results").mkdir(exist_ok=True)
f.to_csv(out)

print("Saved:", out)
print("Rows:", len(f))
print("Date range:", f.index.min(), "to", f.index.max())
print("Missing values:", int(f.isna().sum().sum()))
print(f.loc["2017":"2024"].groupby(f.index.year).size())
