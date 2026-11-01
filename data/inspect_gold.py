
import pandas as pd

file = r"C:\Users\dhruv\Downloads\rl_trading_project4\rl_trading_project\data\XAU_5m_data.csv"

# Read the CSV without modifying the original
df = pd.read_csv(file, sep=";")


# Parse dates correctly
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%Y.%m.%d %H:%M",
    errors="coerce"
)

df = df.sort_values("Date").reset_index(drop=True)

print("\n--- DATA QUALITY CHECK ---")

# 1. Invalid dates
print("Invalid dates:", df["Date"].isna().sum())

# 2. Duplicate timestamps
print("Duplicate timestamps:", df["Date"].duplicated().sum())

# 3. Invalid OHLC candles
invalid_ohlc = (
    (df["High"] < df["Low"]) |
    (df["High"] < df["Open"]) |
    (df["High"] < df["Close"]) |
    (df["Low"] > df["Open"]) |
    (df["Low"] > df["Close"])
)
print("Invalid OHLC candles:", invalid_ohlc.sum())

# 4. Zero or negative prices
print("Non-positive closing prices:",
      (df["Close"] <= 0).sum())

# 5. Largest time gaps
gaps = df["Date"].diff()
gap_table = pd.DataFrame({
    "Previous candle": df["Date"].shift(1),
    "Next candle": df["Date"],
    "Gap": gaps
})

print("\nLargest 15 time gaps:")
print(
    gap_table.nlargest(15, "Gap").to_string(index=False)
)

# 6. Check the final date of each year
print("\nLast available candle per year:")
print(
    df.groupby(df["Date"].dt.year)["Date"].max()
)

# 7. Price range
print("\nPrice summary:")
print(df[["Open", "High", "Low", "Close"]].describe())


print("Total rows:", len(df))
print("Columns:", df.columns.tolist())
print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

print("\nLast 10 rows:")
print(df.tail(10).to_string(index=False))

print("\nMissing values per column:")
print(df.isna().sum())

# Change the column name if your CSV uses a different name
date_col = next(
    (c for c in df.columns if c.lower() in ["date", "datetime", "timestamp", "time"]),
    None
)

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    print("\nDate range:", df[date_col].min(), "to", df[date_col].max())
    print("\nRows per year:")
    print(df.groupby(df[date_col].dt.year).size())
