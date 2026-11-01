
from pathlib import Path
from io import BytesIO
import urllib.request
import zipfile

import pandas as pd

out = Path("data/eth_daily.csv")
out.parent.mkdir(exist_ok=True)

frames = []

for year in range(2017, 2025):
    for month in range(1, 13):
        filename = f"ETHUSDT-1d-{year}-{month:02d}.zip"
        url = (
            "https://data.binance.vision/data/spot/monthly/"
            f"klines/ETHUSDT/1d/{filename}"
        )

        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                z = zipfile.ZipFile(BytesIO(response.read()))

            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, header=None)

            # Binance timestamp may be milliseconds or microseconds.
            ts = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            unit = "us" if ts.dropna().median() > 1e14 else "ms"

            daily = pd.DataFrame({
                "Date": pd.to_datetime(
                    ts, unit=unit, utc=True
                ).dt.tz_localize(None),
                "Open": df.iloc[:, 1],
                "High": df.iloc[:, 2],
                "Low": df.iloc[:, 3],
                "Close": df.iloc[:, 4],
                "Volume": df.iloc[:, 5],
            })

            frames.append(daily)
            print("Downloaded:", filename, len(daily))

        except Exception as e:
            print("Skipped:", filename, "|", e)

if not frames:
    raise RuntimeError("No data downloaded.")

result = pd.concat(frames, ignore_index=True)
result = (
    result.dropna(subset=["Date", "Close"])
    .drop_duplicates("Date")
    .sort_values("Date")
)

result.to_csv(out, index=False)

print("\nSaved:", out)
print("Rows:", len(result))
print("Date range:", result["Date"].min(), "to", result["Date"].max())
print("Duplicate dates:", result["Date"].duplicated().sum())
