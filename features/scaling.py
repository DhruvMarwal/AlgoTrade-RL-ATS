"""
Robust (median / IQR) scaler fitted on the TRAINING split only, then applied
unchanged to validation and test. Medians are not thrown off by crash days.
Binary columns (regime_flag) are passed through untouched.
"""

import numpy as np
import pandas as pd

PASSTHROUGH = ("regime_flag",)


class RobustScaler:
    def __init__(self, clip: float = 5.0):
        self.clip = clip
        self.median_ = None
        self.iqr_ = None

    def fit(self, df: pd.DataFrame):
        self.median_ = df.median()
        q75, q25 = df.quantile(0.75), df.quantile(0.25)
        iqr = (q75 - q25).replace(0, np.nan)
        self.iqr_ = iqr.fillna(df.std()).replace(0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = (df - self.median_[df.columns]) / self.iqr_[df.columns]
        out = out.clip(-self.clip, self.clip)
        for c in PASSTHROUGH:
            if c in df.columns:
                out[c] = df[c]
        return out.astype(np.float32)
