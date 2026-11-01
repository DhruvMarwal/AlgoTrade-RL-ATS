"""
Applied Time Series diagnostics for the trading project.

This module produces both:
  1) the plots/tests your ATS assignment asks for (stationarity, ACF/PACF,
     decomposition, ARIMA, GARCH, whitenoise checks), and
  2) the numeric features the RL agent consumes:
       garch_cond_vol  one-step-ahead volatility forecast, known at the close of day t
       regime_flag     1 = high-volatility regime, 0 = calm (causal rolling quantile)

LEAKAGE CONTROL
  The GARCH parameters are estimated on the TRAINING window only (<= --train_end).
  They are then frozen and the model is run forward over the whole sample, so no
  validation/test information reaches the parameters. The descriptive tests
  (ADF/KPSS, decomposition, ACF/PACF, ARIMA) use the full sample because they are
  reporting tools for the ATS assignment; nothing the RL agent sees depends on them.

Usage:
    python ats/diagnostics.py --csv data/gold.csv --train_end 2019-12-31
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


def load_series(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, index_col="Date", parse_dates=True)


def stationarity_tests(returns: pd.Series) -> dict:
    """ADF (null = unit root / non-stationary) and KPSS (null = stationary)."""
    adf_stat, adf_p, *_ = adfuller(returns.dropna())
    kpss_stat, kpss_p, *_ = kpss(returns.dropna(), nlags="auto")
    return {
        "adf_stat": float(adf_stat), "adf_pvalue": float(adf_p),
        "adf_stationary": bool(adf_p < 0.05),
        "kpss_stat": float(kpss_stat), "kpss_pvalue": float(kpss_p),
        "kpss_stationary": bool(kpss_p > 0.05),
    }


def whitenoise_test(residuals: pd.Series, lags=10) -> pd.DataFrame:
    """Ljung-Box test. Large p-values -> residuals look like white noise."""
    return acorr_ljungbox(residuals.dropna(), lags=[lags], return_df=True)


def plot_seasonal_decomposition(series: pd.Series, period: int, out_path: str):
    result = seasonal_decompose(series.dropna(), model="additive", period=period)
    fig = result.plot()
    fig.set_size_inches(10, 8)
    fig.savefig(out_path)
    plt.close(fig)


def plot_acf_pacf(returns: pd.Series, lags: int, out_path: str):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(returns.dropna(), lags=lags, ax=axes[0])
    plot_pacf(returns.dropna(), lags=lags, ax=axes[1])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fit_arima(returns: pd.Series, order=(1, 0, 1)):
    """Fit an ARMA/ARIMA model on returns. Order = (p, d, q)."""
    return ARIMA(returns.dropna(), order=order).fit()


def fit_garch_causal(returns: pd.Series, train_end: str):
    """
    GARCH(1,1) with Student-t errors, estimated on the training window only.

    Returns
      res_train : fitted results (for printing / the ATS report)
      vol_next  : Series indexed by date t holding the volatility forecast for day t+1
                  made with information up to and including the close of day t
                  (sigma^2_{t+1} = omega + alpha * eps_t^2 + beta * sigma^2_t),
                  expressed in return units (not percent).
    """
    r = returns.dropna() * 100.0                      # arch works best in percent
    train = r.loc[:train_end]
    if len(train) < 500:
        raise ValueError(f"Only {len(train)} training observations before {train_end}; "
                         "pick a later --train_end.")

    res_train = arch_model(train, mean="Constant", vol="Garch", p=1, q=1, dist="t").fit(disp="off")

    # Frozen parameters, run forward over the whole sample (no re-estimation)
    res_full = arch_model(r, mean="Constant", vol="Garch", p=1, q=1, dist="t").fix(res_train.params)
    sigma_t = res_full.conditional_volatility          # sigma_t: forecast of day t made at t-1

    mu = res_train.params["mu"]
    omega = res_train.params["omega"]
    alpha = res_train.params["alpha[1]"]
    beta = res_train.params["beta[1]"]
    eps_t = r - mu
    sigma2_next = omega + alpha * eps_t ** 2 + beta * sigma_t ** 2
    vol_next = np.sqrt(sigma2_next) / 100.0
    return res_train, vol_next.rename("garch_cond_vol")


def regime_flag(vol: pd.Series, window=60, threshold_quantile=0.75) -> pd.Series:
    """1 = volatility above its own trailing 75th percentile (causal), else 0."""
    thr = vol.rolling(window, min_periods=window // 2).quantile(threshold_quantile)
    return (vol > thr).astype(int).fillna(0).rename("regime_flag")


def run_full_diagnostics(csv_path: str, seasonal_period: int, lags: int, train_end: str):
    df = load_series(csv_path)
    returns = df["log_return"]

    print("=== Stationarity tests on returns ===")
    print(stationarity_tests(returns))

    print("\n=== Seasonal decomposition (results/seasonal_decomposition.png) ===")
    plot_seasonal_decomposition(df["Close"], seasonal_period, "results/seasonal_decomposition.png")

    print("=== ACF / PACF (results/acf_pacf.png) ===")
    plot_acf_pacf(returns, lags, "results/acf_pacf.png")

    print("\n=== ARIMA(1,0,1) fit on returns ===")
    arima_fit = fit_arima(returns, order=(1, 0, 1))
    print(arima_fit.summary())
    print("\n=== Whitenoise check on ARIMA residuals (Ljung-Box) ===")
    print(whitenoise_test(arima_fit.resid))

    print(f"\n=== GARCH(1,1)-t, parameters estimated on data up to {train_end} ONLY ===")
    garch_train, vol_next = fit_garch_causal(returns, train_end)
    print(garch_train.summary())
    p = garch_train.params
    print(f"\nalpha + beta (persistence) = {p['alpha[1]'] + p['beta[1]']:.4f}")

    regimes = regime_flag(vol_next)
    print(f"High-volatility regime days: {int(regimes.sum())} / {len(regimes)}")

    out = pd.DataFrame({
        "log_return": returns,
        "garch_cond_vol": vol_next,
        "regime_flag": regimes,
    }).dropna()
    out.index.name = "Date"
    out.to_csv("results/ats_features.csv")
    print("\nSaved ATS-derived features to results/ats_features.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--seasonal_period", type=int, default=252)
    parser.add_argument("--lags", type=int, default=40)
    parser.add_argument("--train_end", type=str, default="2019-12-31",
                        help="last date of the training window; GARCH is fitted on data up to here only")
    args = parser.parse_args()
    run_full_diagnostics(args.csv, args.seasonal_period, args.lags, args.train_end)
