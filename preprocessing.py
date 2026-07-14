"""Parameter estimation from historical prices.

Estimates surfaced to the user:
* annualized drift (mean of log returns * 252, plus arithmetic version)
* annualized volatility (std of log returns * sqrt(252))
* rolling volatility (configurable window)
* EWMA volatility (RiskMetrics lambda = 0.94 by default)
* maximum-likelihood fits: Gaussian (exact) and Student-t (scipy MLE),
  plus MLE for OU speed/mean via the exact AR(1) mapping of the OU process.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from config import TRADING_DAYS_PER_YEAR


def log_returns(prices: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    r = np.diff(np.log(p))
    return r[np.isfinite(r)]


@dataclass
class EstimationReport:
    n_obs: int
    last_price: float
    ann_drift_log: float          # mean log return, annualized
    ann_drift_arith: float        # annualized arithmetic mean return
    ann_volatility: float
    ewma_volatility: float
    skewness: float
    kurtosis_excess: float
    t_df_mle: float               # Student-t degrees of freedom (MLE)
    ou_speed: float               # OU kappa from AR(1) fit on log price
    ou_mean_price: float          # OU long-run price level

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def rolling_volatility(
    prices: pd.Series, window: int = 21
) -> pd.Series:
    r = pd.Series(np.log(prices)).diff()
    return r.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def ewma_volatility(returns: np.ndarray, lam: float = 0.94) -> float:
    """RiskMetrics EWMA: sigma^2_t = lam*sigma^2_{t-1} + (1-lam)*r^2_{t-1}."""
    r2 = returns**2
    var = r2[: min(30, r2.size)].mean()
    for x in r2:
        var = lam * var + (1.0 - lam) * x
    return float(np.sqrt(var * TRADING_DAYS_PER_YEAR))


def ewma_volatility_series(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    r2 = returns**2
    out = np.empty_like(r2)
    var = r2[: min(30, r2.size)].mean()
    for i, x in enumerate(r2):
        var = lam * var + (1.0 - lam) * x
        out[i] = var
    return np.sqrt(out * TRADING_DAYS_PER_YEAR)


def fit_ou_ar1(log_prices: np.ndarray, dt: float) -> tuple[float, float]:
    """Exact-discretization MLE for OU speed and mean via AR(1) regression.

    X_{t+1} = c + phi X_t + eps  maps to  kappa = -ln(phi)/dt,
    x_bar = c / (1 - phi).
    """
    x = np.asarray(log_prices, float)
    x0, x1 = x[:-1], x[1:]
    phi, c = np.polyfit(x0, x1, 1)
    phi = float(np.clip(phi, 1e-6, 1 - 1e-6))
    kappa = -np.log(phi) / dt
    x_bar = c / (1.0 - phi)
    return float(kappa), float(np.exp(x_bar))


def estimate_parameters(prices: pd.Series | np.ndarray) -> EstimationReport:
    p = np.asarray(prices, dtype=float)
    r = log_returns(p)
    n = r.size
    ann = TRADING_DAYS_PER_YEAR

    mean_daily = r.mean()
    vol_daily = r.std(ddof=1)
    ann_vol = vol_daily * np.sqrt(ann)
    ann_drift_log = mean_daily * ann
    # Arithmetic drift consistent with GBM: mu = m + sigma^2/2
    ann_drift_arith = ann_drift_log + 0.5 * ann_vol**2

    # Student-t MLE (location-scale)
    try:
        df_mle, _, _ = sps.t.fit(r)
        df_mle = float(np.clip(df_mle, 2.1, 200.0))
    except Exception:
        df_mle = 5.0

    kappa, mean_price = fit_ou_ar1(np.log(p), dt=1.0 / ann)

    return EstimationReport(
        n_obs=n,
        last_price=float(p[-1]),
        ann_drift_log=float(ann_drift_log),
        ann_drift_arith=float(ann_drift_arith),
        ann_volatility=float(ann_vol),
        ewma_volatility=ewma_volatility(r),
        skewness=float(sps.skew(r)),
        kurtosis_excess=float(sps.kurtosis(r)),
        t_df_mle=df_mle,
        ou_speed=kappa,
        ou_mean_price=mean_price,
    )
