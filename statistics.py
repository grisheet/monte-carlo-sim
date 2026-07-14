"""Descriptive and performance statistics for simulation results.

All estimators accept optional importance-sampling weights and reduce to the
standard unweighted formulas when weights are uniform.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from config import SimulationConfig
from simulator.engine import SimulationResult


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.average(x, weights=w))


def _wvar(x: np.ndarray, w: np.ndarray) -> float:
    m = _wmean(x, w)
    return float(np.average((x - m) ** 2, weights=w))


def weighted_quantile(x: np.ndarray, q, w: np.ndarray) -> np.ndarray:
    """Weighted quantiles (linear interpolation on the weighted CDF)."""
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cum = np.cumsum(ws) - 0.5 * ws
    cum /= ws.sum()
    return np.interp(np.atleast_1d(q), cum, xs)


def summary_statistics(res: SimulationResult) -> dict[str, float]:
    """Full summary table for terminal prices and simple returns."""
    cfg = res.config
    s_t = res.terminal_prices
    w = res.weights
    ret = s_t / cfg.initial_price - 1.0            # simple return over horizon
    T = cfg.horizon_years

    mean_price = _wmean(s_t, w)
    var_price = _wvar(s_t, w)
    mean_ret = _wmean(ret, w)
    std_ret = np.sqrt(_wvar(ret, w))

    # Annualized return/vol from horizon return distribution
    ann_ret = (1.0 + mean_ret) ** (1.0 / T) - 1.0 if T > 0 else mean_ret
    log_ret = np.log(np.maximum(s_t, 1e-300) / cfg.initial_price)
    ann_vol = np.sqrt(_wvar(log_ret, w) / T) if T > 0 else np.sqrt(_wvar(log_ret, w))

    downside = np.minimum(ret - 0.0, 0.0)
    semi_var = float(np.average(downside**2, weights=w))
    downside_dev = np.sqrt(semi_var)

    rf_T = (1.0 + cfg.risk_free_rate) ** T - 1.0
    sharpe = (mean_ret - rf_T) / std_ret if std_ret > 0 else np.nan
    sortino = (mean_ret - rf_T) / downside_dev if downside_dev > 0 else np.nan

    mdd = res.max_drawdown
    mean_mdd = _wmean(mdd, w)
    calmar = ann_ret / mean_mdd if mean_mdd > 0 else np.nan

    # Ulcer index on the mean path (percent drawdown RMS)
    mp = res.mean_path
    dd_pct = 100.0 * (1.0 - mp / np.maximum.accumulate(mp))
    ulcer = float(np.sqrt(np.mean(dd_pct**2)))

    pcts = weighted_quantile(s_t, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99], w)

    n_eff = float(w.sum() ** 2 / np.sum(w**2))     # effective sample size
    se = np.sqrt(var_price / n_eff)

    return {
        "n_paths": float(s_t.size),
        "effective_n": n_eff,
        "mean_terminal": mean_price,
        "median_terminal": float(weighted_quantile(s_t, 0.5, w)[0]),
        "std_terminal": float(np.sqrt(var_price)),
        "variance_terminal": var_price,
        "skewness": float(sps.skew(s_t)),
        "kurtosis_excess": float(sps.kurtosis(s_t)),
        "min_terminal": float(s_t.min()),
        "max_terminal": float(s_t.max()),
        "p01": float(pcts[0]), "p05": float(pcts[1]), "p10": float(pcts[2]),
        "p25": float(pcts[3]), "p50": float(pcts[4]), "p75": float(pcts[5]),
        "p90": float(pcts[6]), "p95": float(pcts[7]), "p99": float(pcts[8]),
        "expected_return": mean_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "semi_variance": semi_var,
        "downside_deviation": downside_dev,
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "mean_max_drawdown": mean_mdd,
        "worst_max_drawdown": float(mdd.max()),
        "ulcer_index": ulcer,
        "std_error_mean": float(se),
        "ci95_low": mean_price - 1.96 * se,
        "ci95_high": mean_price + 1.96 * se,
    }


def confidence_interval(
    x: np.ndarray, w: np.ndarray, level: float = 0.95
) -> tuple[float, float]:
    """CLT confidence interval for the Monte Carlo mean."""
    n_eff = w.sum() ** 2 / np.sum(w**2)
    m = _wmean(x, w)
    se = np.sqrt(_wvar(x, w) / n_eff)
    z = sps.norm.ppf(0.5 + level / 2.0)
    return m - z * se, m + z * se


def convergence_table(res: SimulationResult) -> list[dict[str, float]]:
    """Mean estimate and standard error at growing sample sizes."""
    s_t = res.terminal_prices
    out = []
    n = s_t.size
    k = 100
    while k <= n:
        sub = s_t[:k]
        out.append({
            "n": k,
            "mean": float(sub.mean()),
            "std_error": float(sub.std(ddof=1) / np.sqrt(k)),
        })
        k *= 2
    if not out or out[-1]["n"] != n:
        out.append({
            "n": n,
            "mean": float(s_t.mean()),
            "std_error": float(s_t.std(ddof=1) / np.sqrt(n)),
        })
    return out
