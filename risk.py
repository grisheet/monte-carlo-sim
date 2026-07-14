"""Risk analytics: VaR, CVaR/Expected Shortfall, and event probabilities.

Conventions
-----------
* Returns are simple horizon returns: R = S_T / S_0 - 1.
* VaR at level alpha (e.g. 95%) is reported as a POSITIVE loss fraction:
      VaR_a = -Quantile_{1-a}(R)
* CVaR (= Expected Shortfall) is the mean loss beyond VaR, also positive.
* Barrier probabilities (stop-loss / take-profit / bankruptcy) use the
  per-path running min/max collected by the engine, i.e. they are true
  first-passage probabilities on the daily grid, not terminal-only checks.
"""

from __future__ import annotations

import numpy as np

from simulator.engine import SimulationResult
from simulator.statistics import weighted_quantile


def var_cvar(
    returns: np.ndarray, weights: np.ndarray, level: float = 0.95
) -> tuple[float, float]:
    """(VaR, CVaR) at confidence ``level``, as positive loss fractions."""
    q = float(weighted_quantile(returns, 1.0 - level, weights)[0])
    var = -q
    tail = returns <= q
    if tail.any():
        cvar = -float(np.average(returns[tail], weights=weights[tail]))
    else:
        cvar = var
    return var, cvar


def probability(mask: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(mask.astype(float), weights=weights))


def risk_report(
    res: SimulationResult,
    stop_loss_pct: float = 0.20,
    take_profit_pct: float = 0.30,
    bankruptcy_threshold_pct: float = 0.90,
) -> dict[str, float]:
    """All headline risk numbers for the dashboard and reports."""
    cfg = res.config
    s0 = cfg.initial_price
    ret = res.terminal_prices / s0 - 1.0
    w = res.weights

    var95, cvar95 = var_cvar(ret, w, 0.95)
    var99, cvar99 = var_cvar(ret, w, 0.99)

    stop_level = s0 * (1.0 - stop_loss_pct)
    tp_level = s0 * (1.0 + take_profit_pct)
    ruin_level = s0 * (1.0 - bankruptcy_threshold_pct)

    return {
        "var_95": var95,
        "cvar_95": cvar95,
        "var_99": var99,
        "cvar_99": cvar99,
        "expected_shortfall_95": cvar95,          # alias, standard name
        "prob_loss": probability(ret < 0.0, w),
        "prob_gain_20": probability(ret >= 0.20, w),
        "prob_double": probability(res.terminal_prices >= 2.0 * s0, w),
        "prob_bankruptcy": probability(res.path_min <= ruin_level, w),
        "prob_hit_stop_loss": probability(res.path_min <= stop_level, w),
        "prob_hit_take_profit": probability(res.path_max >= tp_level, w),
        "stop_loss_level": stop_level,
        "take_profit_level": tp_level,
        "bankruptcy_level": ruin_level,
        "mean_max_drawdown": float(np.average(res.max_drawdown, weights=w)),
        "p95_max_drawdown": float(weighted_quantile(res.max_drawdown, 0.95, w)[0]),
    }
