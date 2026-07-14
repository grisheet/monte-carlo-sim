"""Pricing utilities: Monte Carlo expectations and Black-Scholes benchmarks.

The Monte Carlo expectation of a payoff f is

    E[f(S_T)] ≈ (1/N) Σ f(S_T^(i)),   SE = std(f) / sqrt(N)

Discounted at the risk-free rate this gives a price. The closed-form
Black-Scholes price is provided as a validation benchmark: under risk-neutral
GBM the simulator's call/put prices must converge to Black-Scholes, which the
test suite verifies.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def mc_expectation(payoffs: np.ndarray, weights: np.ndarray | None = None
                   ) -> tuple[float, float]:
    """(mean, standard error) of a Monte Carlo estimate, weight-aware."""
    if weights is None:
        weights = np.ones_like(payoffs)
    m = float(np.average(payoffs, weights=weights))
    var = float(np.average((payoffs - m) ** 2, weights=weights))
    n_eff = weights.sum() ** 2 / np.sum(weights**2)
    return m, float(np.sqrt(var / n_eff))


def discount(value: float, rate: float, T: float) -> float:
    return value * np.exp(-rate * T)


def mc_option_price(
    terminal_prices: np.ndarray,
    strike: float,
    rate: float,
    T: float,
    kind: str = "call",
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """European option price and standard error from simulated terminals."""
    if kind == "call":
        payoff = np.maximum(terminal_prices - strike, 0.0)
    elif kind == "put":
        payoff = np.maximum(strike - terminal_prices, 0.0)
    else:
        raise ValueError("kind must be 'call' or 'put'")
    mean, se = mc_expectation(payoff, weights)
    df = np.exp(-rate * T)
    return mean * df, se * df


def black_scholes(
    s0: float, strike: float, rate: float, sigma: float, T: float,
    kind: str = "call", dividend_yield: float = 0.0,
) -> float:
    """Closed-form Black-Scholes-Merton price."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(s0 - strike, 0.0) if kind == "call" else max(strike - s0, 0.0)
        return float(intrinsic)
    d1 = (np.log(s0 / strike) + (rate - dividend_yield + 0.5 * sigma**2) * T) / (
        sigma * np.sqrt(T)
    )
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return float(
            s0 * np.exp(-dividend_yield * T) * sps.norm.cdf(d1)
            - strike * np.exp(-rate * T) * sps.norm.cdf(d2)
        )
    return float(
        strike * np.exp(-rate * T) * sps.norm.cdf(-d2)
        - s0 * np.exp(-dividend_yield * T) * sps.norm.cdf(-d1)
    )
