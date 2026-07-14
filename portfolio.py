"""Multi-asset portfolio simulation and mean-variance analysis.

* Correlated GBM for N assets (Cholesky-coupled shocks).
* Buy-and-hold or periodic rebalancing to target weights.
* Analytic mean-variance efficient frontier (long-only optional off; we use
  the closed-form frontier with a Monte Carlo cloud of random portfolios,
  which is robust without requiring an external optimizer).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from simulator.correlation import cholesky_factor, corr_to_cov


@dataclass
class PortfolioSpec:
    names: list[str]
    s0: np.ndarray                # (n_assets,) initial prices
    mu: np.ndarray                # (n_assets,) annual drifts
    sigma: np.ndarray             # (n_assets,) annual vols
    corr: np.ndarray              # (n_assets, n_assets)
    weights: np.ndarray           # (n_assets,) target weights, sum to 1
    rebalance_every: int = 0      # steps between rebalances; 0 = buy & hold
    initial_value: float = 100_000.0

    def __post_init__(self):
        for attr in ("s0", "mu", "sigma", "weights"):
            setattr(self, attr, np.asarray(getattr(self, attr), dtype=float))
        self.corr = np.asarray(self.corr, dtype=float)
        total = self.weights.sum()
        if total > 0:
            self.weights = self.weights / total

    @classmethod
    def equal_weight(cls, **kwargs) -> "PortfolioSpec":
        n = len(kwargs["names"])
        kwargs["weights"] = np.full(n, 1.0 / n)
        return cls(**kwargs)


@dataclass
class PortfolioResult:
    time_grid: np.ndarray
    value_paths: np.ndarray       # (n_kept, n_steps + 1)
    terminal_values: np.ndarray   # (n_paths,)
    asset_terminals: np.ndarray   # (n_paths, n_assets)
    max_drawdown: np.ndarray
    quantiles: np.ndarray
    mean_path: np.ndarray
    spec: PortfolioSpec = field(repr=False, default=None)


def simulate_portfolio(
    spec: PortfolioSpec,
    n_paths: int,
    horizon_years: float,
    trading_days: int = 252,
    seed: int | None = 42,
    keep_paths: int = 300,
) -> PortfolioResult:
    """Vectorized correlated-GBM portfolio simulation."""
    n_assets = len(spec.names)
    n_steps = max(1, int(round(horizon_years * trading_days)))
    dt = horizon_years / n_steps
    rng = np.random.default_rng(seed)

    chol = cholesky_factor(spec.corr)
    z = rng.standard_normal((n_paths, n_steps, n_assets)) @ chol.T

    drift = (spec.mu - 0.5 * spec.sigma**2) * dt
    shock = spec.sigma * np.sqrt(dt)
    log_inc = drift[None, None, :] + shock[None, None, :] * z
    gross = np.exp(log_inc)                          # per-step price relatives

    values = np.empty((n_paths, n_steps + 1))
    values[:, 0] = spec.initial_value
    holdings_value = spec.initial_value * spec.weights[None, :] * np.ones((n_paths, 1))

    reb = spec.rebalance_every
    for t in range(n_steps):
        holdings_value = holdings_value * gross[:, t, :]
        total = holdings_value.sum(axis=1)
        values[:, t + 1] = total
        if reb and (t + 1) % reb == 0:
            holdings_value = total[:, None] * spec.weights[None, :]

    running_max = np.maximum.accumulate(values, axis=1)
    mdd = (1.0 - values / running_max).max(axis=1)

    keep = min(keep_paths, n_paths)
    sel = np.linspace(0, n_paths - 1, keep).astype(int)
    qs = (0.05, 0.25, 0.50, 0.75, 0.95)

    asset_term = spec.s0[None, :] * np.exp(log_inc.sum(axis=1))

    return PortfolioResult(
        time_grid=np.linspace(0, horizon_years, n_steps + 1),
        value_paths=values[sel],
        terminal_values=values[:, -1],
        asset_terminals=asset_term,
        max_drawdown=mdd,
        quantiles=np.quantile(values, qs, axis=0),
        mean_path=values.mean(axis=0),
        spec=spec,
    )


# --------------------------------------------------------------------------- #
# Mean-variance frontier
# --------------------------------------------------------------------------- #
def efficient_frontier(
    mu: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    n_points: int = 50,
    n_random: int = 3_000,
    risk_free_rate: float = 0.0,
    seed: int | None = 7,
) -> dict:
    """Closed-form unconstrained frontier + random long-only portfolio cloud.

    Returns dict with frontier (vol, ret), random cloud (vol, ret, sharpe,
    weights), max-Sharpe and min-variance portfolios (from the cloud, i.e.
    long-only).
    """
    mu = np.asarray(mu, float)
    cov = corr_to_cov(np.asarray(corr, float), np.asarray(sigma, float))
    n = mu.size
    rng = np.random.default_rng(seed)

    # Random long-only portfolios (Dirichlet = uniform on the simplex).
    w_cloud = rng.dirichlet(np.ones(n), size=n_random)
    ret_cloud = w_cloud @ mu
    vol_cloud = np.sqrt(np.einsum("ij,jk,ik->i", w_cloud, cov, w_cloud))
    sharpe_cloud = (ret_cloud - risk_free_rate) / np.maximum(vol_cloud, 1e-12)

    i_ms = int(np.argmax(sharpe_cloud))
    i_mv = int(np.argmin(vol_cloud))

    # Analytic unconstrained frontier for reference curve.
    inv = np.linalg.pinv(cov)
    ones = np.ones(n)
    a = ones @ inv @ ones
    b = ones @ inv @ mu
    c = mu @ inv @ mu
    d = a * c - b**2
    targets = np.linspace(ret_cloud.min(), ret_cloud.max(), n_points)
    if abs(d) > 1e-12:
        frontier_var = (a * targets**2 - 2 * b * targets + c) / d
        frontier_vol = np.sqrt(np.maximum(frontier_var, 0.0))
    else:
        frontier_vol = np.full_like(targets, np.nan)

    return {
        "frontier_ret": targets,
        "frontier_vol": frontier_vol,
        "cloud_ret": ret_cloud,
        "cloud_vol": vol_cloud,
        "cloud_sharpe": sharpe_cloud,
        "cloud_weights": w_cloud,
        "max_sharpe": {
            "weights": w_cloud[i_ms], "ret": float(ret_cloud[i_ms]),
            "vol": float(vol_cloud[i_ms]), "sharpe": float(sharpe_cloud[i_ms]),
        },
        "min_variance": {
            "weights": w_cloud[i_mv], "ret": float(ret_cloud[i_mv]),
            "vol": float(vol_cloud[i_mv]), "sharpe": float(sharpe_cloud[i_mv]),
        },
    }
