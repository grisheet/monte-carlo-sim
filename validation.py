"""Validation of simulation inputs.

All user-facing entry points funnel through :func:`validate_config`, which
raises :class:`ValidationError` with a human-readable message list.
"""

from __future__ import annotations

import math

import numpy as np

from config import (
    BOOTSTRAP_METHODS,
    DISTRIBUTIONS,
    MODELS,
    RNG_ENGINES,
    VARIANCE_REDUCTION,
    SimulationConfig,
)


class ValidationError(ValueError):
    """Raised when a configuration is not simulatable."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def validate_config(cfg: SimulationConfig) -> None:
    """Raise :class:`ValidationError` if ``cfg`` cannot be simulated safely."""
    errors: list[str] = []

    if not _finite(cfg.initial_price) or cfg.initial_price <= 0:
        errors.append("Initial price must be a positive, finite number.")
    if not _finite(cfg.sigma) or cfg.sigma < 0:
        errors.append("Volatility cannot be negative or NaN.")
    if not _finite(cfg.mu):
        errors.append("Expected return must be a finite number.")
    if not _finite(cfg.horizon_years) or cfg.horizon_years <= 0:
        errors.append("Time horizon must be positive.")
    if cfg.trading_days <= 0:
        errors.append("Trading days per year must be positive.")
    if cfg.n_simulations <= 0:
        errors.append("Number of simulations must be at least 1.")
    if cfg.n_simulations > 2_000_000:
        errors.append("Number of simulations capped at 2,000,000 for memory safety.")
    if cfg.model not in MODELS:
        errors.append(f"Unknown model '{cfg.model}'.")
    if cfg.rng_engine not in RNG_ENGINES:
        errors.append(f"Unknown RNG engine '{cfg.rng_engine}'.")
    if cfg.variance_reduction not in VARIANCE_REDUCTION:
        errors.append(f"Unknown variance reduction '{cfg.variance_reduction}'.")
    if cfg.distribution.name not in DISTRIBUTIONS:
        errors.append(f"Unknown distribution '{cfg.distribution.name}'.")
    if cfg.bootstrap.method not in BOOTSTRAP_METHODS:
        errors.append(f"Unknown bootstrap method '{cfg.bootstrap.method}'.")

    if cfg.model == "jump_diffusion":
        j = cfg.jumps
        if not _finite(j.intensity) or j.intensity < 0:
            errors.append("Jump intensity (lambda) must be >= 0.")
        if not _finite(j.volatility) or j.volatility < 0:
            errors.append("Jump volatility must be >= 0.")

    if cfg.model == "heston":
        h = cfg.heston
        if h.kappa < 0 or h.theta < 0 or h.xi < 0 or h.v0 < 0:
            errors.append("Heston kappa, theta, xi and v0 must be non-negative.")
        if not (-1.0 <= h.rho <= 1.0):
            errors.append("Heston correlation rho must lie in [-1, 1].")
        if 2 * h.kappa * h.theta < h.xi**2:
            # Not fatal (full truncation handles it), but worth flagging.
            pass

    if cfg.model == "variance_gamma":
        v = cfg.vg
        if v.nu <= 0:
            errors.append("Variance Gamma nu must be strictly positive.")
        if v.sigma < 0:
            errors.append("Variance Gamma sigma must be >= 0.")
        # Martingale correction requires 1 - theta*nu - sigma^2*nu/2 > 0
        if 1.0 - v.theta * v.nu - 0.5 * v.sigma**2 * v.nu <= 0:
            errors.append(
                "Variance Gamma parameters violate 1 - theta*nu - sigma^2*nu/2 > 0 "
                "(martingale correction undefined)."
            )

    if cfg.model == "mean_reversion":
        o = cfg.ou
        if o.speed < 0:
            errors.append("OU mean-reversion speed must be >= 0.")
        if o.mean <= 0:
            errors.append("OU long-run price level must be positive.")
        if o.volatility < 0:
            errors.append("OU volatility must be >= 0.")

    if cfg.distribution.name == "student_t" and cfg.distribution.student_t_df <= 2:
        errors.append("Student t degrees of freedom must exceed 2 (finite variance).")

    if errors:
        raise ValidationError(errors)


def validate_correlation_matrix(corr: np.ndarray) -> None:
    """Validate a correlation matrix for portfolio simulation."""
    errors: list[str] = []
    corr = np.asarray(corr, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValidationError(["Correlation matrix must be square."])
    if np.any(~np.isfinite(corr)):
        errors.append("Correlation matrix contains NaN or infinite values.")
    if not np.allclose(corr, corr.T, atol=1e-8):
        errors.append("Correlation matrix must be symmetric.")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
        errors.append("Correlation matrix diagonal must be 1.")
    if np.any(corr < -1 - 1e-12) or np.any(corr > 1 + 1e-12):
        errors.append("Correlations must lie in [-1, 1].")
    if not errors:
        eigvals = np.linalg.eigvalsh(corr)
        if eigvals.min() < -1e-8:
            errors.append("Correlation matrix is not positive semi-definite.")
    if errors:
        raise ValidationError(errors)


def validate_price_series(prices: np.ndarray) -> None:
    """Validate a historical price series used for estimation/bootstrap."""
    prices = np.asarray(prices, dtype=float)
    errors: list[str] = []
    if prices.size < 30:
        errors.append("Need at least 30 historical prices for estimation.")
    if np.any(~np.isfinite(prices)):
        errors.append("Price series contains NaN or infinite values.")
    elif np.any(prices <= 0):
        errors.append("Price series contains non-positive values.")
    if errors:
        raise ValidationError(errors)
