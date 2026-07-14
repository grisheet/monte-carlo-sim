"""Correlation machinery for multi-asset simulation.

Correlated Gaussian shocks are produced with a Cholesky factor L of the
correlation matrix: if Z has iid N(0,1) columns, then Z @ L.T has correlation
matrix C = L L^T. A nearest-PSD repair (Higham-style eigenvalue clipping) is
applied when a user-entered matrix is slightly indefinite.
"""

from __future__ import annotations

import numpy as np

from simulator.validation import ValidationError, validate_correlation_matrix


def nearest_psd(corr: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Clip negative eigenvalues and renormalize the diagonal to 1."""
    corr = np.asarray(corr, dtype=float)
    sym = 0.5 * (corr + corr.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, eps, None)
    fixed = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def cholesky_factor(corr: np.ndarray) -> np.ndarray:
    """Cholesky factor of a (repaired if needed) correlation matrix."""
    try:
        validate_correlation_matrix(corr)
        return np.linalg.cholesky(np.asarray(corr, dtype=float))
    except (ValidationError, np.linalg.LinAlgError):
        repaired = nearest_psd(np.asarray(corr, dtype=float))
        validate_correlation_matrix(repaired)
        return np.linalg.cholesky(repaired)


def correlated_normals(
    rng: np.random.Generator, n_paths: int, n_steps: int, chol: np.ndarray
) -> np.ndarray:
    """Correlated standard normals of shape (n_paths, n_steps, n_assets)."""
    n_assets = chol.shape[0]
    z = rng.standard_normal((n_paths, n_steps, n_assets))
    return z @ chol.T


def corr_to_cov(corr: np.ndarray, vols: np.ndarray) -> np.ndarray:
    """Covariance matrix from correlations and per-asset volatilities."""
    vols = np.asarray(vols, dtype=float)
    return corr * np.outer(vols, vols)
