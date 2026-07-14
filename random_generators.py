"""Random number generation for Monte Carlo simulation.

Provides a single façade, :class:`InnovationGenerator`, that yields matrices of
standardized innovations (mean 0, variance 1) of shape ``(n_paths, n_steps)``
from any combination of:

* engine: PCG64, Mersenne Twister, Sobol, Halton, Latin Hypercube
* distribution: normal, Student t, Laplace, lognormal, uniform, skew normal,
  generalized error distribution (GED)
* variance reduction: antithetic variates (control variates and importance
  sampling are applied at the estimator level in :mod:`simulator.engine`).

Quasi-random engines generate uniforms which are mapped through the inverse
CDF of the requested distribution, preserving their low-discrepancy structure.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import qmc


# --------------------------------------------------------------------------- #
# Uniform sources
# --------------------------------------------------------------------------- #
def _pseudo_rng(engine: str, seed: int | None) -> np.random.Generator:
    if engine == "mt19937":
        return np.random.Generator(np.random.MT19937(seed))
    return np.random.Generator(np.random.PCG64(seed))


def _quasi_uniforms(engine: str, n: int, d: int, seed: int | None) -> np.ndarray:
    """Low-discrepancy uniforms in (0, 1), shape (n, d)."""
    d = max(1, d)
    if engine == "sobol":
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        u = sampler.random(n)
    elif engine == "halton":
        sampler = qmc.Halton(d=d, scramble=True, seed=seed)
        u = sampler.random(n)
    elif engine == "lhs":
        sampler = qmc.LatinHypercube(d=d, seed=seed)
        u = sampler.random(n)
    else:  # pragma: no cover - guarded by validation
        raise ValueError(f"Unknown quasi-random engine '{engine}'")
    # Clip away exact 0/1 which would map to +-inf under inverse CDFs.
    return np.clip(u, 1e-12, 1 - 1e-12)


# --------------------------------------------------------------------------- #
# Distributions (all standardized to mean 0, variance 1)
# --------------------------------------------------------------------------- #
def _standardized_ppf(name: str, u: np.ndarray, params) -> np.ndarray:
    """Inverse-CDF transform of uniforms, standardized to unit variance."""
    if name == "normal":
        return stats.norm.ppf(u)
    if name == "student_t":
        df = params.student_t_df
        scale = np.sqrt((df - 2.0) / df)  # variance of t is df/(df-2)
        return stats.t.ppf(u, df) * scale
    if name == "laplace":
        return stats.laplace.ppf(u) / np.sqrt(2.0)  # laplace var = 2b^2, b=1
    if name == "uniform":
        return stats.uniform.ppf(u, loc=-np.sqrt(3), scale=2 * np.sqrt(3))
    if name == "lognormal":
        s = 0.5
        mean = np.exp(s**2 / 2)
        var = (np.exp(s**2) - 1) * np.exp(s**2)
        return (stats.lognorm.ppf(u, s) - mean) / np.sqrt(var)
    if name == "skew_normal":
        a = params.skew
        delta = a / np.sqrt(1 + a**2)
        mean = delta * np.sqrt(2 / np.pi)
        var = 1 - 2 * delta**2 / np.pi
        return (stats.skewnorm.ppf(u, a) - mean) / np.sqrt(var)
    if name == "ged":
        beta = params.ged_beta
        var = stats.gennorm.var(beta)
        return stats.gennorm.ppf(u, beta) / np.sqrt(var)
    raise ValueError(f"Unknown distribution '{name}'")


def _standardized_draw(
    name: str, rng: np.random.Generator, shape: tuple[int, ...], params
) -> np.ndarray:
    """Direct sampling (fast path for pseudo-random engines)."""
    if name == "normal":
        return rng.standard_normal(shape)
    if name == "student_t":
        df = params.student_t_df
        return rng.standard_t(df, shape) * np.sqrt((df - 2.0) / df)
    if name == "laplace":
        return rng.laplace(0.0, 1.0, shape) / np.sqrt(2.0)
    if name == "uniform":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), shape)
    # Heavier distributions: transform uniforms.
    u = np.clip(rng.random(shape), 1e-12, 1 - 1e-12)
    return _standardized_ppf(name, u, params)


# --------------------------------------------------------------------------- #
# Façade
# --------------------------------------------------------------------------- #
class InnovationGenerator:
    """Produces standardized innovation matrices for path simulation.

    Parameters
    ----------
    engine:
        One of ``pcg64``, ``mt19937``, ``sobol``, ``halton``, ``lhs``.
    distribution:
        A :class:`config.DistributionParams` instance.
    antithetic:
        If True, the second half of each block is the negation of the first,
        which cancels odd-moment sampling error (variance reduction).
    seed:
        Base seed. Chunked calls should pass distinct ``stream`` offsets.
    """

    def __init__(self, engine: str, distribution, antithetic: bool, seed: int | None):
        self.engine = engine
        self.distribution = distribution
        self.antithetic = antithetic
        self.seed = seed
        self.is_quasi = engine in ("sobol", "halton", "lhs")

    @classmethod
    def from_config(cls, cfg) -> "InnovationGenerator":
        """Build a generator straight from a SimulationConfig."""
        return cls(
            engine=cfg.rng_engine,
            distribution=cfg.distribution,
            antithetic=(cfg.variance_reduction == "antithetic"),
            seed=cfg.seed,
        )

    def _stream_seed(self, stream: int) -> int | None:
        if self.seed is None:
            return None
        return int(np.random.SeedSequence([self.seed, stream]).generate_state(1)[0])

    def normals(self, n_paths: int, n_steps: int, stream: int = 0) -> np.ndarray:
        """Strictly Gaussian innovations (needed by Heston / VG subordination)."""
        return self._generate(n_paths, n_steps, stream, force_normal=True)

    def innovations(self, n_paths: int, n_steps: int, stream: int = 0) -> np.ndarray:
        """Innovations from the configured distribution."""
        return self._generate(n_paths, n_steps, stream, force_normal=False)

    def uniforms(self, n_paths: int, n_steps: int, stream: int = 0) -> np.ndarray:
        """Uniforms in (0,1) — used for Poisson thinning and bootstrap indices."""
        seed = self._stream_seed(stream)
        if self.is_quasi:
            return _quasi_uniforms(self.engine, n_paths, n_steps, seed).reshape(
                n_paths, n_steps
            )
        rng = _pseudo_rng(self.engine, seed)
        return np.clip(rng.random((n_paths, n_steps)), 1e-12, 1 - 1e-12)

    def rng(self, stream: int = 0) -> np.random.Generator:
        """A plain NumPy Generator (for Poisson/gamma draws etc.)."""
        base = "mt19937" if self.engine == "mt19937" else "pcg64"
        return _pseudo_rng(base, self._stream_seed(stream))

    # ------------------------------------------------------------------ #
    def _generate(
        self, n_paths: int, n_steps: int, stream: int, force_normal: bool
    ) -> np.ndarray:
        name = "normal" if force_normal else self.distribution.name
        seed = self._stream_seed(stream)

        n_base = (n_paths + 1) // 2 if self.antithetic else n_paths

        if self.is_quasi:
            u = _quasi_uniforms(self.engine, n_base, n_steps, seed)
            z = _standardized_ppf(name, u, self.distribution).reshape(n_base, n_steps)
        else:
            rng = _pseudo_rng(self.engine, seed)
            z = _standardized_draw(name, rng, (n_base, n_steps), self.distribution)

        if self.antithetic:
            z = np.concatenate([z, -z], axis=0)[:n_paths]
        return z


# --------------------------------------------------------------------------- #
# Educational tooltips (surfaced in the UI)
# --------------------------------------------------------------------------- #
RNG_TOOLTIPS = {
    "pcg64": "PCG64: NumPy's default pseudo-random generator. Excellent statistical "
             "quality, long period (2^128), fast, and supports independent streams.",
    "mt19937": "Mersenne Twister: the classic generator (period 2^19937-1). Included "
               "for legacy comparability; PCG64 is statistically superior.",
    "sobol": "Sobol sequence: a quasi-random (low-discrepancy) sequence that fills "
             "space more evenly than random points, often achieving O(1/N) "
             "convergence versus O(1/sqrt(N)) for pseudo-random Monte Carlo.",
    "halton": "Halton sequence: a low-discrepancy sequence built from radical-inverse "
              "functions of coprime bases. Best in low-to-moderate dimensions.",
    "lhs": "Latin Hypercube Sampling: stratifies each dimension into N equal bins and "
           "samples exactly once per bin, reducing variance of estimated means.",
}

VR_TOOLTIPS = {
    "none": "Plain Monte Carlo: independent draws, error shrinks as 1/sqrt(N).",
    "antithetic": "Antithetic variates: pair each draw Z with -Z. Errors from the two "
                  "halves partially cancel, reducing variance at zero extra cost for "
                  "monotone payoffs.",
    "control_variate": "Control variates: use a quantity with a known expectation "
                       "(here, the terminal price under GBM, E[S_T]=S_0*e^{mu*T}) to "
                       "correct the estimator: X_cv = X - b*(C - E[C]).",
    "importance": "Importance sampling: simulate under a tilted drift so rare regions "
                  "are sampled more often, then reweight by the likelihood ratio. "
                  "Useful for tail probabilities (e.g., deep losses).",
}

DIST_TOOLTIPS = {
    "normal": "Gaussian innovations — the Black-Scholes assumption. Thin tails.",
    "student_t": "Student t: fat tails controlled by degrees of freedom. Lower df → "
                 "more extreme daily moves, closer to real equity returns.",
    "laplace": "Laplace: sharper peak and heavier tails than normal; historically a "
               "good fit for daily FX and equity returns.",
    "lognormal": "Centered lognormal innovations: positively skewed shocks.",
    "uniform": "Uniform: bounded, thin-tailed shocks. Mostly pedagogical — shows how "
               "distributional shape flows into terminal prices.",
    "skew_normal": "Skew normal: adds asymmetry via a shape parameter alpha. Negative "
                   "alpha produces a heavier left (crash) tail.",
    "ged": "Generalized Error Distribution: shape beta interpolates tails — beta=2 is "
           "normal, beta=1 is Laplace, beta<2 means fat tails. Common in GARCH work.",
}
