"""Variance Gamma model (Madan, Carr & Chang 1998) — exact simulation.

The VG process is Brownian motion with drift theta and volatility sigma,
evaluated at a random gamma "business time" G_t:

    X_t = theta * G_t + sigma * W(G_t),
    G_t ~ Gamma(shape = t / nu, scale = nu)   (so E[G_t] = t, Var[G_t] = nu t)

This is NOT approximated by a compound-Poisson or small-jump truncation: each
step's gamma increment is drawn exactly from the gamma distribution, and the
Brownian part conditional on G is exactly Gaussian, so the scheme samples the
true VG transition law.

The price is
    S_t = S_0 * exp( (mu + omega) t + X_t ),
    omega = (1 / nu) * log(1 - theta*nu - sigma^2 * nu / 2)

where omega is the martingale correction making E[S_t] = S_0 e^{mu t}.
VG produces fat tails and skew through nu (kurtosis) and theta (skew).
"""

from __future__ import annotations

import numpy as np

from simulator.models.base import StochasticModel
from simulator.random_generators import InnovationGenerator


class VarianceGamma(StochasticModel):
    name = "variance_gamma"
    latex = (
        r"X_t = \theta G_t + \sigma W(G_t),\quad "
        r"G_t \sim \Gamma(t/\nu,\ \nu)"
    )

    def simulate(
        self, n_paths: int, gen: InnovationGenerator, stream: int = 0
    ) -> np.ndarray:
        cfg = self.cfg
        n_steps, dt = self._grid()
        mu = cfg.effective_drift()
        v = cfg.vg
        sigma = v.sigma * cfg.scenario.vol_multiplier
        nu, theta = v.nu, v.theta

        omega = np.log(1.0 - theta * nu - 0.5 * sigma**2 * nu) / nu

        rng = gen.rng(stream + 1)
        # Exact gamma subordinator increments: shape dt/nu, scale nu.
        g = rng.gamma(shape=dt / nu, scale=nu, size=(n_paths, n_steps))
        z = gen.normals(n_paths, n_steps, stream)

        increments = (mu + omega) * dt + theta * g + sigma * np.sqrt(g) * z

        sc = cfg.scenario
        if sc.crash_day is not None and 0 <= sc.crash_day < n_steps:
            increments[:, sc.crash_day] += sc.crash_size

        return self._prepend_s0(increments)
