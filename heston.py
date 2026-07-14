"""Heston (1993) stochastic volatility model.

SDEs:
    dS_t = mu * S_t dt + sqrt(v_t) * S_t dW_t^S
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^v
    corr(dW^S, dW^v) = rho

Discretization: full truncation Euler (Lord, Koekkoek & van Dijk 2010), the
standard robust scheme when the Feller condition 2*kappa*theta >= xi^2 is not
guaranteed:

    v_{t+dt} = v_t + kappa (theta - v_t^+) dt + xi sqrt(v_t^+ dt) Z_v
    log S_{t+dt} = log S_t + (mu - v_t^+/2) dt + sqrt(v_t^+ dt) Z_s

with v^+ = max(v, 0) and correlated Gaussians
    Z_s = rho * Z_v + sqrt(1 - rho^2) * Z_perp.

The variance recursion is sequential in time (an inherent property of the
model) but fully vectorized across paths, so the loop runs n_steps times
regardless of path count.
"""

from __future__ import annotations

import numpy as np

from simulator.models.base import StochasticModel
from simulator.random_generators import InnovationGenerator


class HestonModel(StochasticModel):
    name = "heston"
    latex = (
        r"dS_t = \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S,\quad "
        r"dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v"
    )

    def simulate(
        self, n_paths: int, gen: InnovationGenerator, stream: int = 0
    ) -> np.ndarray:
        cfg = self.cfg
        n_steps, dt = self._grid()
        mu = cfg.effective_drift()
        h = cfg.heston

        vol_mult2 = cfg.scenario.vol_multiplier**2
        v0 = h.v0 * vol_mult2
        theta = h.theta * vol_mult2

        z_v = gen.normals(n_paths, n_steps, stream)
        z_perp = gen.rng(stream + 1).standard_normal((n_paths, n_steps))
        z_s = h.rho * z_v + np.sqrt(1.0 - h.rho**2) * z_perp

        sqrt_dt = np.sqrt(dt)
        v = np.full(n_paths, v0)
        increments = np.empty((n_paths, n_steps))

        for t in range(n_steps):
            v_pos = np.maximum(v, 0.0)
            sqrt_v_dt = np.sqrt(v_pos) * sqrt_dt
            increments[:, t] = (mu - 0.5 * v_pos) * dt + sqrt_v_dt * z_s[:, t]
            v = v + h.kappa * (theta - v_pos) * dt + h.xi * sqrt_v_dt * z_v[:, t]

        sc = cfg.scenario
        if sc.crash_day is not None and 0 <= sc.crash_day < n_steps:
            increments[:, sc.crash_day] += sc.crash_size

        return self._prepend_s0(increments)

    def simulate_with_variance(
        self, n_paths: int, gen: InnovationGenerator, stream: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """As :meth:`simulate` but also returns the variance paths
        (n_paths, n_steps + 1) for volatility diagnostics."""
        cfg = self.cfg
        n_steps, dt = self._grid()
        mu = cfg.effective_drift()
        h = cfg.heston
        vol_mult2 = cfg.scenario.vol_multiplier**2
        v0 = h.v0 * vol_mult2
        theta = h.theta * vol_mult2

        z_v = gen.normals(n_paths, n_steps, stream)
        z_perp = gen.rng(stream + 1).standard_normal((n_paths, n_steps))
        z_s = h.rho * z_v + np.sqrt(1.0 - h.rho**2) * z_perp

        sqrt_dt = np.sqrt(dt)
        v = np.full(n_paths, v0)
        variances = np.empty((n_paths, n_steps + 1))
        variances[:, 0] = v0
        increments = np.empty((n_paths, n_steps))

        for t in range(n_steps):
            v_pos = np.maximum(v, 0.0)
            sqrt_v_dt = np.sqrt(v_pos) * sqrt_dt
            increments[:, t] = (mu - 0.5 * v_pos) * dt + sqrt_v_dt * z_s[:, t]
            v = v + h.kappa * (theta - v_pos) * dt + h.xi * sqrt_v_dt * z_v[:, t]
            variances[:, t + 1] = np.maximum(v, 0.0)

        return self._prepend_s0(increments), variances
