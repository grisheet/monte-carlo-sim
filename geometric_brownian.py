"""Geometric Brownian Motion — the Black-Scholes model.

SDE:
    dS_t = mu * S_t dt + sigma * S_t dW_t

Exact discretization (log-Euler, exact for GBM):
    S_{t+dt} = S_t * exp( (mu - sigma^2/2) dt + sigma * sqrt(dt) * Z )

with Z standardized innovations. When a non-Gaussian distribution is chosen,
Z is drawn from that distribution standardized to unit variance, giving a
"generalized GBM" whose log returns inherit the chosen shape.

Scenario support: a deterministic crash gap of size ``crash_size`` (in log
return) can be injected at ``crash_day``, modeling earnings gaps or flash
crashes.
"""

from __future__ import annotations

import numpy as np

from simulator.models.base import StochasticModel
from simulator.random_generators import InnovationGenerator


class GeometricBrownianMotion(StochasticModel):
    name = "gbm"
    latex = r"dS_t = \mu S_t\,dt + \sigma S_t\,dW_t"

    def simulate(
        self, n_paths: int, gen: InnovationGenerator, stream: int = 0
    ) -> np.ndarray:
        cfg = self.cfg
        n_steps, dt = self._grid()
        mu = cfg.effective_drift()
        sigma = cfg.effective_sigma()

        z = gen.innovations(n_paths, n_steps, stream)
        increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

        sc = cfg.scenario
        if sc.crash_day is not None and 0 <= sc.crash_day < n_steps:
            increments[:, sc.crash_day] += sc.crash_size

        return self._prepend_s0(increments)
