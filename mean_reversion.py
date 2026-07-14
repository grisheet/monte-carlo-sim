"""Ornstein-Uhlenbeck mean-reverting model (exponential OU on price).

We model the log price X_t = log S_t as an OU process reverting to
x_bar = log(S_bar) (with a lognormal convexity correction so that the
long-run *price* level matches the configured mean):

    dX_t = kappa (x_bar - X_t) dt + sigma dW_t

The transition density is Gaussian and known in closed form, so we use the
EXACT discretization (no Euler bias):

    X_{t+dt} = x_bar + (X_t - x_bar) e^{-kappa dt}
               + sigma * sqrt( (1 - e^{-2 kappa dt}) / (2 kappa) ) * Z

As kappa -> 0 the process degenerates to arithmetic Brownian motion on the
log price; the code handles that limit explicitly.
"""

from __future__ import annotations

import numpy as np

from simulator.models.base import StochasticModel
from simulator.random_generators import InnovationGenerator


class OrnsteinUhlenbeck(StochasticModel):
    name = "mean_reversion"
    latex = r"dX_t = \kappa(\bar{x} - X_t)\,dt + \sigma\,dW_t,\quad X_t=\log S_t"

    def simulate(
        self, n_paths: int, gen: InnovationGenerator, stream: int = 0
    ) -> np.ndarray:
        cfg = self.cfg
        n_steps, dt = self._grid()
        o = cfg.ou
        kappa = o.speed
        sigma = o.volatility * cfg.scenario.vol_multiplier

        # Stationary variance of X is sigma^2/(2 kappa); subtract half of it
        # so the long-run E[S] equals the configured price mean.
        if kappa > 1e-12:
            stat_var = sigma**2 / (2.0 * kappa)
        else:
            stat_var = 0.0
        x_bar = np.log(o.mean) - 0.5 * stat_var

        if kappa > 1e-12:
            phi = np.exp(-kappa * dt)
            step_std = sigma * np.sqrt((1.0 - phi**2) / (2.0 * kappa))
        else:
            phi = 1.0
            step_std = sigma * np.sqrt(dt)

        z = gen.innovations(n_paths, n_steps, stream)

        x = np.full(n_paths, np.log(cfg.initial_price))
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = cfg.initial_price
        for t in range(n_steps):
            x = x_bar + (x - x_bar) * phi + step_std * z[:, t]
            paths[:, t + 1] = np.exp(x)

        sc = cfg.scenario
        if sc.crash_day is not None and 0 <= sc.crash_day < n_steps:
            paths[:, sc.crash_day + 1:] *= np.exp(sc.crash_size)

        return paths
