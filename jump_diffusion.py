"""Merton (1976) jump-diffusion model.

SDE:
    dS_t / S_t = (mu - lambda * k) dt + sigma dW_t + (J - 1) dN_t

where N_t is a Poisson process with intensity lambda, log J ~ N(mu_J, sigma_J^2)
and k = E[J - 1] = exp(mu_J + sigma_J^2 / 2) - 1 is the drift compensation
that keeps the expected return equal to mu.

Discretization over a step dt:
    log S_{t+dt} - log S_t =
        (mu - lambda*k - sigma^2/2) dt
        + sigma sqrt(dt) Z
        + sum of n_t jump sizes,  n_t ~ Poisson(lambda*dt),
        jump sizes ~ N(mu_J, sigma_J^2)

The sum of n_t iid normals is N(n_t * mu_J, n_t * sigma_J^2), so the jump term
is simulated exactly and fully vectorized as
    n * mu_J + sqrt(n) * sigma_J * Z_J.
"""

from __future__ import annotations

import numpy as np

from simulator.models.base import StochasticModel
from simulator.random_generators import InnovationGenerator


class MertonJumpDiffusion(StochasticModel):
    name = "jump_diffusion"
    latex = (
        r"\frac{dS_t}{S_t} = (\mu - \lambda k)\,dt + \sigma\,dW_t + (J-1)\,dN_t"
    )

    def simulate(
        self, n_paths: int, gen: InnovationGenerator, stream: int = 0
    ) -> np.ndarray:
        cfg = self.cfg
        n_steps, dt = self._grid()
        mu = cfg.effective_drift()
        sigma = cfg.effective_sigma()

        lam = max(0.0, cfg.jumps.intensity + cfg.scenario.jump_intensity_add)
        mu_j = cfg.jumps.mean
        sig_j = cfg.jumps.volatility

        # Compensator so that E[S_T] = S_0 * exp(mu * T)
        k = np.exp(mu_j + 0.5 * sig_j**2) - 1.0

        z = gen.innovations(n_paths, n_steps, stream)
        rng = gen.rng(stream + 1)
        n_jumps = rng.poisson(lam * dt, size=(n_paths, n_steps))
        z_jump = rng.standard_normal((n_paths, n_steps))

        diffusion = (mu - lam * k - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        jumps = n_jumps * mu_j + np.sqrt(n_jumps) * sig_j * z_jump

        increments = diffusion + jumps

        sc = cfg.scenario
        if sc.crash_day is not None and 0 <= sc.crash_day < n_steps:
            increments[:, sc.crash_day] += sc.crash_size

        return self._prepend_s0(increments)
