"""Abstract base class for stochastic price models.

Adding a new model (GARCH, SABR, regime switching, ...) only requires
subclassing :class:`StochasticModel` and registering it in
``simulator/models/__init__.py`` — no other code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from config import SimulationConfig
from simulator.random_generators import InnovationGenerator


class StochasticModel(ABC):
    """Interface all price models implement.

    A model turns a configuration plus a stream of innovations into a matrix
    of simulated price paths of shape ``(n_paths, n_steps + 1)``, where column
    0 is the (deterministic) initial price.
    """

    name: str = "abstract"
    latex: str = ""

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg

    @abstractmethod
    def simulate(
        self, n_paths: int, gen: InnovationGenerator, stream: int = 0
    ) -> np.ndarray:
        """Simulate ``n_paths`` price paths. Must be fully vectorized."""

    # Convenience shared by subclasses -------------------------------- #
    def _grid(self) -> tuple[int, float]:
        return self.cfg.n_steps, self.cfg.dt

    def _prepend_s0(self, log_increments: np.ndarray) -> np.ndarray:
        """Cumulate log increments and prepend S0. Shape in: (n, steps)."""
        log_paths = np.cumsum(log_increments, axis=1)
        s0 = self.cfg.initial_price
        paths = np.empty((log_increments.shape[0], log_increments.shape[1] + 1))
        paths[:, 0] = s0
        np.exp(log_paths, out=log_paths)
        paths[:, 1:] = s0 * log_paths
        return paths
