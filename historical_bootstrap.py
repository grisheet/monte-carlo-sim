"""Historical bootstrap simulation.

Instead of assuming a parametric distribution, future daily log returns are
resampled from an observed return history. Four schemes are supported:

* ``iid``                — sample returns independently with replacement.
* ``without_replacement``— draw a permutation of history (per path) and take
                           the first n_steps returns; preserves the empirical
                           distribution exactly but not autocorrelation.
* ``block``              — sample fixed-length contiguous blocks with
                           replacement (Künsch 1989), preserving short-range
                           dependence such as volatility clustering.
* ``rolling``            — circular bootstrap: each path picks a random start
                           and rolls forward through history, wrapping around.

Set the return history with :meth:`set_returns` (the engine does this from
the loaded data). If none is provided, a synthetic Gaussian history matching
the configured mu/sigma is generated so the model remains runnable.
"""

from __future__ import annotations

import numpy as np

from simulator.models.base import StochasticModel
from simulator.random_generators import InnovationGenerator


class HistoricalBootstrap(StochasticModel):
    name = "historical_bootstrap"
    latex = r"r_{t}^{sim} \sim \hat{F}_{empirical}(r_{hist})"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._returns: np.ndarray | None = None

    def set_returns(self, log_returns: np.ndarray) -> None:
        arr = np.asarray(log_returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 30:
            raise ValueError("Bootstrap requires at least 30 historical returns.")
        self._returns = arr

    def _history(self, gen: InnovationGenerator, stream: int) -> np.ndarray:
        if self._returns is not None:
            return self._returns
        # Fallback synthetic history consistent with configured parameters.
        cfg = self.cfg
        dt = 1.0 / cfg.trading_days
        rng = gen.rng(stream + 7)
        n = max(5 * cfg.trading_days, 1_000)
        return (cfg.effective_drift() - 0.5 * cfg.effective_sigma() ** 2) * dt + (
            cfg.effective_sigma() * np.sqrt(dt) * rng.standard_normal(n)
        )

    def simulate(
        self, n_paths: int, gen: InnovationGenerator, stream: int = 0
    ) -> np.ndarray:
        cfg = self.cfg
        n_steps, _ = self._grid()
        hist = self._history(gen, stream)
        m = hist.size
        rng = gen.rng(stream)
        method = cfg.bootstrap.method

        if method == "iid":
            idx = rng.integers(0, m, size=(n_paths, n_steps))
            increments = hist[idx]

        elif method == "without_replacement":
            if n_steps > m:
                raise ValueError(
                    f"Sampling without replacement needs history ({m}) >= "
                    f"steps ({n_steps})."
                )
            # Vectorized per-path permutation via random keys + argsort.
            keys = rng.random((n_paths, m))
            order = np.argsort(keys, axis=1)[:, :n_steps]
            increments = hist[order]

        elif method == "block":
            b = max(1, min(cfg.bootstrap.block_size, m))
            n_blocks = int(np.ceil(n_steps / b))
            starts = rng.integers(0, m - b + 1, size=(n_paths, n_blocks))
            offsets = np.arange(b)
            idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_paths, -1)
            increments = hist[idx[:, :n_steps]]

        elif method == "rolling":
            starts = rng.integers(0, m, size=(n_paths, 1))
            offsets = np.arange(n_steps)[None, :]
            idx = (starts + offsets) % m
            increments = hist[idx]

        else:  # pragma: no cover - guarded by validation
            raise ValueError(f"Unknown bootstrap method '{method}'")

        sc = cfg.scenario
        # Scenario adjustments still apply: drift shift and vol multiplier.
        if sc.vol_multiplier != 1.0 or sc.drift_shift != 0.0:
            dt = 1.0 / cfg.trading_days
            mean = increments.mean()
            increments = mean + (increments - mean) * sc.vol_multiplier
            increments = increments + sc.drift_shift * dt
        if sc.crash_day is not None and 0 <= sc.crash_day < n_steps:
            increments[:, sc.crash_day] += sc.crash_size

        return self._prepend_s0(increments)
