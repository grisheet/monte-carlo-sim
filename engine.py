"""Monte Carlo engine: chunked, cancellable, instrumented.

Responsibilities
----------------
* Validate the configuration before touching memory.
* Split large runs into chunks (memory control) with distinct RNG streams.
* Report progress (fraction complete, paths/sec, ETA, memory footprint)
  through an optional callback; support cooperative cancellation.
* Apply estimator-level variance reduction:
    - control variates (terminal price against its known GBM expectation)
    - importance sampling (drift tilt with likelihood-ratio weights)
* Attach per-path weights so downstream statistics remain unbiased.

The engine keeps at most ``keep_paths`` full paths for plotting (a uniform
subsample) while accumulating *all* terminal prices, maxima, minima and
drawdowns, so statistics use every simulated path even at 100k+ paths.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from config import SimulationConfig
from simulator.models import create_model
from simulator.models.historical_bootstrap import HistoricalBootstrap
from simulator.random_generators import InnovationGenerator
from simulator.validation import validate_config

logger = logging.getLogger("mcsim.engine")

ProgressCallback = Callable[[float, str], bool]
"""Called with (fraction_done, message). Return False to cancel."""


@dataclass
class SimulationResult:
    """Container for everything downstream analysis needs."""

    config: SimulationConfig
    time_grid: np.ndarray                 # (n_steps + 1,) in years
    sample_paths: np.ndarray              # (n_kept, n_steps + 1) for plotting
    terminal_prices: np.ndarray           # (n_total,)
    weights: np.ndarray                   # (n_total,) importance weights (sum ~ n)
    path_max: np.ndarray                  # (n_total,) running maxima
    path_min: np.ndarray                  # (n_total,)
    max_drawdown: np.ndarray              # (n_total,) per-path max drawdown
    quantile_bands: np.ndarray            # (len(QUANTILES), n_steps + 1) fan chart
    mean_path: np.ndarray                 # (n_steps + 1,)
    elapsed_seconds: float = 0.0
    paths_per_second: float = 0.0
    peak_chunk_mb: float = 0.0
    variance_paths: np.ndarray | None = None  # Heston diagnostic (subsample)
    metadata: dict = field(default_factory=dict)

    QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


class MonteCarloEngine:
    """Runs a configured simulation. Stateless between runs."""

    def __init__(self, keep_paths: int = 400):
        self.keep_paths = keep_paths

    def run(
        self,
        cfg: SimulationConfig,
        historical_returns: np.ndarray | None = None,
        progress: ProgressCallback | None = None,
    ) -> SimulationResult:
        validate_config(cfg)
        logger.info("Simulation start | model=%s n=%d steps=%d seed=%s",
                    cfg.model, cfg.n_simulations, cfg.n_steps, cfg.seed)
        t0 = time.perf_counter()

        model = create_model(cfg)
        if isinstance(model, HistoricalBootstrap) and historical_returns is not None:
            model.set_returns(historical_returns)

        antithetic = cfg.variance_reduction == "antithetic"
        importance = cfg.variance_reduction == "importance"

        # Importance sampling: simulate under a tilted drift, reweight later.
        tilt = 0.0
        run_cfg = cfg
        if importance:
            # Tilt drift downward one sigma to enrich the loss tail.
            tilt = -cfg.effective_sigma()
            run_cfg = SimulationConfig.from_dict(cfg.to_dict())
            run_cfg.scenario.drift_shift += tilt
            model = create_model(run_cfg)
            if isinstance(model, HistoricalBootstrap) and historical_returns is not None:
                model.set_returns(historical_returns)

        gen = InnovationGenerator(
            engine=cfg.rng_engine,
            distribution=cfg.distribution,
            antithetic=antithetic,
            seed=cfg.seed,
        )

        n_total = cfg.n_simulations
        n_steps = cfg.n_steps
        chunk = max(1, min(cfg.chunk_size, n_total))
        # Quasi-random engines lose their structure if split; run in one block
        # (they are used with moderate path counts by design).
        if gen.is_quasi:
            chunk = n_total

        time_grid = np.linspace(0.0, cfg.horizon_years, n_steps + 1)

        terminal, wmax, wmin, mdd, wts = [], [], [], [], []
        sum_paths = np.zeros(n_steps + 1)
        # Streaming quantiles are approximated by keeping a stratified sample
        # of full paths (up to sample_cap) — exact for runs <= sample_cap.
        sample_cap = max(self.keep_paths, 4_000)
        kept_paths: list[np.ndarray] = []
        kept_so_far = 0
        peak_mb = 0.0
        variance_paths = None

        done = 0
        stream = 0
        while done < n_total:
            n = min(chunk, n_total - done)
            if cfg.model == "heston" and variance_paths is None:
                paths, var_paths = model.simulate_with_variance(n, gen, stream)
                variance_paths = var_paths[: min(n, 200)].copy()
            else:
                paths = model.simulate(n, gen, stream)
            peak_mb = max(peak_mb, paths.nbytes / 1e6)

            terminal.append(paths[:, -1].copy())
            running_max = np.maximum.accumulate(paths, axis=1)
            wmax.append(paths.max(axis=1))
            wmin.append(paths.min(axis=1))
            dd = 1.0 - paths / running_max
            mdd.append(dd.max(axis=1))
            sum_paths += paths.sum(axis=0)

            if importance:
                wts.append(self._is_weights(cfg, tilt, paths))
            else:
                wts.append(np.ones(n))

            room = sample_cap - kept_so_far
            if room > 0:
                take = min(room, n)
                sel = np.linspace(0, n - 1, take).astype(int)
                kept_paths.append(paths[sel].copy())
                kept_so_far += take

            del paths
            done += n
            stream += 100
            if progress is not None:
                rate = done / max(time.perf_counter() - t0, 1e-9)
                eta = (n_total - done) / max(rate, 1e-9)
                keep_going = progress(
                    done / n_total,
                    f"{done:,}/{n_total:,} paths | {rate:,.0f} paths/s | "
                    f"ETA {eta:,.1f}s | chunk {peak_mb:,.0f} MB",
                )
                if keep_going is False:
                    logger.warning("Simulation cancelled at %d/%d paths", done, n_total)
                    break

        terminal_prices = np.concatenate(terminal)
        weights = np.concatenate(wts)
        n_done = terminal_prices.size
        sample = np.vstack(kept_paths) if kept_paths else np.empty((0, n_steps + 1))

        result = SimulationResult(
            config=cfg,
            time_grid=time_grid,
            sample_paths=sample[: max(self.keep_paths, 1)],
            terminal_prices=terminal_prices,
            weights=weights,
            path_max=np.concatenate(wmax),
            path_min=np.concatenate(wmin),
            max_drawdown=np.concatenate(mdd),
            quantile_bands=np.quantile(sample, SimulationResult.QUANTILES, axis=0)
            if sample.size
            else np.zeros((len(SimulationResult.QUANTILES), n_steps + 1)),
            mean_path=sum_paths / max(n_done, 1),
            variance_paths=variance_paths,
        )

        if cfg.variance_reduction == "control_variate":
            result.metadata["control_variate"] = self._control_variate_mean(cfg, result)

        result.metadata.update(
            model=cfg.model,
            rng_engine=cfg.rng_engine,
            distribution=cfg.distribution.name,
            variance_reduction=cfg.variance_reduction,
            scenario=cfg.scenario.name,
            n_requested=n_total,
            n_completed=n_done,
            seed=cfg.seed,
        )

        result.elapsed_seconds = time.perf_counter() - t0
        result.paths_per_second = n_done / max(result.elapsed_seconds, 1e-9)
        result.peak_chunk_mb = peak_mb
        logger.info(
            "Simulation end | %d paths in %.2fs (%.0f paths/s, peak chunk %.0f MB)",
            n_done, result.elapsed_seconds, result.paths_per_second, peak_mb,
        )
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_weights(cfg: SimulationConfig, tilt: float, paths: np.ndarray) -> np.ndarray:
        """Likelihood ratio dP/dQ for a drift tilt of a lognormal path.

        For GBM, tilting drift by ``tilt`` changes the terminal log-price mean
        by tilt*T; the Radon-Nikodym derivative is
            exp( -(tilt/sigma^2) * (logS_T - logS_0 - (mu_Q - sigma^2/2)T)
                 - (tilt^2 / (2 sigma^2)) * T ) ... simplified via Girsanov:
            L = exp(-a * W_T - a^2 T / 2),  a = tilt / sigma.
        We recover sigma*W_T from the realized log return under Q.
        """
        sigma = max(cfg.effective_sigma(), 1e-12)
        T = cfg.horizon_years
        mu_q = cfg.effective_drift() + tilt
        log_ret = np.log(paths[:, -1] / cfg.initial_price)
        w_T = (log_ret - (mu_q - 0.5 * sigma**2) * T) / sigma
        a = tilt / sigma
        w = np.exp(-a * w_T - 0.5 * a**2 * T)
        # Normalize so weights average to 1 (self-normalized IS).
        return w * (w.size / w.sum())

    @staticmethod
    def _control_variate_mean(cfg: SimulationConfig, res: SimulationResult) -> dict:
        """Control-variate estimate of E[S_T] using the known GBM expectation."""
        c = res.terminal_prices                     # control = terminal price
        known = cfg.initial_price * np.exp(cfg.effective_drift() * cfg.horizon_years)
        x = res.terminal_prices                     # target (same here; the CV
        # machinery matters when target != control, e.g. payoffs — we expose
        # the corrected mean and the variance reduction achieved).
        cov = np.cov(x, c, ddof=1)
        b = cov[0, 1] / max(cov[1, 1], 1e-18)
        corrected = x - b * (c - known)
        return {
            "beta": float(b),
            "raw_mean": float(x.mean()),
            "cv_mean": float(corrected.mean()),
            "raw_se": float(x.std(ddof=1) / np.sqrt(x.size)),
            "cv_se": float(corrected.std(ddof=1) / np.sqrt(x.size)),
            "known_expectation": float(known),
        }
