"""Tests for the RNG layer, variance reduction, and the simulation engine."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from config import DISTRIBUTIONS, RNG_ENGINES, SimulationConfig
from simulator.engine import MonteCarloEngine
from simulator.random_generators import InnovationGenerator


def _cfg(**kwargs) -> SimulationConfig:
    base = dict(initial_price=100.0, mu=0.07, sigma=0.20, horizon_years=1.0,
                trading_days=252, n_simulations=20_000, seed=42)
    base.update(kwargs)
    return SimulationConfig(**base)


# --------------------------------------------------------------------------- #
# Random number generation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("engine", list(RNG_ENGINES))
def test_every_engine_produces_standard_normals(engine):
    gen = InnovationGenerator.from_config(_cfg(rng_engine=engine))
    z = gen.normals(8192, 16, stream=0)
    assert z.shape == (8192, 16) and np.all(np.isfinite(z))
    assert abs(z.mean()) < 0.05
    assert abs(z.std() - 1.0) < 0.05


@pytest.mark.parametrize("dist", list(DISTRIBUTIONS))
def test_every_distribution_is_standardized(dist):
    """Innovations must be zero-mean and unit-variance, whatever the shape."""
    cfg = _cfg()
    cfg.distribution.name = dist
    x = InnovationGenerator.from_config(cfg).innovations(40_000, 10, stream=0)
    assert abs(x.mean()) < 0.03, f"{dist}: mean={x.mean():.4f}"
    assert abs(x.std() - 1.0) < 0.05, f"{dist}: std={x.std():.4f}"


def test_student_t_and_ged_are_fat_tailed_relative_to_normal():
    def kurt(name: str, **over) -> float:
        cfg = _cfg()
        cfg.distribution.name = name
        for k, v in over.items():
            setattr(cfg.distribution, k, v)
        x = InnovationGenerator.from_config(cfg).innovations(60_000, 4, stream=0)
        return stats.kurtosis(x.ravel())
    assert kurt("student_t", student_t_df=4.0) > kurt("normal") + 0.5
    assert kurt("ged", ged_beta=1.0) > kurt("normal") + 0.3
    assert kurt("uniform") < kurt("normal")  # platykurtic, as expected


def test_the_seed_makes_output_reproducible():
    a = InnovationGenerator.from_config(_cfg(seed=99)).normals(1000, 10)
    b = InnovationGenerator.from_config(_cfg(seed=99)).normals(1000, 10)
    c = InnovationGenerator.from_config(_cfg(seed=100)).normals(1000, 10)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_independent_streams_are_uncorrelated():
    gen = InnovationGenerator.from_config(_cfg())
    a = gen.normals(20_000, 1, stream=0).ravel()
    b = gen.normals(20_000, 1, stream=1).ravel()
    assert not np.array_equal(a, b)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.03


def test_antithetic_innovations_are_exact_mirrors():
    z = InnovationGenerator.from_config(
        _cfg(variance_reduction="antithetic")).normals(1000, 12)
    half = 500
    assert np.allclose(z[:half], -z[half:])
    assert abs(z.mean()) < 1e-12  # balanced by construction


def test_antithetic_variates_reduce_the_standard_error():
    def se(vr: str) -> float:
        cfg = _cfg(n_simulations=4000, trading_days=64, variance_reduction=vr)
        res = MonteCarloEngine().run(cfg)
        return res.terminal_prices.std(ddof=1) / math.sqrt(cfg.n_simulations)
    assert se("antithetic") < se("none")


def test_quasi_random_points_fill_the_unit_interval_evenly():
    u = InnovationGenerator.from_config(
        _cfg(rng_engine="sobol")).uniforms(4096, 2, stream=0)
    assert u.min() >= 0.0 and u.max() <= 1.0
    ks_sobol = stats.kstest(u[:, 0], "uniform").statistic
    up = InnovationGenerator.from_config(
        _cfg(rng_engine="pcg64")).uniforms(4096, 2, stream=0)
    ks_pcg = stats.kstest(up[:, 0], "uniform").statistic
    assert ks_sobol < ks_pcg  # lower discrepancy is the whole point


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def test_engine_result_shapes_and_metadata():
    cfg = _cfg(n_simulations=5000, trading_days=100)
    res = MonteCarloEngine().run(cfg)
    assert res.terminal_prices.shape == (5000,)
    assert res.time_grid.shape == (cfg.n_steps + 1,)
    assert res.sample_paths.shape[1] == cfg.n_steps + 1
    assert res.sample_paths.shape[0] <= 400  # capped for plotting
    assert res.mean_path.shape == (cfg.n_steps + 1,)
    assert np.all((res.max_drawdown >= 0) & (res.max_drawdown <= 1))
    assert res.elapsed_seconds > 0 and res.paths_per_second > 0
    assert res.metadata and res.config.model == cfg.model


def test_chunk_size_does_not_change_the_estimate():
    """Chunking is a memory knob: estimates must agree within MC error."""
    big = MonteCarloEngine().run(_cfg(n_simulations=40_000, chunk_size=100_000))
    small = MonteCarloEngine().run(_cfg(n_simulations=40_000, chunk_size=5000))
    se = big.terminal_prices.std(ddof=1) / math.sqrt(40_000)
    assert abs(big.terminal_prices.mean() - small.terminal_prices.mean()) < 5 * se


def test_path_extremes_bracket_the_terminal_price():
    res = MonteCarloEngine().run(_cfg(n_simulations=3000, trading_days=60))
    assert np.all(res.path_min <= res.terminal_prices + 1e-9)
    assert np.all(res.path_max >= res.terminal_prices - 1e-9)
    assert np.all(res.path_max >= res.path_min)


def test_quantile_bands_never_cross():
    from simulator.engine import SimulationResult
    res = MonteCarloEngine().run(_cfg(n_simulations=8000, trading_days=50))
    bands = res.quantile_bands  # rows follow SimulationResult.QUANTILES
    assert bands.shape[0] == len(SimulationResult.QUANTILES)
    for lo, hi in zip(bands, bands[1:]):
        assert np.all(lo <= hi + 1e-9)


def test_progress_callback_receives_updates_and_can_cancel():
    seen: list[float] = []

    def progress(frac: float, message: str) -> bool:
        seen.append(frac)
        assert "paths" in message  # human-readable status line
        return False  # cancel at the first chunk boundary

    res = MonteCarloEngine().run(
        _cfg(n_simulations=60_000, chunk_size=5000), progress=progress)
    assert seen and 0.0 < seen[0] <= 1.0
    assert res is None or res.terminal_prices.size < 60_000


def test_risk_neutral_mode_swaps_the_drift_for_the_risk_free_rate():
    cfg = _cfg(n_simulations=60_000, mu=0.20, risk_free_rate=0.03,
               use_risk_neutral=True)
    res = MonteCarloEngine().run(cfg)
    theo = 100.0 * math.exp(0.03)
    se = res.terminal_prices.std(ddof=1) / math.sqrt(cfg.n_simulations)
    assert abs(res.terminal_prices.mean() - theo) < 4 * se


def test_importance_sampling_weights_stay_unbiased():
    """The Girsanov likelihood ratio must recover the same expectation."""
    cfg = _cfg(n_simulations=40_000, trading_days=100,
               variance_reduction="importance")
    res = MonteCarloEngine().run(cfg)
    w = res.weights
    assert w is not None and np.all(w > 0)
    est = np.average(res.terminal_prices, weights=w)
    theo = 100.0 * math.exp(0.07)
    assert abs(est - theo) < 0.03 * theo


def test_control_variate_estimator_lands_near_theory():
    res = MonteCarloEngine().run(
        _cfg(n_simulations=20_000, trading_days=100,
             variance_reduction="control_variate"))
    assert abs(res.terminal_prices.mean() - 100.0 * math.exp(0.07)) < 1.0


def test_crash_scenario_shifts_the_distribution_downward():
    base = MonteCarloEngine().run(_cfg(n_simulations=20_000))
    cfg = _cfg(n_simulations=20_000)
    cfg.scenario.name = "market_crash"
    cfg.scenario.vol_multiplier = 2.5
    cfg.scenario.drift_shift = -0.30
    crash = MonteCarloEngine().run(cfg)
    assert crash.terminal_prices.mean() < base.terminal_prices.mean()
    assert crash.terminal_prices.std() > base.terminal_prices.std()
    assert crash.max_drawdown.mean() > base.max_drawdown.mean()
