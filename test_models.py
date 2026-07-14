"""Statistical correctness tests for each stochastic model.

Each model is checked against a property with a known closed form -- chiefly
the drift condition E[S_T] = S_0 * exp(mu * T). Tolerances are derived from the
Monte Carlo standard error rather than guessed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from config import MODELS, SimulationConfig
from simulator.models import MODEL_REGISTRY, create_model
from simulator.random_generators import InnovationGenerator

N_PATHS = 40_000
S0 = 100.0
MU = 0.08
SIGMA = 0.20
EXPECTED = S0 * math.exp(MU * 1.0)


def _cfg(**kwargs) -> SimulationConfig:
    base = dict(initial_price=S0, mu=MU, sigma=SIGMA, horizon_years=1.0,
                trading_days=252, n_simulations=N_PATHS, seed=7)
    base.update(kwargs)
    return SimulationConfig(**base)


def _run(cfg: SimulationConfig, returns: np.ndarray | None = None) -> np.ndarray:
    gen = InnovationGenerator.from_config(cfg)
    model = create_model(cfg)
    if returns is not None:
        model.set_returns(returns)
    paths = model.simulate(cfg.n_simulations, gen, stream=0)
    n_steps = int(round(cfg.horizon_years * cfg.trading_days))
    assert paths.shape == (cfg.n_simulations, n_steps + 1)
    assert np.all(np.isfinite(paths)) and np.all(paths > 0)
    assert np.allclose(paths[:, 0], cfg.initial_price)
    return paths


def _assert_drift(terminal: np.ndarray, expected: float, n_sigma: float = 4.0) -> None:
    se = terminal.std(ddof=1) / math.sqrt(terminal.size)
    assert abs(terminal.mean() - expected) < n_sigma * se, (
        f"mean {terminal.mean():.4f} vs expected {expected:.4f} (SE={se:.4f})")


def test_gbm_expected_terminal():
    _assert_drift(_run(_cfg(model="gbm"))[:, -1], EXPECTED)


def test_gbm_lognormal_variance():
    """Var[S_T] = S0^2 e^{2 mu T} (e^{sigma^2 T} - 1)."""
    terminal = _run(_cfg(model="gbm"))[:, -1]
    theo = S0**2 * math.exp(2 * MU) * (math.exp(SIGMA**2) - 1)
    assert abs(terminal.var(ddof=1) / theo - 1.0) < 0.06


def test_jump_diffusion_compensator_preserves_drift():
    """Merton's compensator must leave E[S_T] unchanged by the jump term."""
    cfg = _cfg(model="jump_diffusion")
    cfg.jumps.intensity, cfg.jumps.mean, cfg.jumps.volatility = 1.0, -0.05, 0.12
    _assert_drift(_run(cfg)[:, -1], EXPECTED, n_sigma=4.5)


def test_jump_diffusion_has_fatter_tails_than_gbm():
    cfg = _cfg(model="jump_diffusion")
    cfg.jumps.intensity, cfg.jumps.mean, cfg.jumps.volatility = 3.0, -0.10, 0.20
    jd = np.log(_run(cfg)[:, -1] / S0)
    gbm = np.log(_run(_cfg(model="gbm"))[:, -1] / S0)
    assert stats.kurtosis(jd) > stats.kurtosis(gbm) + 0.3


def test_heston_expected_terminal():
    cfg = _cfg(model="heston")
    cfg.heston.v0 = cfg.heston.theta = SIGMA**2
    cfg.heston.kappa, cfg.heston.xi, cfg.heston.rho = 2.0, 0.3, -0.7
    _assert_drift(_run(cfg)[:, -1], EXPECTED, n_sigma=5.0)


def test_heston_variance_stays_non_negative_under_feller_violation():
    cfg = _cfg(model="heston", n_simulations=2000)
    cfg.heston.kappa, cfg.heston.xi = 1.0, 0.9  # 2*kappa*theta < xi^2
    gen = InnovationGenerator.from_config(cfg)
    _, variance = create_model(cfg).simulate_with_variance(2000, gen, stream=0)
    assert np.all(variance >= 0.0)  # full-truncation guarantee


def test_mean_reversion_pulls_back_to_the_long_run_mean():
    cfg = _cfg(model="mean_reversion", initial_price=150.0, horizon_years=10.0,
               n_simulations=20_000)
    cfg.ou.speed, cfg.ou.mean, cfg.ou.volatility = 3.0, 100.0, 0.20
    assert abs(_run(cfg)[:, -1].mean() - 100.0) < 2.0


def test_mean_reversion_handles_the_zero_speed_limit():
    cfg = _cfg(model="mean_reversion", n_simulations=2000)
    cfg.ou.speed = 1e-12  # must not divide by zero
    assert np.all(np.isfinite(_run(cfg)))


def test_variance_gamma_expected_terminal():
    cfg = _cfg(model="variance_gamma")
    cfg.vg.sigma, cfg.vg.nu, cfg.vg.theta = 0.20, 0.20, -0.15
    _assert_drift(_run(cfg)[:, -1], EXPECTED, n_sigma=5.0)


def test_variance_gamma_is_leptokurtic():
    cfg = _cfg(model="variance_gamma")
    cfg.vg.nu = 0.50  # heavier tails as nu grows
    x = np.log(_run(cfg)[:, -1] / S0)
    assert stats.kurtosis(x) > 0.3


@pytest.mark.parametrize("method", ["iid", "block", "rolling", "without_replacement"])
def test_bootstrap_schemes_track_the_sample_drift(method):
    cfg = _cfg(model="historical_bootstrap", n_simulations=3000)
    cfg.bootstrap.method, cfg.bootstrap.block_size = method, 20
    returns = np.random.default_rng(1).normal(
        MU / 252, SIGMA / math.sqrt(252), 2000)
    paths = _run(cfg, returns=returns)
    n_steps = 252
    realized = np.log(paths[:, -1] / S0).mean()
    assert abs(realized - returns.mean() * n_steps) < 0.05


def test_without_replacement_reproduces_the_sample_sum_exactly():
    """A permutation of every historical return must land on the same terminal."""
    cfg = _cfg(model="historical_bootstrap", trading_days=250,
               n_simulations=500)
    cfg.bootstrap.method = "without_replacement"
    returns = np.random.default_rng(3).normal(0.0004, 0.01, 250)
    paths = _run(cfg, returns=returns)
    assert np.allclose(np.log(paths[:, -1] / S0), returns.sum(), atol=1e-8)


def test_every_model_is_registered_and_documented():
    assert set(MODELS) == set(MODEL_REGISTRY)
    for key in MODELS:
        model = create_model(_cfg(model=key, n_simulations=100))
        assert model.name and model.latex  # tooltip/report metadata present
