"""Tests for statistics, risk, pricing, portfolio, config I/O, and validation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sps

from config import SimulationConfig
from simulator.correlation import (cholesky_factor, corr_to_cov,
                                   correlated_normals, nearest_psd)
from simulator.engine import MonteCarloEngine
from simulator.portfolio import PortfolioSpec, efficient_frontier, simulate_portfolio
from simulator.pricing import black_scholes, mc_expectation, mc_option_price
from simulator.risk import risk_report, var_cvar
from simulator.statistics import (confidence_interval, convergence_table,
                                  summary_statistics, weighted_quantile)
from simulator.validation import (ValidationError, validate_config,
                                  validate_correlation_matrix,
                                  validate_price_series)


def _cfg(**kwargs) -> SimulationConfig:
    base = dict(initial_price=100.0, mu=0.08, sigma=0.20, horizon_years=1.0,
                trading_days=252, n_simulations=20_000, seed=11)
    base.update(kwargs)
    return SimulationConfig(**base)


def _run(**kwargs):
    return MonteCarloEngine().run(_cfg(**kwargs))


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_weighted_quantile_matches_numpy_under_uniform_weights():
    x = np.random.default_rng(0).normal(size=5000)
    w = np.ones_like(x)
    for q in (0.05, 0.5, 0.95):
        wq = float(np.asarray(weighted_quantile(x, q, w)).ravel()[0])
        assert abs(wq - np.quantile(x, q)) < 0.02


def test_summary_statistics_recover_lognormal_moments():
    res = _run(n_simulations=60_000)
    s = summary_statistics(res)
    assert abs(s["mean_terminal"] - 100 * math.exp(0.08)) < 0.5
    assert s["skewness"] > 0                       # lognormal is right-skewed
    assert 0.15 < s["annualized_volatility"] < 0.25
    assert s["ci95_low"] < s["mean_terminal"] < s["ci95_high"]
    assert s["p05"] < s["p50"] < s["p95"]
    assert 0.0 <= s["mean_max_drawdown"] <= 1.0
    assert s["worst_max_drawdown"] >= s["mean_max_drawdown"]
    assert s["min_terminal"] <= s["p01"] and s["max_terminal"] >= s["p99"]


def test_sharpe_ratio_is_negative_when_drift_lags_the_risk_free_rate():
    res = _run(mu=0.02, risk_free_rate=0.10, n_simulations=30_000)
    assert summary_statistics(res)["sharpe_ratio"] < 0


def test_confidence_interval_narrows_with_sample_size():
    x = np.random.default_rng(2).normal(100, 20, 100_000)
    w = np.ones_like(x)
    lo_s, hi_s = confidence_interval(x[:1000], w[:1000])
    lo_l, hi_l = confidence_interval(x, w)
    assert (hi_l - lo_l) < (hi_s - lo_s)
    assert lo_l < 100 < hi_l


def test_convergence_table_standard_error_decays_like_one_over_sqrt_n():
    rows = convergence_table(_run(n_simulations=40_000))
    assert rows[-1]["std_error"] < rows[0]["std_error"]
    ns = [r["n"] for r in rows]
    assert ns == sorted(ns)
    # SE(n) * sqrt(n) should be roughly constant.
    ratios = [r["std_error"] * math.sqrt(r["n"]) for r in rows]
    assert max(ratios) / min(ratios) < 1.5


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
def test_var_cvar_against_the_normal_closed_form():
    r = np.random.default_rng(5).normal(0.0, 1.0, 400_000)
    w = np.ones_like(r)
    var, cvar = var_cvar(r, w, 0.95)
    assert abs(var - 1.6449) < 0.02                       # z_{0.95}
    assert abs(cvar - sps.norm.pdf(1.6449) / 0.05) < 0.03  # ~2.063
    assert cvar > var


def test_cvar_dominates_var_at_every_level():
    r = risk_report(_run(n_simulations=30_000))
    assert r["cvar_95"] >= r["var_95"]
    assert r["cvar_99"] >= r["var_99"]
    assert r["var_99"] >= r["var_95"]


def test_probabilities_are_bounded_and_ordered():
    r = risk_report(_run(n_simulations=30_000))
    for key in ("prob_loss", "prob_gain_20", "prob_double",
                "prob_bankruptcy", "prob_hit_stop_loss", "prob_hit_take_profit"):
        assert 0.0 <= r[key] <= 1.0, key
    assert r["prob_gain_20"] >= r["prob_double"]  # doubling implies +20%


def test_barrier_probabilities_use_the_full_path_not_just_the_terminal():
    """A stop-loss touched mid-path must count even if the path recovers."""
    res = _run(n_simulations=20_000, sigma=0.60)
    r = risk_report(res, stop_loss_pct=0.20)
    ended_below = float(np.mean(res.terminal_prices <= 80.0))
    assert r["prob_hit_stop_loss"] >= ended_below
    assert r["stop_loss_level"] == pytest.approx(80.0)


# --------------------------------------------------------------------------- #
# Pricing — the external benchmark
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_monte_carlo_call_matches_black_scholes(strike):
    cfg = _cfg(mu=0.05, use_risk_neutral=True, risk_free_rate=0.05,
               n_simulations=200_000, trading_days=100,
               variance_reduction="antithetic")
    res = MonteCarloEngine().run(cfg)
    mc, se = mc_option_price(res.terminal_prices, strike, rate=0.05,
                             T=1.0, kind="call", weights=res.weights)
    bs = black_scholes(100.0, strike, rate=0.05, sigma=0.20, T=1.0, kind="call")
    assert abs(mc - bs) < 4 * se + 0.02


def test_put_call_parity_holds_in_closed_form():
    c = black_scholes(100, 105, rate=0.03, sigma=0.25, T=0.5, kind="call")
    p = black_scholes(100, 105, rate=0.03, sigma=0.25, T=0.5, kind="put")
    assert abs((c - p) - (100 - 105 * math.exp(-0.03 * 0.5))) < 1e-8


def test_mc_expectation_reports_a_sane_standard_error():
    x = np.random.default_rng(9).normal(10.0, 2.0, 50_000)
    mean, se = mc_expectation(x)
    assert abs(mean - 10.0) < 5 * se
    assert abs(se - 2.0 / math.sqrt(50_000)) < 0.002


# --------------------------------------------------------------------------- #
# Correlation & portfolio
# --------------------------------------------------------------------------- #
def test_nearest_psd_repairs_an_indefinite_matrix():
    bad = np.array([[1.0, 0.95, -0.95],
                    [0.95, 1.0, 0.95],
                    [-0.95, 0.95, 1.0]])
    assert np.min(np.linalg.eigvalsh(bad)) < 0
    fixed = nearest_psd(bad)
    assert np.min(np.linalg.eigvalsh(fixed)) >= -1e-10
    assert np.allclose(np.diag(fixed), 1.0, atol=1e-8)
    cholesky_factor(fixed)  # must not raise


def test_correlated_normals_hit_the_target_correlation():
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    chol = cholesky_factor(corr)
    z = correlated_normals(np.random.default_rng(4), 40_000, 5, chol)
    assert z.shape == (40_000, 5, 2)  # (paths, steps, assets)
    realized = np.corrcoef(z[..., 0].ravel(), z[..., 1].ravel())[0, 1]
    assert abs(realized - 0.6) < 0.02


def test_portfolio_terminal_correlation_matches_the_target():
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    spec = PortfolioSpec(
        names=["A", "B"], s0=np.array([100.0, 50.0]),
        mu=np.array([0.08, 0.06]), sigma=np.array([0.20, 0.30]),
        corr=corr, weights=np.array([0.5, 0.5]), rebalance_every=0)
    res = simulate_portfolio(spec, n_paths=30_000, horizon_years=1.0, seed=3)
    a, b = res.asset_terminals[:, 0], res.asset_terminals[:, 1]
    realized = np.corrcoef(np.log(a / 100.0), np.log(b / 50.0))[0, 1]
    assert abs(realized - 0.6) < 0.03


def test_diversification_lowers_portfolio_volatility():
    spec = PortfolioSpec(
        names=["A", "B"], s0=np.array([100.0, 100.0]),
        mu=np.array([0.08, 0.08]), sigma=np.array([0.30, 0.30]),
        corr=np.eye(2), weights=np.array([0.5, 0.5]), rebalance_every=21)
    res = simulate_portfolio(spec, n_paths=20_000, horizon_years=1.0, seed=5)
    port_sd = np.log(res.terminal_values / res.terminal_values.mean()).std()
    assert port_sd < 0.30  # below single-asset vol at zero correlation


def test_efficient_frontier_max_sharpe_beats_both_endpoints():
    mu = np.array([0.10, 0.06])
    sigma = np.array([0.25, 0.15])
    corr = np.array([[1.0, 0.2], [0.2, 1.0]])
    fr = efficient_frontier(mu, sigma, corr, risk_free_rate=0.02)
    best = fr["max_sharpe"]
    assert best["sharpe"] >= (0.10 - 0.02) / 0.25 - 1e-9
    assert best["sharpe"] >= (0.06 - 0.02) / 0.15 - 1e-9
    assert fr["min_variance"]["vol"] <= 0.15 + 1e-9
    assert abs(best["weights"].sum() - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Config round-trip, scenarios & validation
# --------------------------------------------------------------------------- #
def test_config_json_round_trip_preserves_nested_params():
    cfg = _cfg(model="heston", rng_engine="sobol")
    cfg.heston.kappa = 2.5
    cfg.jumps.intensity = 0.75
    cfg.distribution.name = "student_t"
    cfg.distribution.student_t_df = 4.0
    back = SimulationConfig.from_json(cfg.to_json())
    assert back.model == "heston" and back.rng_engine == "sobol"
    assert back.heston.kappa == 2.5
    assert back.jumps.intensity == 0.75
    assert back.distribution.student_t_df == 4.0
    assert back.to_dict() == cfg.to_dict()


def test_scenarios_shift_the_effective_parameters():
    cfg = _cfg()
    base_mu, base_sig = cfg.effective_drift(), cfg.effective_sigma()
    cfg.scenario.drift_shift = -0.30
    cfg.scenario.vol_multiplier = 2.5
    assert cfg.effective_drift() < base_mu
    assert cfg.effective_sigma() > base_sig


@pytest.mark.parametrize("field,value", [
    ("initial_price", -1.0),
    ("sigma", -0.1),
    ("n_simulations", 0),
    ("trading_days", 0),
    ("horizon_years", 0.0),
])
def test_validate_config_rejects_bad_inputs(field, value):
    with pytest.raises(ValidationError):
        validate_config(_cfg(**{field: value}))


def test_validate_config_rejects_student_t_with_infinite_variance():
    cfg = _cfg()
    cfg.distribution.name = "student_t"
    cfg.distribution.student_t_df = 1.5   # variance undefined for df <= 2
    with pytest.raises(ValidationError):
        validate_config(cfg)


def test_validate_config_rejects_a_vg_martingale_violation():
    cfg = _cfg(model="variance_gamma")
    cfg.vg.theta, cfg.vg.nu = 0.5, 3.0    # 1 - theta*nu - sigma^2*nu/2 <= 0
    with pytest.raises(ValidationError):
        validate_config(cfg)


def test_validate_correlation_matrix_rejects_asymmetry():
    with pytest.raises(ValidationError):
        validate_correlation_matrix(np.array([[1.0, 0.5], [0.2, 1.0]]))


def test_validate_price_series_rejects_non_positive_prices():
    with pytest.raises(ValidationError):
        validate_price_series(np.array([100.0, 0.0, 101.0]))


def test_a_sensible_config_passes_validation():
    validate_config(_cfg())  # must not raise
