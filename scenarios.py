"""Scenario library: named macro/market shocks applied to any model.

Each scenario is expressed as parameter shifts (drift, volatility multiplier,
rates, jump intensity) plus an optional deterministic gap event, so scenarios
compose with every stochastic model rather than being hard-coded paths.
"""

from __future__ import annotations

from config import ScenarioParams

SCENARIO_LIBRARY: dict[str, ScenarioParams] = {
    "base": ScenarioParams(name="base"),
    "bull_market": ScenarioParams(
        name="bull_market", drift_shift=+0.08, vol_multiplier=0.85),
    "bear_market": ScenarioParams(
        name="bear_market", drift_shift=-0.12, vol_multiplier=1.30),
    "recession": ScenarioParams(
        name="recession", drift_shift=-0.18, vol_multiplier=1.50,
        rate_shift=-0.015),
    "rate_hike": ScenarioParams(
        name="rate_hike", drift_shift=-0.04, rate_shift=+0.02),
    "inflation_shock": ScenarioParams(
        name="inflation_shock", drift_shift=-0.05, vol_multiplier=1.25,
        rate_shift=+0.03),
    "volatility_shock": ScenarioParams(
        name="volatility_shock", vol_multiplier=2.0),
    "market_crash": ScenarioParams(
        name="market_crash", drift_shift=-0.10, vol_multiplier=1.8,
        crash_day=21, crash_size=-0.22),
    "black_swan": ScenarioParams(
        name="black_swan", vol_multiplier=2.5, jump_intensity_add=2.0,
        crash_day=10, crash_size=-0.35),
    "flash_crash": ScenarioParams(
        name="flash_crash", crash_day=5, crash_size=-0.12, vol_multiplier=1.1),
    "earnings_gap_down": ScenarioParams(
        name="earnings_gap_down", crash_day=15, crash_size=-0.08),
    "earnings_gap_up": ScenarioParams(
        name="earnings_gap_up", crash_day=15, crash_size=+0.08),
}

SCENARIO_DESCRIPTIONS = {
    "base": "No adjustments — parameters exactly as configured.",
    "bull_market": "Sustained risk-on: drift +8%/yr, volatility damped 15%.",
    "bear_market": "Grinding decline: drift −12%/yr, volatility +30%.",
    "recession": "Recession: drift −18%/yr, vol +50%, policy rates cut 150bp.",
    "rate_hike": "Central bank tightening: +200bp rates, drift drag −4%/yr.",
    "inflation_shock": "Inflation surprise: +300bp rates, drift −5%, vol +25%.",
    "volatility_shock": "Pure vol event (e.g. VIX spike): volatility doubles.",
    "market_crash": "Crash regime: −22% gap in month 1, elevated vol after.",
    "black_swan": "Extreme tail event: −35% gap, jump intensity +2/yr, vol ×2.5.",
    "flash_crash": "One-day −12% air pocket in week 1; conditions normalize.",
    "earnings_gap_down": "Single −8% earnings gap three weeks in.",
    "earnings_gap_up": "Single +8% earnings gap three weeks in.",
}
