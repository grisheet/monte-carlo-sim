"""Central configuration for the Monte Carlo Stock Market Simulator.

Every simulation is fully described by a :class:`SimulationConfig`. Configs are
plain dataclasses so they can be serialized to/from JSON, making every run
reproducible and replayable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

APP_NAME = "Monte Carlo Stock Market Simulator"
APP_VERSION = "1.0.0"

TRADING_DAYS_PER_YEAR = 252

SIMULATION_COUNTS = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 250_000]

MODELS = {
    "gbm": "Geometric Brownian Motion",
    "jump_diffusion": "Merton Jump Diffusion",
    "heston": "Heston Stochastic Volatility",
    "mean_reversion": "Ornstein-Uhlenbeck Mean Reversion",
    "variance_gamma": "Variance Gamma",
    "historical_bootstrap": "Historical Bootstrap",
}

RNG_ENGINES = {
    "pcg64": "PCG64 (NumPy default)",
    "mt19937": "Mersenne Twister",
    "sobol": "Sobol sequence (quasi-random)",
    "halton": "Halton sequence (quasi-random)",
    "lhs": "Latin Hypercube Sampling",
}

DISTRIBUTIONS = {
    "normal": "Normal",
    "student_t": "Student t",
    "laplace": "Laplace",
    "lognormal": "Lognormal (centered)",
    "uniform": "Uniform",
    "skew_normal": "Skew Normal",
    "ged": "Generalized Error Distribution",
}

VARIANCE_REDUCTION = {
    "none": "None",
    "antithetic": "Antithetic Variates",
    "control_variate": "Control Variates (terminal price)",
    "importance": "Importance Sampling (drift tilt)",
}

BOOTSTRAP_METHODS = {
    "iid": "IID with replacement",
    "without_replacement": "Without replacement",
    "block": "Block bootstrap",
    "rolling": "Rolling (circular) bootstrap",
}


@dataclass
class JumpParams:
    """Merton jump-diffusion parameters."""

    intensity: float = 0.5        # lambda: expected jumps per year
    mean: float = -0.05           # mu_J: mean of log jump size
    volatility: float = 0.10      # sigma_J: std of log jump size


@dataclass
class HestonParams:
    """Heston stochastic-volatility parameters."""

    kappa: float = 2.0            # speed of mean reversion of variance
    theta: float = 0.04           # long-run variance
    xi: float = 0.30              # volatility of volatility
    rho: float = -0.70            # corr(price shocks, variance shocks)
    v0: float = 0.04              # initial variance


@dataclass
class OUParams:
    """Ornstein-Uhlenbeck (mean reversion) parameters, applied to log-price."""

    speed: float = 3.0            # kappa
    mean: float = 100.0           # long-run price level
    volatility: float = 0.20      # sigma


@dataclass
class VGParams:
    """Variance Gamma parameters."""

    sigma: float = 0.20           # diffusion scale
    nu: float = 0.20              # variance rate of the gamma clock
    theta: float = -0.10          # drift of the Brownian motion in gamma time


@dataclass
class BootstrapParams:
    method: str = "iid"           # key into BOOTSTRAP_METHODS
    block_size: int = 10


@dataclass
class DistributionParams:
    name: str = "normal"          # key into DISTRIBUTIONS
    student_t_df: float = 5.0
    skew: float = 4.0             # skew-normal shape parameter alpha
    ged_beta: float = 1.5         # GED shape (2 = normal, 1 = Laplace)


@dataclass
class ScenarioParams:
    """Additive shocks applied on top of the base parameters."""

    name: str = "base"
    drift_shift: float = 0.0          # added to annual drift
    vol_multiplier: float = 1.0       # multiplies annual volatility
    rate_shift: float = 0.0           # added to risk-free rate
    jump_intensity_add: float = 0.0   # added to jump intensity
    crash_day: int | None = None      # index of a deterministic gap day
    crash_size: float = 0.0           # log-return applied on crash_day


@dataclass
class SimulationConfig:
    """Complete, serializable description of one simulation run."""

    ticker: str = "AAPL"
    initial_price: float = 100.0
    mu: float = 0.08              # expected annual return (drift)
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.0
    sigma: float = 0.20           # annual volatility
    horizon_years: float = 1.0
    trading_days: int = TRADING_DAYS_PER_YEAR
    n_simulations: int = 10_000
    seed: int | None = 42

    model: str = "gbm"
    rng_engine: str = "pcg64"
    variance_reduction: str = "none"
    use_risk_neutral: bool = False

    distribution: DistributionParams = field(default_factory=DistributionParams)
    jumps: JumpParams = field(default_factory=JumpParams)
    heston: HestonParams = field(default_factory=HestonParams)
    ou: OUParams = field(default_factory=OUParams)
    vg: VGParams = field(default_factory=VGParams)
    bootstrap: BootstrapParams = field(default_factory=BootstrapParams)
    scenario: ScenarioParams = field(default_factory=ScenarioParams)

    chunk_size: int = 25_000      # paths simulated per chunk (memory control)

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #
    @property
    def n_steps(self) -> int:
        return max(1, int(round(self.horizon_years * self.trading_days)))

    @property
    def dt(self) -> float:
        return self.horizon_years / self.n_steps

    def effective_drift(self) -> float:
        """Drift after scenario shifts and optional risk-neutral switch."""
        base = self.risk_free_rate if self.use_risk_neutral else self.mu
        return base - self.dividend_yield + self.scenario.drift_shift

    def effective_sigma(self) -> float:
        return self.sigma * self.scenario.vol_multiplier

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationConfig":
        data = dict(data)
        nested = {
            "distribution": DistributionParams,
            "jumps": JumpParams,
            "heston": HestonParams,
            "ou": OUParams,
            "vg": VGParams,
            "bootstrap": BootstrapParams,
            "scenario": ScenarioParams,
        }
        for key, klass in nested.items():
            if key in data and isinstance(data[key], dict):
                data[key] = klass(**data[key])
        known = {f for f in cls.__dataclass_fields__}  # tolerate extra keys
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, text: str) -> "SimulationConfig":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: str | Path) -> "SimulationConfig":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure application-wide logging once and return the root app logger."""
    logger = logging.getLogger("mcsim")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
