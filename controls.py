"""Sidebar controls: every simulation parameter with an educational tooltip.

Each tooltip states the symbol, the formula it enters, and its real-world
meaning, satisfying the parameter-documentation requirement in one place.
Returns a fully-populated :class:`SimulationConfig`.
"""

from __future__ import annotations

import json

import streamlit as st

from config import (
    BOOTSTRAP_METHODS,
    DISTRIBUTIONS,
    MODELS,
    RNG_ENGINES,
    SIMULATION_COUNTS,
    VARIANCE_REDUCTION,
    BootstrapParams,
    DistributionParams,
    HestonParams,
    JumpParams,
    OUParams,
    SimulationConfig,
    VGParams,
)
from simulator.random_generators import DIST_TOOLTIPS, RNG_TOOLTIPS, VR_TOOLTIPS
from simulator.scenarios import SCENARIO_DESCRIPTIONS, SCENARIO_LIBRARY

HELP = {
    "ticker": "Label for the asset. With 'Download' data mode this symbol is "
              "fetched from Yahoo Finance (e.g. AAPL, MSFT, ^GSPC).",
    "s0": "S₀ — the price at time zero. Every path starts here. All terminal "
          "statistics are relative to this anchor.",
    "mu": "μ — expected annual return (drift). In GBM: dS = μS dt + σS dW. "
          "Over horizon T, E[S_T] = S₀·e^{μT}. Historical equity drift is "
          "roughly 6–10%/yr; it is the hardest parameter to estimate.",
    "rf": "r — annualized risk-free rate. Used in Sharpe/Sortino, discounting, "
          "and as the drift when 'risk-neutral' is enabled (Black-Scholes world).",
    "div": "q — continuous dividend yield. Reduces the effective drift: "
           "total return μ splits into price drift (μ − q) plus dividends.",
    "sigma": "σ — annualized volatility, the std-dev of log returns scaled by "
             "√252. Daily moves are ≈ σ/√252. Typical large-cap equity: 15–35%.",
    "horizon": "T — simulation horizon in years. Uncertainty grows like σ√T: "
               "doubling the horizon widens the distribution by √2.",
    "days": "Trading periods per year. 252 = daily steps; 12 = monthly; "
            "52 = weekly. Step size dt = T / (T × periods).",
    "nsim": "N — number of simulated paths. Monte Carlo standard error decays "
            "as 1/√N: 4× more paths halves the error.",
    "seed": "Seed for the random generator. Same seed + same config = "
            "identical results (full reproducibility). Blank = random.",
    "risk_neutral": "Use r − q as drift instead of μ. Required for option "
                    "pricing (risk-neutral measure ℚ); real-world forecasting "
                    "uses the physical measure ℙ with drift μ.",
    "jump_lambda": "λ — expected number of jumps per year (Poisson intensity). "
                   "P(k jumps in dt) = e^{−λdt}(λdt)^k / k!",
    "jump_mu": "μ_J — mean of the log jump size. Negative values model crash "
               "risk; the drift is compensated by λk, k = e^{μ_J+σ_J²/2} − 1.",
    "jump_sigma": "σ_J — std-dev of log jump sizes; controls how dispersed "
                  "individual jumps are.",
    "kappa": "κ — speed at which variance reverts to θ. Half-life of a vol "
             "shock ≈ ln(2)/κ years.",
    "theta": "θ — long-run variance. Long-run volatility = √θ "
             "(θ = 0.04 → 20% vol).",
    "xi": "ξ — volatility of volatility. Larger ξ → fatter tails and stronger "
          "smile. Feller condition 2κθ ≥ ξ² keeps variance strictly positive.",
    "rho": "ρ — correlation between price and variance shocks. Negative ρ "
           "(typical for equities, ≈ −0.7) creates the leverage effect: "
           "prices fall, volatility rises.",
    "v0": "v₀ — initial variance. Set to (current implied vol)² to start the "
          "simulation from today's volatility level.",
    "ou_speed": "κ — mean-reversion speed of the log price. Half-life of a "
                "deviation ≈ ln(2)/κ years.",
    "ou_mean": "S̄ — long-run price level the process is pulled toward.",
    "ou_sigma": "σ — volatility of the OU process.",
    "vg_sigma": "σ — volatility of the Brownian motion in gamma time.",
    "vg_nu": "ν — variance of the gamma clock. Larger ν → more clustering of "
             "activity → fatter tails (excess kurtosis ≈ 3ν for θ=0).",
    "vg_theta": "θ — drift in gamma time; negative θ generates left skew.",
    "block": "Block length for the block bootstrap: longer blocks preserve "
             "more autocorrelation and volatility clustering.",
    "chunk": "Paths per memory chunk. Lower this if RAM is constrained; the "
             "engine streams chunks and aggregates statistics.",
    "t_df": "Degrees of freedom ν for Student-t innovations. Small ν → fat "
            "tails; variance requires ν > 2. Equity daily returns fit ν≈3–6.",
    "skew": "Skew-normal shape α: negative → heavier left (crash) tail.",
    "ged": "GED shape β: 2 = normal, 1 = Laplace, <2 = fat tails.",
}


def sidebar_controls() -> tuple[SimulationConfig, dict]:
    """Render sidebar; returns (config, ui_options)."""
    st.sidebar.title("Simulation Parameters")

    loaded = st.session_state.get("loaded_config", {})

    def gv(key, default):
        return loaded.get(key, default)

    with st.sidebar.expander("Asset & market", expanded=True):
        ticker = st.text_input("Ticker / label", gv("ticker", "AAPL"), help=HELP["ticker"])
        s0 = st.number_input("Initial price S₀", 0.01, 1e7,
                             float(gv("initial_price", 100.0)), step=1.0, help=HELP["s0"])
        mu = st.slider("Expected return μ (annual)", -0.50, 0.60,
                       float(gv("mu", 0.08)), 0.005, format="%.3f", help=HELP["mu"])
        sigma = st.slider("Volatility σ (annual)", 0.0, 1.50,
                          float(gv("sigma", 0.20)), 0.005, format="%.3f", help=HELP["sigma"])
        rf = st.slider("Risk-free rate r", -0.02, 0.15,
                       float(gv("risk_free_rate", 0.04)), 0.0025, format="%.4f", help=HELP["rf"])
        div = st.slider("Dividend yield q", 0.0, 0.10,
                        float(gv("dividend_yield", 0.0)), 0.0025, format="%.4f", help=HELP["div"])
        risk_neutral = st.checkbox("Risk-neutral drift (r − q)",
                                   bool(gv("use_risk_neutral", False)),
                                   help=HELP["risk_neutral"])

    with st.sidebar.expander("Horizon & sampling", expanded=True):
        horizon = st.slider("Time horizon T (years)", 0.05, 10.0,
                            float(gv("horizon_years", 1.0)), 0.05, help=HELP["horizon"])
        days = st.selectbox("Steps per year", [252, 52, 12],
                            index=[252, 52, 12].index(int(gv("trading_days", 252))),
                            help=HELP["days"])
        nsim = st.select_slider("Number of simulations N", SIMULATION_COUNTS,
                                value=int(gv("n_simulations", 10_000)), help=HELP["nsim"])
        seed_str = st.text_input("Random seed", str(gv("seed", 42)), help=HELP["seed"])
        seed = int(seed_str) if seed_str.strip().lstrip("-").isdigit() else None

    with st.sidebar.expander("Model", expanded=True):
        model = st.selectbox("Stochastic model", list(MODELS),
                             index=list(MODELS).index(gv("model", "gbm")),
                             format_func=MODELS.get)
        jumps, heston, ou, vg, boot = (JumpParams(), HestonParams(), OUParams(),
                                       VGParams(), BootstrapParams())
        if model == "jump_diffusion":
            j = loaded.get("jumps", {})
            jumps = JumpParams(
                intensity=st.slider("Jump intensity λ (/yr)", 0.0, 10.0,
                                    float(j.get("intensity", 0.5)), 0.1,
                                    help=HELP["jump_lambda"]),
                mean=st.slider("Jump mean μ_J (log)", -0.5, 0.5,
                               float(j.get("mean", -0.05)), 0.01, help=HELP["jump_mu"]),
                volatility=st.slider("Jump volatility σ_J", 0.0, 0.6,
                                     float(j.get("volatility", 0.10)), 0.01,
                                     help=HELP["jump_sigma"]),
            )
        elif model == "heston":
            h = loaded.get("heston", {})
            heston = HestonParams(
                kappa=st.slider("Mean reversion κ", 0.0, 10.0,
                                float(h.get("kappa", 2.0)), 0.1, help=HELP["kappa"]),
                theta=st.slider("Long-run variance θ", 0.001, 0.5,
                                float(h.get("theta", 0.04)), 0.001, format="%.3f",
                                help=HELP["theta"]),
                xi=st.slider("Vol of vol ξ", 0.0, 2.0,
                             float(h.get("xi", 0.30)), 0.01, help=HELP["xi"]),
                rho=st.slider("Correlation ρ", -0.99, 0.99,
                              float(h.get("rho", -0.70)), 0.01, help=HELP["rho"]),
                v0=st.slider("Initial variance v₀", 0.001, 0.5,
                             float(h.get("v0", 0.04)), 0.001, format="%.3f",
                             help=HELP["v0"]),
            )
            if 2 * heston.kappa * heston.theta < heston.xi**2:
                st.caption("⚠ Feller condition 2κθ ≥ ξ² violated — variance can "
                           "touch zero (full-truncation scheme keeps it stable).")
        elif model == "mean_reversion":
            o = loaded.get("ou", {})
            ou = OUParams(
                speed=st.slider("Reversion speed κ", 0.0, 20.0,
                                float(o.get("speed", 3.0)), 0.1, help=HELP["ou_speed"]),
                mean=st.number_input("Long-run price S̄", 0.01, 1e7,
                                     float(o.get("mean", s0)), help=HELP["ou_mean"]),
                volatility=st.slider("Volatility σ (OU)", 0.0, 1.5,
                                     float(o.get("volatility", 0.20)), 0.005,
                                     help=HELP["ou_sigma"]),
            )
        elif model == "variance_gamma":
            v = loaded.get("vg", {})
            vg = VGParams(
                sigma=st.slider("VG σ", 0.01, 1.0, float(v.get("sigma", 0.20)),
                                0.005, help=HELP["vg_sigma"]),
                nu=st.slider("VG ν (kurtosis)", 0.01, 2.0, float(v.get("nu", 0.20)),
                             0.01, help=HELP["vg_nu"]),
                theta=st.slider("VG θ (skew)", -0.5, 0.5, float(v.get("theta", -0.10)),
                                0.01, help=HELP["vg_theta"]),
            )
        elif model == "historical_bootstrap":
            b = loaded.get("bootstrap", {})
            boot = BootstrapParams(
                method=st.selectbox("Bootstrap method", list(BOOTSTRAP_METHODS),
                                    index=list(BOOTSTRAP_METHODS).index(
                                        b.get("method", "iid")),
                                    format_func=BOOTSTRAP_METHODS.get),
                block_size=st.slider("Block size (days)", 2, 63,
                                     int(b.get("block_size", 10)), help=HELP["block"]),
            )
            st.caption("Uses downloaded/CSV returns when loaded in the Data tab; "
                       "otherwise a synthetic history matching μ, σ.")

    with st.sidebar.expander("Randomness & distribution"):
        rng_engine = st.selectbox("RNG engine", list(RNG_ENGINES),
                                  index=list(RNG_ENGINES).index(gv("rng_engine", "pcg64")),
                                  format_func=RNG_ENGINES.get,
                                  help="Hover each option's docs below.")
        st.caption(RNG_TOOLTIPS[rng_engine])
        d = loaded.get("distribution", {})
        dist_name = st.selectbox("Innovation distribution", list(DISTRIBUTIONS),
                                 index=list(DISTRIBUTIONS).index(d.get("name", "normal")),
                                 format_func=DISTRIBUTIONS.get)
        st.caption(DIST_TOOLTIPS[dist_name])
        dist = DistributionParams(name=dist_name)
        if dist_name == "student_t":
            dist.student_t_df = st.slider("t degrees of freedom ν", 2.1, 30.0,
                                          float(d.get("student_t_df", 5.0)), 0.1,
                                          help=HELP["t_df"])
        elif dist_name == "skew_normal":
            dist.skew = st.slider("Skew α", -10.0, 10.0, float(d.get("skew", 4.0)),
                                  0.5, help=HELP["skew"])
        elif dist_name == "ged":
            dist.ged_beta = st.slider("GED shape β", 0.5, 4.0,
                                      float(d.get("ged_beta", 1.5)), 0.05,
                                      help=HELP["ged"])
        vr = st.selectbox("Variance reduction", list(VARIANCE_REDUCTION),
                          index=list(VARIANCE_REDUCTION).index(
                              gv("variance_reduction", "none")),
                          format_func=VARIANCE_REDUCTION.get)
        st.caption(VR_TOOLTIPS[vr])

    with st.sidebar.expander("Scenario"):
        sc_loaded = loaded.get("scenario", {}).get("name", "base")
        options = list(SCENARIO_LIBRARY)
        scen_name = st.selectbox(
            "Scenario preset", options,
            index=options.index(sc_loaded) if sc_loaded in options else 0,
            format_func=lambda k: k.replace("_", " ").title())
        st.caption(SCENARIO_DESCRIPTIONS[scen_name])

    with st.sidebar.expander("Performance & output"):
        chunk = st.select_slider("Chunk size (paths)", [5_000, 10_000, 25_000,
                                                        50_000, 100_000],
                                 value=int(gv("chunk_size", 25_000)), help=HELP["chunk"])
        dark = st.toggle("Dark charts", value=st.session_state.get("dark_mode", True))
        st.session_state["dark_mode"] = dark
        max_lines = st.slider("Paths drawn on chart", 25, 400, 150, 25)

    cfg = SimulationConfig(
        ticker=ticker, initial_price=s0, mu=mu, risk_free_rate=rf,
        dividend_yield=div, sigma=sigma, horizon_years=horizon,
        trading_days=days, n_simulations=nsim, seed=seed, model=model,
        rng_engine=rng_engine, variance_reduction=vr,
        use_risk_neutral=risk_neutral, distribution=dist, jumps=jumps,
        heston=heston, ou=ou, vg=vg, bootstrap=boot,
        scenario=SCENARIO_LIBRARY[scen_name], chunk_size=chunk,
    )

    # ------------------------------------------------------------------ #
    st.sidebar.divider()
    c1, c2 = st.sidebar.columns(2)
    run = c1.button("▶ Run Simulation", type="primary", width="stretch")
    reset = c2.button("↺ Reset", width="stretch")
    if reset:
        for key in ("result", "loaded_config", "portfolio_result"):
            st.session_state.pop(key, None)
        st.rerun()

    st.sidebar.download_button(
        "💾 Save configuration (JSON)", cfg.to_json(),
        file_name=f"mcsim_{cfg.ticker}_{cfg.model}.json", mime="application/json",
        width="stretch",
    )
    up = st.sidebar.file_uploader("Load configuration", type="json",
                                  key="cfg_upload")
    if up is not None and st.session_state.get("_cfg_loaded_name") != up.name:
        try:
            st.session_state["loaded_config"] = json.loads(up.getvalue())
            st.session_state["_cfg_loaded_name"] = up.name
            st.sidebar.success("Configuration loaded — controls updated.")
            st.rerun()
        except (json.JSONDecodeError, TypeError) as exc:
            st.sidebar.error(f"Invalid configuration file: {exc}")

    return cfg, {"run": run, "dark": dark, "max_lines": max_lines}
