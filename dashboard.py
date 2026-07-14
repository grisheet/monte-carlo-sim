"""Dashboard tabs. Each render_* function owns one tab of the main area."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import streamlit as st

from config import MODELS, SimulationConfig
from data.downloader import DataSourceError, download_prices, load_csv
from data.preprocessing import estimate_parameters, log_returns, rolling_volatility
from simulator import visualization as viz
from simulator.engine import MonteCarloEngine, SimulationResult
from simulator.correlation import cholesky_factor
from simulator.portfolio import PortfolioSpec, efficient_frontier, simulate_portfolio
from simulator.report import generate_html_report
from simulator.risk import risk_report
from simulator.statistics import convergence_table, summary_statistics
from simulator.validation import ValidationError, validate_correlation_matrix
from ui.charts import figure_bytes, render


# --------------------------------------------------------------------------- #
# Data tab
# --------------------------------------------------------------------------- #
def render_data_tab(cfg: SimulationConfig) -> None:
    st.subheader("Historical data & parameter estimation")
    st.caption("Optional: load history to auto-estimate μ, σ and feed the "
               "bootstrap model. You can also run entirely from manual "
               "parameters in the sidebar.")

    mode = st.radio("Data source", ["Manual parameters only",
                                    "Download (yfinance)", "Upload CSV"],
                    horizontal=True)
    prices = None
    if mode == "Download (yfinance)":
        c1, c2, c3 = st.columns([2, 1, 1])
        symbol = c1.text_input("Symbol", cfg.ticker)
        period = c2.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=2)
        if c3.button("Download", width="stretch"):
            try:
                with st.spinner(f"Downloading {symbol}…"):
                    prices = download_prices(symbol, period)
                st.session_state["prices"] = prices
                st.success(f"Loaded {prices.size:,} prices for {symbol}.")
            except DataSourceError as exc:
                st.error(str(exc))
    elif mode == "Upload CSV":
        up = st.file_uploader("CSV with a date column and a price column",
                              type=["csv"])
        if up is not None:
            try:
                prices = load_csv(up)
                st.session_state["prices"] = prices
                st.success(f"Loaded {prices.size:,} prices from {up.name}.")
            except DataSourceError as exc:
                st.error(str(exc))

    prices = st.session_state.get("prices")
    if prices is None:
        st.info("No history loaded — simulations use the manual sidebar "
                "parameters (and a synthetic history for the bootstrap model).")
        return

    est = estimate_parameters(prices)
    st.session_state["hist_returns"] = log_returns(prices)

    c = st.columns(4)
    c[0].metric("Last price", f"{est.last_price:,.2f}")
    c[1].metric("Ann. drift (GBM μ)", f"{est.ann_drift_arith:.2%}",
                help="mean log return × 252 + σ²/2 (GBM-consistent).")
    c[2].metric("Ann. volatility", f"{est.ann_volatility:.2%}",
                help="std of log returns × √252.")
    c[3].metric("EWMA volatility", f"{est.ewma_volatility:.2%}",
                help="RiskMetrics λ=0.94 exponentially weighted vol.")
    c = st.columns(4)
    c[0].metric("Skewness", f"{est.skewness:.3f}")
    c[1].metric("Excess kurtosis", f"{est.kurtosis_excess:.2f}",
                help=">0 means fatter tails than normal — consider Student-t "
                     "innovations or jump diffusion.")
    c[2].metric("Student-t df (MLE)", f"{est.t_df_mle:.1f}")
    c[3].metric("OU speed κ (MLE)", f"{est.ou_speed:.2f}",
                help="AR(1)-mapped mean-reversion speed of the log price.")

    if st.button("→ Apply estimates to sidebar parameters", type="secondary"):
        lc = st.session_state.get("loaded_config", {})
        lc.update({
            "initial_price": est.last_price,
            "mu": round(est.ann_drift_arith, 4),
            "sigma": round(est.ann_volatility, 4),
            "ticker": getattr(prices, "name", cfg.ticker) or cfg.ticker,
            "ou": {"speed": round(est.ou_speed, 3),
                   "mean": round(est.ou_mean_price, 2),
                   "volatility": round(est.ann_volatility, 4)},
            "distribution": {"name": "student_t",
                             "student_t_df": round(est.t_df_mle, 1)},
        })
        st.session_state["loaded_config"] = lc
        st.rerun()

    import plotly.graph_objects as go
    p = pd.Series(prices)
    rv = rolling_volatility(p.reset_index(drop=True), 21)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(p.size)), y=p.values, name="Price",
                             line=dict(color="#4C9BE8")))
    fig.update_layout(title="Loaded price history", height=300,
                      template="plotly_dark" if st.session_state.get("dark_mode", True)
                      else "plotly_white",
                      margin=dict(l=40, r=20, t=50, b=30))
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# Simulation tab
# --------------------------------------------------------------------------- #
def run_simulation(cfg: SimulationConfig) -> SimulationResult | None:
    try:
        bar = st.progress(0.0, text="Starting…")
        cancel_holder = st.empty()
        cancelled = {"flag": False}
        if cancel_holder.button("✖ Cancel"):
            cancelled["flag"] = True

        def progress(frac: float, msg: str) -> bool:
            bar.progress(min(frac, 1.0), text=msg)
            return not cancelled["flag"]

        engine = MonteCarloEngine()
        res = engine.run(cfg, historical_returns=st.session_state.get("hist_returns"),
                         progress=progress)
        bar.progress(1.0, text=f"Done — {res.terminal_prices.size:,} paths in "
                               f"{res.elapsed_seconds:.2f}s "
                               f"({res.paths_per_second:,.0f} paths/s, "
                               f"peak chunk {res.peak_chunk_mb:.0f} MB)")
        cancel_holder.empty()
        st.toast("Simulation complete ✔", icon="✅")
        return res
    except ValidationError as exc:
        for e in exc.errors:
            st.error(e)
        return None


def render_results_tab(res: SimulationResult, dark: bool, max_lines: int) -> None:
    cfg = res.config
    stats = summary_statistics(res)
    st.subheader(f"{cfg.ticker} — {MODELS[cfg.model]}")

    c = st.columns(5)
    ret = stats["expected_return"]
    c[0].metric("Mean terminal", f"{stats['mean_terminal']:,.2f}",
                f"{ret:+.1%} vs S₀")
    c[1].metric("Median terminal", f"{stats['median_terminal']:,.2f}")
    c[2].metric("5–95% range", f"{stats['p05']:,.0f} – {stats['p95']:,.0f}")
    c[3].metric("Sharpe", f"{stats['sharpe_ratio']:.2f}")
    c[4].metric("Prob. of loss",
                f"{np.mean(res.terminal_prices < cfg.initial_price):.1%}")

    t1, t2, t3, t4 = st.tabs(["Paths & fan chart", "Distributions",
                              "Volatility & drawdown", "Playback & heatmap"])
    with t1:
        render(viz.price_paths(res, max_lines, dark), "paths")
        render(viz.fan_chart(res, dark), "fan")
    with t2:
        render(viz.terminal_distribution(res, dark), "terminal")
        col1, col2 = st.columns(2)
        with col1:
            render(viz.log_return_distribution(res, dark), "logret", download=False)
        with col2:
            render(viz.qq_plot(res, dark), "qq", download=False)
        render(viz.box_violin(res, dark), "violin", download=False)
    with t3:
        render(viz.volatility_plot(res, dark), "vol")
        render(viz.drawdown_plot(res, dark), "dd", download=False)
    with t4:
        render(viz.animated_playback(res, dark=dark), "anim", download=False)
        render(viz.path_heatmap(res, dark), "heat", download=False)


# --------------------------------------------------------------------------- #
# Statistics tab
# --------------------------------------------------------------------------- #
def render_statistics_tab(res: SimulationResult, dark: bool) -> None:
    stats = summary_statistics(res)
    st.subheader("Summary statistics")
    left, right = st.columns(2)

    def table(items):
        return pd.DataFrame(items, columns=["Statistic", "Value"]).set_index("Statistic")

    fmt2 = "{:,.2f}".format
    pct = "{:.2%}".format
    left.markdown("**Terminal price distribution**")
    left.dataframe(table([
        ("Paths (effective N)", f"{int(stats['n_paths']):,} "
                                f"({stats['effective_n']:,.0f})"),
        ("Mean", fmt2(stats["mean_terminal"])),
        ("Median", fmt2(stats["median_terminal"])),
        ("Std deviation", fmt2(stats["std_terminal"])),
        ("Variance", fmt2(stats["variance_terminal"])),
        ("Skewness", f"{stats['skewness']:.3f}"),
        ("Excess kurtosis", f"{stats['kurtosis_excess']:.3f}"),
        ("Minimum", fmt2(stats["min_terminal"])),
        ("Maximum", fmt2(stats["max_terminal"])),
        ("95% CI for mean", f"{fmt2(stats['ci95_low'])} – {fmt2(stats['ci95_high'])}"),
    ]), width="stretch")
    left.markdown("**Percentiles**")
    left.dataframe(table([(f"P{p}", fmt2(stats[f"p{p:02d}"]))
                          for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)]),
                   width="stretch")

    right.markdown("**Return & risk-adjusted performance**")
    right.dataframe(table([
        ("Expected horizon return", pct(stats["expected_return"])),
        ("Annualized return", pct(stats["annualized_return"])),
        ("Annualized volatility", pct(stats["annualized_volatility"])),
        ("Semi-variance", f"{stats['semi_variance']:.5f}"),
        ("Downside deviation", pct(stats["downside_deviation"])),
        ("Sharpe ratio", f"{stats['sharpe_ratio']:.3f}"),
        ("Sortino ratio", f"{stats['sortino_ratio']:.3f}"),
        ("Calmar ratio", f"{stats['calmar_ratio']:.3f}"),
        ("Mean max drawdown", pct(stats["mean_max_drawdown"])),
        ("Worst max drawdown", pct(stats["worst_max_drawdown"])),
        ("Ulcer index", f"{stats['ulcer_index']:.2f}"),
        ("Std error of mean", fmt2(stats["std_error_mean"])),
    ]), width="stretch")

    if "control_variate" in res.metadata:
        cv = res.metadata["control_variate"]
        st.markdown("**Control variate diagnostics**")
        st.write(f"Raw mean {cv['raw_mean']:.3f} (SE {cv['raw_se']:.4f}) → "
                 f"CV mean {cv['cv_mean']:.3f} (SE {cv['cv_se']:.4f}), "
                 f"β = {cv['beta']:.3f}, known E[S_T] = "
                 f"{cv['known_expectation']:.3f}")

    st.subheader("Monte Carlo convergence")
    render(viz.convergence_plot(convergence_table(res), dark), "conv",
           download=False)


# --------------------------------------------------------------------------- #
# Risk tab
# --------------------------------------------------------------------------- #
def render_risk_tab(res: SimulationResult) -> None:
    st.subheader("Risk analysis")
    c1, c2, c3 = st.columns(3)
    sl = c1.slider("Stop-loss (% below S₀)", 1, 90, 20,
                   help="Barrier: probability any day's price touches "
                        "S₀ × (1 − x%). First-passage, not terminal-only.") / 100
    tp = c2.slider("Take-profit (% above S₀)", 1, 300, 30) / 100
    bk = c3.slider("Bankruptcy threshold (% below S₀)", 50, 99, 90) / 100

    rr = risk_report(res, sl, tp, bk)
    pct = "{:.2%}".format

    a, b = st.columns(2)
    a.markdown("**Value at Risk (horizon returns)**")
    a.dataframe(pd.DataFrame([
        ("VaR 95%", pct(rr["var_95"])),
        ("CVaR / Expected Shortfall 95%", pct(rr["cvar_95"])),
        ("VaR 99%", pct(rr["var_99"])),
        ("CVaR 99%", pct(rr["cvar_99"])),
        ("Mean max drawdown", pct(rr["mean_max_drawdown"])),
        ("95th pct max drawdown", pct(rr["p95_max_drawdown"])),
    ], columns=["Metric", "Value"]).set_index("Metric"), width="stretch")
    a.caption("VaR₉₅ answers: 'With 95% confidence, my horizon loss will not "
              "exceed X%.' CVaR is the average loss in the worst 5% of cases — "
              "always ≥ VaR and more informative about tail severity.")

    b.markdown("**Event probabilities**")
    b.dataframe(pd.DataFrame([
        ("P(loss at horizon)", pct(rr["prob_loss"])),
        ("P(gain ≥ +20%)", pct(rr["prob_gain_20"])),
        ("P(price doubles)", pct(rr["prob_double"])),
        (f"P(hit stop-loss @ {rr['stop_loss_level']:,.2f})",
         pct(rr["prob_hit_stop_loss"])),
        (f"P(hit take-profit @ {rr['take_profit_level']:,.2f})",
         pct(rr["prob_hit_take_profit"])),
        (f"P(bankruptcy @ {rr['bankruptcy_level']:,.2f})",
         pct(rr["prob_bankruptcy"])),
    ], columns=["Event", "Probability"]).set_index("Event"),
        width="stretch")
    b.caption("Barrier probabilities use each path's running minimum/maximum "
              "on the simulation grid — a path that recovers still counts as "
              "having hit the barrier.")
    st.session_state["risk_report"] = rr


# --------------------------------------------------------------------------- #
# Sensitivity tab
# --------------------------------------------------------------------------- #
def render_sensitivity_tab(cfg: SimulationConfig, dark: bool) -> None:
    st.subheader("Sensitivity analysis")
    st.caption("A fast 2,000-path GBM/jump re-simulation runs on every slider "
               "move to show how outputs respond to each input.")
    c1, c2 = st.columns(2)
    s_sigma = c1.slider("Volatility σ", 0.02, 1.0, float(cfg.sigma), 0.01,
                        key="sens_sigma")
    s_mu = c1.slider("Drift μ", -0.3, 0.4, float(cfg.mu), 0.01, key="sens_mu")
    s_T = c2.slider("Horizon (years)", 0.1, 5.0, float(cfg.horizon_years), 0.1,
                    key="sens_T")
    s_rate = c2.slider("Risk-free rate", 0.0, 0.12, float(cfg.risk_free_rate),
                       0.005, key="sens_rate")
    s_lam = c1.slider("Jump intensity λ", 0.0, 5.0,
                      float(cfg.jumps.intensity if cfg.model == "jump_diffusion"
                            else 0.0), 0.1, key="sens_lam")

    scfg = SimulationConfig.from_dict(cfg.to_dict())
    scfg.sigma, scfg.mu, scfg.horizon_years = s_sigma, s_mu, s_T
    scfg.risk_free_rate = s_rate
    scfg.n_simulations = 2_000
    scfg.chunk_size = 2_000
    scfg.model = "jump_diffusion" if s_lam > 0 else "gbm"
    scfg.jumps.intensity = s_lam
    scfg.rng_engine = "pcg64"

    res = MonteCarloEngine(keep_paths=150).run(scfg)
    stats = summary_statistics(res)
    rr = risk_report(res)
    c = st.columns(4)
    c[0].metric("Mean terminal", f"{stats['mean_terminal']:,.2f}")
    c[1].metric("5–95% width",
                f"{stats['p95'] - stats['p05']:,.1f}",
                help="Spread widens ∝ σ√T.")
    c[2].metric("VaR 95%", f"{rr['var_95']:.1%}")
    c[3].metric("P(loss)", f"{rr['prob_loss']:.1%}")
    render(viz.fan_chart(res, dark), "sens_fan", download=False)


# --------------------------------------------------------------------------- #
# Portfolio tab
# --------------------------------------------------------------------------- #
def render_portfolio_tab(dark: bool) -> None:
    st.subheader("Portfolio simulation")
    st.caption("Correlated multi-asset GBM with rebalancing, plus a "
               "mean-variance efficient frontier.")

    n = st.slider("Number of assets", 2, 6, 3)
    default = pd.DataFrame({
        "Asset": [f"Asset {i+1}" for i in range(n)],
        "S0": [100.0] * n,
        "Drift μ": np.round(np.linspace(0.06, 0.12, n), 3),
        "Vol σ": np.round(np.linspace(0.15, 0.35, n), 3),
        "Weight": [round(1.0 / n, 3)] * n,
    })
    spec_df = st.data_editor(default, hide_index=True, width="stretch",
                             key=f"pf_editor_{n}")

    st.markdown("**Correlation matrix** (symmetric; diagonal fixed at 1)")
    base_corr = np.full((n, n), 0.3)
    np.fill_diagonal(base_corr, 1.0)
    corr_df = st.data_editor(
        pd.DataFrame(base_corr, columns=spec_df["Asset"], index=spec_df["Asset"]),
        width="stretch", key=f"pf_corr_{n}")
    corr = corr_df.to_numpy(dtype=float)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    try:
        validate_correlation_matrix(corr)
    except ValidationError as exc:
        st.warning(f"{exc} — nearest valid (PSD) matrix will be used.")

    c1, c2, c3, c4 = st.columns(4)
    weighting = c1.selectbox("Weighting", ["Custom (table)", "Equal weight"])
    reb = c2.selectbox("Rebalancing", ["Buy & hold", "Monthly", "Quarterly",
                                       "Annual"])
    horizon = c3.slider("Horizon (years)", 0.5, 10.0, 3.0, 0.5, key="pf_T")
    npaths = c4.select_slider("Paths", [1_000, 5_000, 10_000, 25_000],
                              value=5_000, key="pf_n")

    if st.button("Run portfolio simulation", type="primary"):
        weights = (np.full(n, 1.0 / n) if weighting == "Equal weight"
                   else spec_df["Weight"].to_numpy(dtype=float))
        reb_map = {"Buy & hold": 0, "Monthly": 21, "Quarterly": 63, "Annual": 252}
        spec = PortfolioSpec(
            names=list(spec_df["Asset"]),
            s0=spec_df["S0"].to_numpy(dtype=float),
            mu=spec_df["Drift μ"].to_numpy(dtype=float),
            sigma=spec_df["Vol σ"].to_numpy(dtype=float),
            corr=corr, weights=weights, rebalance_every=reb_map[reb],
        )
        with st.spinner("Simulating portfolio…"):
            pres = simulate_portfolio(spec, npaths, horizon)
        st.session_state["portfolio_result"] = pres

    pres = st.session_state.get("portfolio_result")
    if pres is not None:
        term = pres.terminal_values
        v0 = pres.spec.initial_value
        c = st.columns(4)
        c[0].metric("Mean terminal value", f"{term.mean():,.0f}",
                    f"{term.mean()/v0 - 1:+.1%}")
        c[1].metric("Median", f"{np.median(term):,.0f}")
        c[2].metric("P(loss)", f"{np.mean(term < v0):.1%}")
        c[3].metric("Mean max drawdown", f"{pres.max_drawdown.mean():.1%}")
        render(viz.portfolio_paths_plot(pres, dark), "pf_paths")
        render(viz.correlation_heatmap(pres.spec.corr, pres.spec.names, dark),
               "pf_corr", download=False)
        ef = efficient_frontier(pres.spec.mu, pres.spec.sigma, pres.spec.corr,
                                risk_free_rate=0.04)
        render(viz.efficient_frontier_plot(ef, pres.spec.names, dark), "pf_ef",
               download=False)
        ms = ef["max_sharpe"]
        st.write("**Max-Sharpe portfolio (long-only):** " + ", ".join(
            f"{nm} {w:.1%}" for nm, w in zip(pres.spec.names, ms["weights"])) +
            f" → return {ms['ret']:.1%}, vol {ms['vol']:.1%}, "
            f"Sharpe {ms['sharpe']:.2f}")


# --------------------------------------------------------------------------- #
# Export tab
# --------------------------------------------------------------------------- #
def render_export_tab(res: SimulationResult, dark: bool) -> None:
    st.subheader("Exports & reporting")
    cfg = res.config
    stats = summary_statistics(res)
    rr = st.session_state.get("risk_report") or risk_report(res)

    paths_df = pd.DataFrame(
        res.sample_paths.T, index=pd.Index(res.time_grid, name="years"),
        columns=[f"path_{i}" for i in range(res.sample_paths.shape[0])])
    term_df = pd.DataFrame({"terminal_price": res.terminal_prices,
                            "weight": res.weights,
                            "max_drawdown": res.max_drawdown})
    stats_df = pd.DataFrame(sorted(stats.items()), columns=["statistic", "value"])
    risk_df = pd.DataFrame(sorted(rr.items()), columns=["metric", "value"])

    c1, c2, c3 = st.columns(3)
    c1.download_button("CSV — sample paths", paths_df.to_csv(),
                       f"{cfg.ticker}_paths.csv", "text/csv",
                       width="stretch")
    c1.download_button("CSV — terminal prices", term_df.to_csv(index=False),
                       f"{cfg.ticker}_terminals.csv", "text/csv",
                       width="stretch")
    c2.download_button("CSV — summary statistics", stats_df.to_csv(index=False),
                       f"{cfg.ticker}_stats.csv", "text/csv",
                       width="stretch")
    c2.download_button("JSON — config + stats + risk", json.dumps({
        "config": cfg.to_dict(), "statistics": stats, "risk": rr}, indent=2,
        default=float), f"{cfg.ticker}_simulation.json", "application/json",
        width="stretch")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        stats_df.to_excel(xw, sheet_name="Statistics", index=False)
        risk_df.to_excel(xw, sheet_name="Risk", index=False)
        paths_df.iloc[:, :100].to_excel(xw, sheet_name="SamplePaths")
        pd.DataFrame([cfg.to_dict()]).T.rename(columns={0: "value"}).to_excel(
            xw, sheet_name="Config")
    c3.download_button("Excel workbook (.xlsx)", buf.getvalue(),
                       f"{cfg.ticker}_simulation.xlsx",
                       "application/vnd.openxmlformats-officedocument."
                       "spreadsheetml.sheet", width="stretch")

    html = generate_html_report(res, stats, rr, dark_charts=False)
    c3.download_button("HTML report (print → PDF)", html,
                       f"{cfg.ticker}_report.html", "text/html",
                       width="stretch")

    png = figure_bytes(viz.fan_chart(res, dark), "png")
    svg = figure_bytes(viz.fan_chart(res, dark), "svg")
    if png:
        c1.download_button("PNG — fan chart", png, f"{cfg.ticker}_fan.png",
                           "image/png", width="stretch")
    if svg:
        c2.download_button("SVG — fan chart", svg, f"{cfg.ticker}_fan.svg",
                           "image/svg+xml", width="stretch")
    if not png:
        st.caption("Install `kaleido` for one-click PNG/SVG/PDF chart export; "
                   "the 📷 camera icon on every chart also saves PNGs, and the "
                   "HTML report prints to PDF from your browser.")

    st.divider()
    st.markdown("**Replay a saved simulation:** load its JSON in the sidebar "
                "(*Load configuration*) — identical seed and parameters "
                "reproduce the run exactly.")


# --------------------------------------------------------------------------- #
# Math documentation tab
# --------------------------------------------------------------------------- #
def render_math_tab() -> None:
    st.subheader("Mathematical reference")
    st.markdown("Full derivations live in `docs/MATHEMATICS.md`. Key equations:")

    st.markdown("**Geometric Brownian Motion (Black–Scholes dynamics)**")
    st.latex(r"dS_t = \mu S_t\,dt + \sigma S_t\,dW_t \;\Rightarrow\; "
             r"S_T = S_0 \exp\!\big[(\mu - \tfrac{\sigma^2}{2})T + \sigma W_T\big]")
    st.caption("Assumptions: constant μ and σ, continuous trading, no jumps, "
               "lognormal prices, frictionless markets.")

    st.markdown("**Merton Jump Diffusion**")
    st.latex(r"\frac{dS_t}{S_t} = (\mu - \lambda k)\,dt + \sigma\,dW_t + (J-1)\,dN_t,"
             r"\quad k = e^{\mu_J + \sigma_J^2/2} - 1")

    st.markdown("**Heston Stochastic Volatility**")
    st.latex(r"dS_t = \mu S_t\,dt + \sqrt{v_t}S_t\,dW_t^S,\qquad "
             r"dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v,\qquad "
             r"d\langle W^S, W^v\rangle_t = \rho\,dt")

    st.markdown("**Ornstein–Uhlenbeck (mean reversion, exact step)**")
    st.latex(r"X_{t+\Delta} = \bar{x} + (X_t - \bar{x})e^{-\kappa\Delta} + "
             r"\sigma\sqrt{\tfrac{1 - e^{-2\kappa\Delta}}{2\kappa}}\,Z")

    st.markdown("**Variance Gamma (gamma time change)**")
    st.latex(r"X_t = \theta G_t + \sigma W(G_t),\quad G_t \sim \Gamma(t/\nu, \nu),"
             r"\quad \omega = \tfrac{1}{\nu}\ln(1 - \theta\nu - \sigma^2\nu/2)")

    st.markdown("**Monte Carlo expectation & confidence interval**")
    st.latex(r"\hat{\mu}_N = \frac{1}{N}\sum_{i=1}^{N} f(S_T^{(i)}),\qquad "
             r"\hat{\mu}_N \pm z_{\alpha/2}\,\frac{\hat{\sigma}_f}{\sqrt{N}}")

    st.markdown("**Value at Risk & Conditional VaR**")
    st.latex(r"\mathrm{VaR}_\alpha = -\inf\{x : F_R(x) \ge 1-\alpha\},\qquad "
             r"\mathrm{CVaR}_\alpha = -\,\mathbb{E}[R \mid R \le -\mathrm{VaR}_\alpha]")

    st.markdown("**Sharpe ratio**")
    st.latex(r"\mathrm{Sharpe} = \frac{\mathbb{E}[R] - R_f}{\sigma_R}")
