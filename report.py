"""Professional HTML report generation.

Builds a single self-contained HTML file (charts embedded via Plotly's CDN
loader) with assumptions, model parameters, charts, risk tables, an
interpretation section and standard-of-care caveats. The HTML file prints
cleanly to PDF via the browser's print dialog, which keeps the dependency
footprint small; PNG/SVG chart export is available separately when kaleido
is installed.
"""

from __future__ import annotations

import datetime as dt
import html

import numpy as np
import plotly.io as pio

from config import MODELS, SimulationConfig
from simulator.engine import SimulationResult
from simulator import visualization as viz

_CSS = """
body { font-family: Georgia, 'Times New Roman', serif; color: #1c2431;
       max-width: 960px; margin: 2rem auto; padding: 0 1.5rem; line-height: 1.55; }
h1 { font-size: 1.9rem; border-bottom: 3px solid #2d5f8a; padding-bottom: .4rem; }
h2 { color: #2d5f8a; margin-top: 2.2rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .95rem; }
th, td { border: 1px solid #cfd6e0; padding: .45rem .7rem; text-align: right; }
th { background: #eef2f7; text-align: left; }
td:first-child { text-align: left; }
.meta { color: #5a6577; font-size: .9rem; }
.callout { background: #f4f7fb; border-left: 4px solid #2d5f8a;
           padding: .8rem 1rem; margin: 1rem 0; }
.warn { border-left-color: #b3541e; background: #fbf5f0; }
"""


def _fmt(v: float, pct: bool = False, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{100*v:.{digits}f}%" if pct else f"{v:,.{digits}f}"


def _rows(items: list[tuple[str, str]]) -> str:
    return "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in items)


def generate_html_report(
    res: SimulationResult, stats: dict, risk: dict, dark_charts: bool = False
) -> str:
    cfg: SimulationConfig = res.config
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    figs = {
        "Fan chart": viz.fan_chart(res, dark=dark_charts),
        "Terminal distribution": viz.terminal_distribution(res, dark=dark_charts),
        "Maximum drawdown": viz.drawdown_plot(res, dark=dark_charts),
    }
    chart_html = "".join(
        f"<h2>{name}</h2>" + pio.to_html(fig, include_plotlyjs=False, full_html=False)
        for name, fig in figs.items()
    )

    assumptions = _rows([
        ("Ticker / label", html.escape(cfg.ticker)),
        ("Model", MODELS.get(cfg.model, cfg.model)),
        ("Initial price", _fmt(cfg.initial_price)),
        ("Expected annual return (μ)", _fmt(cfg.mu, pct=True)),
        ("Annual volatility (σ)", _fmt(cfg.sigma, pct=True)),
        ("Risk-free rate", _fmt(cfg.risk_free_rate, pct=True)),
        ("Dividend yield", _fmt(cfg.dividend_yield, pct=True)),
        ("Horizon", f"{cfg.horizon_years:g} years ({cfg.n_steps} steps)"),
        ("Simulated paths", f"{int(stats['n_paths']):,}"),
        ("Innovation distribution", cfg.distribution.name),
        ("RNG engine", cfg.rng_engine),
        ("Variance reduction", cfg.variance_reduction),
        ("Scenario", cfg.scenario.name),
        ("Random seed", str(cfg.seed)),
    ])

    summary = _rows([
        ("Mean terminal price", _fmt(stats["mean_terminal"])),
        ("Median terminal price", _fmt(stats["median_terminal"])),
        ("Std deviation", _fmt(stats["std_terminal"])),
        ("Skewness", _fmt(stats["skewness"])),
        ("Excess kurtosis", _fmt(stats["kurtosis_excess"])),
        ("5th / 95th percentile", f"{_fmt(stats['p05'])} / {_fmt(stats['p95'])}"),
        ("Expected horizon return", _fmt(stats["expected_return"], pct=True)),
        ("Annualized return", _fmt(stats["annualized_return"], pct=True)),
        ("Annualized volatility", _fmt(stats["annualized_volatility"], pct=True)),
        ("Sharpe ratio", _fmt(stats["sharpe_ratio"])),
        ("Sortino ratio", _fmt(stats["sortino_ratio"])),
        ("Calmar ratio", _fmt(stats["calmar_ratio"])),
        ("Ulcer index", _fmt(stats["ulcer_index"])),
        ("95% CI for mean", f"{_fmt(stats['ci95_low'])} – {_fmt(stats['ci95_high'])}"),
    ])

    risk_rows = _rows([
        ("VaR 95% (horizon)", _fmt(risk["var_95"], pct=True)),
        ("CVaR / Expected Shortfall 95%", _fmt(risk["cvar_95"], pct=True)),
        ("VaR 99%", _fmt(risk["var_99"], pct=True)),
        ("CVaR 99%", _fmt(risk["cvar_99"], pct=True)),
        ("Probability of loss", _fmt(risk["prob_loss"], pct=True)),
        ("Probability of ≥ +20%", _fmt(risk["prob_gain_20"], pct=True)),
        ("Probability of doubling", _fmt(risk["prob_double"], pct=True)),
        ("Prob. of hitting stop-loss", _fmt(risk["prob_hit_stop_loss"], pct=True)),
        ("Prob. of hitting take-profit", _fmt(risk["prob_hit_take_profit"], pct=True)),
        ("Prob. below bankruptcy threshold", _fmt(risk["prob_bankruptcy"], pct=True)),
        ("Mean max drawdown", _fmt(risk["mean_max_drawdown"], pct=True)),
        ("95th pct max drawdown", _fmt(risk["p95_max_drawdown"], pct=True)),
    ])

    p_loss = risk["prob_loss"]
    skew = stats["skewness"]
    interp = (
        f"Across {int(stats['n_paths']):,} simulated paths of "
        f"{MODELS.get(cfg.model, cfg.model)}, the terminal price averages "
        f"{_fmt(stats['mean_terminal'])} (95% CI {_fmt(stats['ci95_low'])}–"
        f"{_fmt(stats['ci95_high'])}) against a starting price of "
        f"{_fmt(cfg.initial_price)}. The middle 90% of outcomes spans "
        f"{_fmt(stats['p05'])} to {_fmt(stats['p95'])}. The probability of "
        f"ending below the starting price is {_fmt(p_loss, pct=True)}, with a "
        f"95% VaR of {_fmt(risk['var_95'], pct=True)} and expected shortfall of "
        f"{_fmt(risk['cvar_95'], pct=True)}. Terminal-price skewness of "
        f"{_fmt(skew)} indicates a "
        f"{'right-skewed (lognormal-like)' if skew > 0 else 'left-skewed'} "
        f"outcome distribution."
    )

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Monte Carlo Simulation Report — {html.escape(cfg.ticker)}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>{_CSS}</style></head><body>
<h1>Monte Carlo Simulation Report</h1>
<p class="meta">{html.escape(cfg.ticker)} &middot; generated {now} &middot;
seed {cfg.seed} &middot; fully reproducible from the embedded configuration.</p>

<h2>1. Simulation assumptions</h2>
<table><tr><th>Parameter</th><th>Value</th></tr>{assumptions}</table>

<h2>2. Summary statistics</h2>
<table><tr><th>Statistic</th><th>Value</th></tr>{summary}</table>

<h2>3. Risk analysis</h2>
<table><tr><th>Metric</th><th>Value</th></tr>{risk_rows}</table>

{chart_html}

<h2>Interpretation</h2>
<div class="callout">{interp}</div>

<h2>Notes and recommendations</h2>
<div class="callout">
Monte Carlo results are conditional on the model and its parameters: drift and
volatility estimates carry substantial sampling error, and no diffusion model
captures every feature of real markets. Sensible practice is to (1) compare at
least two models (e.g. GBM vs jump diffusion or bootstrap) before drawing
conclusions, (2) stress the volatility and drift inputs across plausible
ranges, and (3) treat tail probabilities as order-of-magnitude guides rather
than precise forecasts.
</div>
<div class="callout warn">
This report is an educational simulation, not investment advice. Past
performance and simulated performance do not guarantee future results.
</div>

<h2>Reproducibility — configuration JSON</h2>
<pre>{html.escape(cfg.to_json())}</pre>
</body></html>"""
