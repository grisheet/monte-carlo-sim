"""Publication-quality Plotly figures for simulation results.

Every function returns a ``plotly.graph_objects.Figure`` so figures can be
rendered in Streamlit, exported to PNG/SVG/HTML, or embedded in reports.
A shared layout template gives the app a consistent, professional look in
both dark and light themes.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy import stats as sps

from simulator.engine import SimulationResult

PALETTE = {
    "accent": "#4C9BE8",
    "accent2": "#E8A14C",
    "good": "#3FB68B",
    "bad": "#E85D5D",
    "median": "#F2C14E",
    "mean": "#7ED0FF",
    "band": "76, 155, 232",
}


def _template(dark: bool) -> dict:
    fg = "#E8EAED" if dark else "#1F2430"
    grid = "rgba(140,150,165,0.18)"
    return dict(
        template="plotly_dark" if dark else "plotly_white",
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=fg),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=55, r=25, t=60, b=45),
        hovermode="x unified",
        xaxis=dict(gridcolor=grid, zeroline=False),
        yaxis=dict(gridcolor=grid, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )


def price_paths(res: SimulationResult, max_lines: int = 200, dark: bool = True) -> go.Figure:
    fig = go.Figure()
    t = res.time_grid
    paths = res.sample_paths[:max_lines]
    for i, p in enumerate(paths):
        fig.add_trace(go.Scatter(
            x=t, y=p, mode="lines",
            line=dict(width=0.7, color=f"rgba({PALETTE['band']},0.25)"),
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=t, y=res.quantile_bands[4], mode="lines", name="Median",
        line=dict(width=2.5, color=PALETTE["median"]),
    ))
    fig.add_trace(go.Scatter(
        x=t, y=res.mean_path, mode="lines", name="Mean",
        line=dict(width=2.5, color=PALETTE["mean"], dash="dash"),
    ))
    fig.update_layout(
        title=f"Simulated Price Paths (showing {len(paths):,} of "
              f"{res.terminal_prices.size:,})",
        xaxis_title="Years", yaxis_title="Price", **_template(dark),
    )
    return fig


def fan_chart(res: SimulationResult, dark: bool = True) -> go.Figure:
    """Confidence-band fan around the median path."""
    t = res.time_grid
    q = res.quantile_bands  # rows: 1,5,10,25,50,75,90,95,99 pct
    fig = go.Figure()
    bands = [(0, 8, 0.10, "1–99%"), (1, 7, 0.16, "5–95%"),
             (2, 6, 0.22, "10–90%"), (3, 5, 0.30, "25–75%")]
    for lo, hi, alpha, label in bands:
        fig.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([q[hi], q[lo][::-1]]),
            fill="toself", fillcolor=f"rgba({PALETTE['band']},{alpha})",
            line=dict(width=0), name=label, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=t, y=q[4], mode="lines", name="Median",
        line=dict(width=2.5, color=PALETTE["median"]),
    ))
    fig.add_trace(go.Scatter(
        x=t, y=res.mean_path, mode="lines", name="Mean",
        line=dict(width=2, color=PALETTE["mean"], dash="dash"),
    ))
    fig.update_layout(title="Fan Chart — Confidence Bands Over Time",
                      xaxis_title="Years", yaxis_title="Price", **_template(dark))
    return fig


def terminal_distribution(res: SimulationResult, dark: bool = True) -> go.Figure:
    """Histogram + KDE of terminal prices with key markers."""
    s = res.terminal_prices
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=s, nbinsx=120, histnorm="probability density", name="Histogram",
        marker=dict(color=f"rgba({PALETTE['band']},0.55)"),
    ))
    kde = sps.gaussian_kde(s[:: max(1, s.size // 50_000)])
    grid = np.linspace(s.min(), np.quantile(s, 0.999), 400)
    fig.add_trace(go.Scatter(
        x=grid, y=kde(grid), mode="lines", name="KDE",
        line=dict(width=2.5, color=PALETTE["accent2"]),
    ))
    s0 = res.config.initial_price
    for x, name, color in [
        (s0, "Initial price", "#9AA4B2"),
        (float(np.median(s)), "Median", PALETTE["median"]),
        (float(s.mean()), "Mean", PALETTE["mean"]),
    ]:
        fig.add_vline(x=x, line_dash="dot", line_color=color,
                      annotation_text=name, annotation_font_color=color)
    fig.update_layout(title="Terminal Price Distribution",
                      xaxis_title="Price at Horizon", yaxis_title="Density",
                      **_template(dark))
    return fig


def log_return_distribution(res: SimulationResult, dark: bool = True) -> go.Figure:
    r = np.log(res.terminal_prices / res.config.initial_price)
    mu, sd = r.mean(), r.std()
    grid = np.linspace(r.min(), r.max(), 400)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r, nbinsx=120, histnorm="probability density", name="Simulated",
        marker=dict(color=f"rgba({PALETTE['band']},0.55)"),
    ))
    fig.add_trace(go.Scatter(
        x=grid, y=sps.norm.pdf(grid, mu, sd), mode="lines",
        name="Normal fit", line=dict(width=2, color=PALETTE["bad"], dash="dash"),
    ))
    fig.update_layout(title="Horizon Log Returns vs Normal Fit",
                      xaxis_title="log(S_T / S_0)", yaxis_title="Density",
                      **_template(dark))
    return fig


def qq_plot(res: SimulationResult, dark: bool = True) -> go.Figure:
    r = np.log(res.terminal_prices / res.config.initial_price)
    r = (r - r.mean()) / r.std()
    n = min(r.size, 5_000)
    sample = np.sort(np.random.default_rng(0).choice(r, n, replace=False))
    theo = sps.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theo, y=sample, mode="markers", name="Quantiles",
                             marker=dict(size=4, color=PALETTE["accent"])))
    lim = [min(theo.min(), sample.min()), max(theo.max(), sample.max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Normal reference",
                             line=dict(color=PALETTE["bad"], dash="dash")))
    fig.update_layout(title="QQ Plot — Simulated Log Returns vs Normal",
                      xaxis_title="Theoretical quantiles",
                      yaxis_title="Sample quantiles", **_template(dark))
    return fig


def drawdown_plot(res: SimulationResult, dark: bool = True) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=100 * res.max_drawdown, nbinsx=80, name="Max drawdown",
        marker=dict(color=f"rgba(232,93,93,0.6)"),
    ))
    fig.update_layout(title="Distribution of Per-Path Maximum Drawdown",
                      xaxis_title="Max drawdown (%)", yaxis_title="Paths",
                      **_template(dark))
    return fig


def volatility_plot(res: SimulationResult, dark: bool = True) -> go.Figure:
    """Realized vol of sample paths over time; Heston shows simulated vol."""
    fig = go.Figure()
    t = res.time_grid
    if res.variance_paths is not None:
        vols = 100 * np.sqrt(res.variance_paths)
        for p in vols[:60]:
            fig.add_trace(go.Scatter(x=t, y=p, mode="lines", showlegend=False,
                                     line=dict(width=0.7,
                                               color=f"rgba({PALETTE['band']},0.3)"),
                                     hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=t, y=vols.mean(axis=0), mode="lines",
                                 name="Mean instantaneous vol",
                                 line=dict(width=2.5, color=PALETTE["accent2"])))
        title = "Heston Instantaneous Volatility √v_t (annualized %)"
    else:
        lr = np.diff(np.log(res.sample_paths), axis=1)
        dt = res.config.dt
        win = max(5, res.config.trading_days // 12)
        csum = np.cumsum(lr**2, axis=1)
        roll = (csum[:, win:] - csum[:, :-win]) / win
        rv = 100 * np.sqrt(roll / dt)
        tt = t[win + 1:]
        med = np.median(rv, axis=0)
        q1, q3 = np.quantile(rv, [0.25, 0.75], axis=0)
        fig.add_trace(go.Scatter(x=np.concatenate([tt, tt[::-1]]),
                                 y=np.concatenate([q3, q1[::-1]]), fill="toself",
                                 fillcolor=f"rgba({PALETTE['band']},0.25)",
                                 line=dict(width=0), name="IQR", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=tt, y=med, mode="lines", name="Median rolling vol",
                                 line=dict(width=2.5, color=PALETTE["accent2"])))
        title = f"Rolling Realized Volatility ({win}-day window, annualized %)"
    fig.update_layout(title=title, xaxis_title="Years",
                      yaxis_title="Volatility (%)", **_template(dark))
    return fig


def path_heatmap(res: SimulationResult, dark: bool = True) -> go.Figure:
    """Density of paths over (time, price) — a 2D histogram heatmap."""
    t = res.time_grid
    paths = res.sample_paths
    n, m = paths.shape
    tt = np.broadcast_to(t, (n, m)).ravel()
    pp = paths.ravel()
    fig = go.Figure(go.Histogram2d(
        x=tt, y=pp, nbinsx=min(120, m), nbinsy=90,
        colorscale="Viridis", colorbar=dict(title="Paths"),
    ))
    fig.add_trace(go.Scatter(x=t, y=res.quantile_bands[4], mode="lines",
                             name="Median", line=dict(color="white", width=2)))
    fig.update_layout(title="Path Density Heatmap", xaxis_title="Years",
                      yaxis_title="Price", **_template(dark))
    return fig


def box_violin(res: SimulationResult, dark: bool = True) -> go.Figure:
    """Box + violin of prices at intermediate horizons."""
    t = res.time_grid
    idxs = np.unique(np.linspace(1, len(t) - 1, 6).astype(int))
    fig = go.Figure()
    for i in idxs:
        fig.add_trace(go.Violin(
            y=res.sample_paths[:, i], name=f"{t[i]:.2f}y",
            box_visible=True, meanline_visible=True,
            fillcolor=f"rgba({PALETTE['band']},0.35)",
            line_color=PALETTE["accent"], showlegend=False,
        ))
    fig.update_layout(title="Price Distribution at Intermediate Horizons",
                      xaxis_title="Horizon", yaxis_title="Price", **_template(dark))
    return fig


def convergence_plot(table: list[dict], dark: bool = True) -> go.Figure:
    n = [row["n"] for row in table]
    mean = [row["mean"] for row in table]
    se = [row["std_error"] for row in table]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n, y=mean, mode="lines+markers", name="MC mean",
        error_y=dict(type="data", array=[1.96 * s for s in se], visible=True),
        line=dict(color=PALETTE["accent"]),
    ))
    fig.update_xaxes(type="log")
    fig.update_layout(title="Monte Carlo Convergence — Mean ± 95% CI vs N",
                      xaxis_title="Number of paths (log scale)",
                      yaxis_title="Estimated E[S_T]", **_template(dark))
    return fig


def animated_playback(res: SimulationResult, n_lines: int = 40,
                      n_frames: int = 40, dark: bool = True) -> go.Figure:
    """Animated reveal of paths through time."""
    t = res.time_grid
    paths = res.sample_paths[:n_lines]
    steps = np.linspace(2, len(t), n_frames).astype(int)
    frames = []
    for k in steps:
        frames.append(go.Frame(
            data=[go.Scatter(x=t[:k], y=p[:k], mode="lines",
                             line=dict(width=1,
                                       color=f"rgba({PALETTE['band']},0.5)"))
                  for p in paths],
            name=str(k),
        ))
    fig = go.Figure(
        data=[go.Scatter(x=t[:2], y=p[:2], mode="lines",
                         line=dict(width=1, color=f"rgba({PALETTE['band']},0.5)"))
              for p in paths],
        frames=frames,
    )
    layout = _template(dark)
    layout["xaxis"].update(range=[t[0], t[-1]], title="Years")
    layout["yaxis"].update(
        range=[float(paths.min()) * 0.95, float(paths.max()) * 1.05], title="Price")
    fig.update_layout(
        title="Animated Simulation Playback",
        updatemenus=[dict(type="buttons", showactive=False, y=1.12, x=1.0,
                          xanchor="right",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=60,
                                                                    redraw=False),
                                                         fromcurrent=True)]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(mode="immediate")])])],
        showlegend=False, **layout,
    )
    return fig


def correlation_heatmap(corr: np.ndarray, names: list[str], dark: bool = True) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu",
        text=np.round(corr, 2), texttemplate="%{text}",
    ))
    fig.update_layout(title="Asset Correlation Matrix", **_template(dark))
    return fig


def efficient_frontier_plot(ef: dict, names: list[str], dark: bool = True) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=100 * ef["cloud_vol"], y=100 * ef["cloud_ret"], mode="markers",
        marker=dict(size=4, color=ef["cloud_sharpe"], colorscale="Viridis",
                    colorbar=dict(title="Sharpe")),
        name="Random portfolios",
        customdata=np.round(ef["cloud_weights"], 2),
        hovertemplate="vol %{x:.1f}% | ret %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=100 * ef["frontier_vol"], y=100 * ef["frontier_ret"], mode="lines",
        name="Efficient frontier", line=dict(color=PALETTE["median"], width=2.5),
    ))
    for key, label, symbol in [("max_sharpe", "Max Sharpe", "star"),
                               ("min_variance", "Min variance", "diamond")]:
        p = ef[key]
        fig.add_trace(go.Scatter(
            x=[100 * p["vol"]], y=[100 * p["ret"]], mode="markers",
            marker=dict(size=14, symbol=symbol, color=PALETTE["bad"]),
            name=label,
        ))
    fig.update_layout(title="Efficient Frontier",
                      xaxis_title="Volatility (%)",
                      yaxis_title="Expected return (%)", **_template(dark))
    return fig


def portfolio_paths_plot(pres, dark: bool = True) -> go.Figure:
    fig = go.Figure()
    t = pres.time_grid
    for p in pres.value_paths[:150]:
        fig.add_trace(go.Scatter(x=t, y=p, mode="lines", showlegend=False,
                                 line=dict(width=0.7,
                                           color=f"rgba({PALETTE['band']},0.25)"),
                                 hoverinfo="skip"))
    q = pres.quantiles
    fig.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]),
                             y=np.concatenate([q[4], q[0][::-1]]), fill="toself",
                             fillcolor=f"rgba({PALETTE['band']},0.18)",
                             line=dict(width=0), name="5–95%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=q[2], mode="lines", name="Median",
                             line=dict(width=2.5, color=PALETTE["median"])))
    fig.add_trace(go.Scatter(x=t, y=pres.mean_path, mode="lines", name="Mean",
                             line=dict(width=2, color=PALETTE["mean"], dash="dash")))
    fig.update_layout(title="Portfolio Value Simulation", xaxis_title="Years",
                      yaxis_title="Portfolio value", **_template(dark))
    return fig
