# Monte Carlo Stock Market Simulator

An industrial-quality Monte Carlo simulator for stock prices and portfolios,
built in Python with a Streamlit UI. Six stochastic models, five RNG engines,
variance reduction, weighted risk analytics, scenario stress testing,
portfolio simulation with an efficient frontier, full reproducibility via
JSON configs, and a self-contained HTML report generator.

> Educational software. Nothing here is investment advice.

---

## Features at a glance

| Area | What you get |
|---|---|
| Models | GBM, Merton jump diffusion, Heston stochastic volatility, Ornstein–Uhlenbeck mean reversion (exact step), Variance Gamma (exact gamma time change), historical bootstrap (4 schemes) |
| Randomness | PCG64, Mersenne Twister, Sobol, Halton, Latin Hypercube; 7 innovation distributions, all standardized |
| Variance reduction | Antithetic variates, control variates, importance sampling (weight-aware statistics throughout) |
| Statistics | Mean/median/std, skew, kurtosis, full percentile ladder, Sharpe, Sortino, Calmar, Ulcer index, drawdowns, convergence table |
| Risk | VaR & CVaR (95/99), probability of loss / +20% / doubling / bankruptcy, stop-loss & take-profit hit probabilities computed from the full path |
| Portfolio | Correlated multi-asset GBM, periodic rebalancing, PSD repair of correlation matrices, efficient frontier with max-Sharpe and min-variance portfolios |
| Scenarios | Bull, bear, recession, rate hike, inflation shock, volatility shock, market crash, black swan, flash crash, earnings gaps |
| Data | yfinance download (optional), CSV upload, or manual parameters; drift/vol/EWMA/Student-t df/OU estimation |
| Visualization | Paths, fan chart, terminal distribution + KDE, QQ plot, drawdowns, volatility, path heatmap, box/violin, animated playback, convergence, correlation heatmap, frontier |
| Reproducibility | Seeded runs, save/load the entire configuration as JSON |
| Export | CSV, Excel, JSON, PNG/SVG (chart camera), standalone HTML report (print to PDF) |

## Installation

Requires Python 3.10+.

```bash
git clone <your-repo-url> MonteCarloSimulator
cd MonteCarloSimulator
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`yfinance` is optional — install it only if you want in-app downloads:

```bash
pip install yfinance
```

## Running the app

```bash
streamlit run app.py
```

Then open http://localhost:8501.

### Quick start

1. **Sidebar → Asset & market**: set the starting price, drift μ, and
   volatility σ (or estimate them from data in the **Data** tab).
2. **Model**: pick GBM to start. Hover any ⓘ for the formula and meaning of
   each parameter.
3. Press **Run Simulation**. The **Simulation** tab shows paths and the fan
   chart; **Statistics** and **Risk** fill in automatically.
4. **Export** offers CSV/Excel/JSON downloads and a one-click HTML report.
5. Save your exact setup with **Download config**; reload it any time for a
   bit-for-bit reproducible run (same seed ⇒ same paths).

### Choosing a model

- **GBM** — the baseline; lognormal prices, constant volatility.
- **Jump diffusion** — adds sudden gaps; use for crash-prone assets.
- **Heston** — volatility itself is random and mean-reverting; produces
  volatility clustering and (with ρ < 0) the leverage effect.
- **Mean reversion** — for spreads, rates, commodities, pairs.
- **Variance Gamma** — fat tails and skew without explicit jumps.
- **Historical bootstrap** — no distributional assumption at all; resamples
  your loaded return history (block mode preserves volatility clustering).

### Portfolio mode

The **Portfolio** tab simulates up to 10 correlated assets with optional
periodic rebalancing, and draws the efficient frontier with the max-Sharpe
and minimum-variance portfolios highlighted. Invalid correlation matrices are
automatically repaired to the nearest positive semi-definite matrix (you are
told when this happens).

### Performance notes

- Paths are generated in memory-bounded chunks (default 25,000); the progress
  bar shows throughput and ETA, and **Cancel** stops cleanly at a chunk
  boundary, keeping the completed paths.
- Quasi-random engines (Sobol/Halton/LHS) run as a single block by design and
  are best with ≤ ~65k paths.
- 100k GBM paths × 252 steps runs in a few seconds on a laptop.

## Project layout

```
MonteCarloSimulator/
├── app.py                    # Streamlit entry point
├── config.py                 # SimulationConfig + JSON save/load + constants
├── simulator/
│   ├── engine.py             # chunked, cancellable Monte Carlo engine
│   ├── random_generators.py  # RNG engines, distributions, antithetics
│   ├── statistics.py         # weighted summary statistics, convergence
│   ├── risk.py               # VaR/CVaR, probabilities, drawdowns
│   ├── correlation.py        # PSD repair, Cholesky, correlated normals
│   ├── portfolio.py          # multi-asset simulation, efficient frontier
│   ├── pricing.py            # MC pricing + Black–Scholes benchmark
│   ├── scenarios.py          # scenario library
│   ├── report.py             # standalone HTML report
│   ├── visualization.py      # all Plotly figures
│   ├── validation.py         # config & data validation
│   └── models/               # one file per stochastic model
├── data/
│   ├── downloader.py         # yfinance (optional) + CSV loader
│   └── preprocessing.py      # returns, EWMA, MLE, OU estimation
├── ui/
│   ├── controls.py           # sidebar with tooltips & config I/O
│   ├── dashboard.py          # all main-area tabs
│   └── charts.py             # Plotly render/export helpers
├── tests/                    # 74 tests: model math, engine, analytics
├── docs/MATHEMATICS.md       # every formula, as implemented
└── requirements.txt
```

## Development

```bash
pip install pytest
python -m pytest            # full suite, ~25 s
```

The tests verify statistical correctness, not just plumbing: closed-form GBM
moments, Merton's compensator, Heston full-truncation positivity, the OU
long-run mean, the VG martingale condition, VaR/CVaR against the normal
closed form, and Monte Carlo option prices against Black–Scholes.

### Adding a model

1. Create `simulator/models/your_model.py` subclassing `StochasticModel`
   (implement `simulate(n_paths, gen, stream)` returning
   `(n_paths, n_steps+1)` prices, plus `name` and `latex`).
2. Register it in `simulator/models/__init__.py` and `config.MODELS`.
3. Add parameters to `config.py` (a nested dataclass) and expose them in
   `ui/controls.py`.
4. Add a drift test in `tests/test_models.py`.

## License

MIT — see `LICENSE`.
