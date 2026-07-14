# Mathematical Reference

This document states the models, estimators, and risk measures implemented in
the Monte Carlo Stock Market Simulator, with the exact discretizations used in
the code. Notation: $S_t$ price, $\mu$ annualized drift, $\sigma$ annualized
volatility, $T$ horizon in years, $\Delta = 1/252$ the daily step,
$Z \sim \mathcal{N}(0,1)$ (or a standardized alternative innovation).

---

## 1. Stochastic models

### 1.1 Geometric Brownian Motion (GBM)

SDE:
$$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$$

The code uses the *exact* log-Euler solution, so there is no discretization
bias at any step size:
$$S_{t+\Delta} = S_t \exp\!\Big[\big(\mu - \tfrac{1}{2}\sigma^2\big)\Delta + \sigma\sqrt{\Delta}\,Z\Big]$$

Closed-form moments used by the tests:
$$\mathbb{E}[S_T] = S_0 e^{\mu T}, \qquad
\mathrm{Var}[S_T] = S_0^2 e^{2\mu T}\big(e^{\sigma^2 T} - 1\big)$$

### 1.2 Merton Jump Diffusion

$$dS_t = (\mu - \lambda k) S_t\,dt + \sigma S_t\,dW_t + S_{t^-}\,dJ_t$$

where jumps arrive with Poisson intensity $\lambda$ (per year), jump sizes are
lognormal with log-mean $\mu_J$ and log-std $\sigma_J$, and the compensator
$$k = e^{\mu_J + \sigma_J^2/2} - 1$$
removes the jump contribution from the drift so that
$\mathbb{E}[S_T] = S_0 e^{\mu T}$ still holds.

Per step, the number of jumps is $n \sim \mathrm{Poisson}(\lambda\Delta)$ and
the aggregate jump log-return is simulated exactly as
$n\,\mu_J + \sqrt{n}\,\sigma_J\,Z_J$.

### 1.3 Heston Stochastic Volatility

$$dS_t = \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S, \qquad
dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v, \qquad
d\langle W^S, W^v\rangle_t = \rho\,dt$$

Discretization: **full truncation Euler** — the variance used in both drift
and diffusion terms is $v^+ = \max(v, 0)$, which guarantees non-negative
variance and is the standard low-bias scheme. The Feller condition
$2\kappa\theta \ge \xi^2$ keeps the *continuous* process strictly positive;
the UI warns when it is violated but simulation remains valid.

### 1.4 Ornstein–Uhlenbeck mean reversion (on log price)

$$dX_t = \kappa(\bar{x} - X_t)\,dt + \sigma\,dW_t, \qquad X_t = \ln S_t$$

The code uses the **exact** transition (no Euler error):
$$X_{t+\Delta} = \bar{x} + (X_t - \bar{x})e^{-\kappa\Delta}
+ \sigma\sqrt{\tfrac{1 - e^{-2\kappa\Delta}}{2\kappa}}\,Z$$

Because $\mathbb{E}[e^X] \ne e^{\mathbb{E}[X]}$, a lognormal convexity
correction is applied to $\bar{x}$ so the *price-level* long-run mean equals
the configured target. The $\kappa \to 0$ limit is handled analytically
(variance $\to \sigma^2\Delta$).

### 1.5 Variance Gamma

A Brownian motion evaluated at a gamma-distributed random time:
$$X_t = \theta G_t + \sigma W(G_t), \qquad
G_t \sim \Gamma\!\big(t/\nu,\ \nu\big)$$

with martingale correction
$$\omega = \tfrac{1}{\nu}\ln\!\big(1 - \theta\nu - \sigma^2\nu/2\big),
\qquad S_t = S_0\,e^{\mu t + X_t + \omega t}$$

which requires $1 - \theta\nu - \sigma^2\nu/2 > 0$ (checked by validation).
$\nu$ controls kurtosis; $\theta < 0$ produces the negative skew typical of
equity returns.

### 1.6 Historical bootstrap

Simulated log-return paths are resampled from the loaded historical
log-returns $r_1,\dots,r_n$:

- **IID** — draw each day independently with replacement.
- **Without replacement** — each path is a random permutation of the sample
  (terminal return equals the historical sum exactly).
- **Block** — fixed-length contiguous blocks preserve short-range
  autocorrelation and volatility clustering.
- **Rolling (circular)** — a random starting offset with wrap-around
  preserves the full sample sequence.

---

## 2. Randomness and variance reduction

### 2.1 Innovation distributions

All non-normal innovations are **standardized to zero mean and unit
variance** before use, so $\sigma$ retains its meaning across choices:
Normal, Student-$t$ (fat tails, requires $\text{df} > 2$), Laplace, centered
lognormal, uniform, skew normal, and the Generalized Error Distribution.

### 2.2 Quasi-random sequences

Sobol, Halton, and Latin Hypercube points are mapped to normals through the
inverse CDF $Z = \Phi^{-1}(U)$. Low-discrepancy sequences reduce integration
error from $O(N^{-1/2})$ toward $O(N^{-1}\log^d N)$ for smooth payoffs.

### 2.3 Variance reduction

- **Antithetic variates.** Pair each $Z$ with $-Z$; the negative correlation
  between paired payoffs lowers the variance of the mean.
- **Control variates.** Use $S_T$ itself, whose expectation
  $S_0 e^{\mu T}$ is known, with the optimal coefficient
  $\beta^* = \mathrm{Cov}(f, S_T)/\mathrm{Var}(S_T)$.
- **Importance sampling.** Simulate under a tilted drift and reweight each
  path with the Girsanov likelihood ratio; the weighted estimator remains
  unbiased. All statistics in this app are weight-aware.

---

## 3. Monte Carlo estimation

$$\hat{\mu}_N = \frac{1}{N}\sum_{i=1}^{N} f\big(S^{(i)}\big), \qquad
\mathrm{SE} = \frac{\hat{\sigma}_f}{\sqrt{N}}, \qquad
\text{CI}_{95\%} = \hat{\mu}_N \pm 1.96\,\mathrm{SE}$$

The convergence tab plots $\hat{\mu}_n$ and its error band as $n$ grows —
the $1/\sqrt{N}$ decay is the practical reason more paths help, slowly.

---

## 4. Risk measures

With portfolio return $R = S_T/S_0 - 1$ and losses $L = -R$:

$$\mathrm{VaR}_\alpha = \inf\{\ell : \Pr(L \le \ell) \ge \alpha\}, \qquad
\mathrm{CVaR}_\alpha = \mathbb{E}\big[L \mid L \ge \mathrm{VaR}_\alpha\big]$$

CVaR (expected shortfall) is coherent and always $\ge$ VaR. Barrier
probabilities (stop-loss hit, take-profit hit, bankruptcy) are computed from
each path's running minimum/maximum — i.e., first passage on the daily grid —
not from the terminal price alone.

Drawdown of a path: $DD_t = 1 - S_t / \max_{s\le t} S_s$;
max drawdown is $\max_t DD_t$. The Ulcer index is the root-mean-square
drawdown.

**Performance ratios** (annualized):
$$\mathrm{Sharpe} = \frac{\mathbb{E}[R] - R_f}{\sigma_R}, \qquad
\mathrm{Sortino} = \frac{\mathbb{E}[R] - R_f}{\sigma_{\text{down}}}, \qquad
\mathrm{Calmar} = \frac{\text{annualized return}}{\text{max drawdown}}$$

---

## 5. Parameter estimation from data

With daily log-returns $r_t = \ln(P_t/P_{t-1})$:

- Annualized drift: $\hat{\mu} = 252\,\bar{r} + \tfrac{1}{2}\hat{\sigma}^2$
  (the $\tfrac12\sigma^2$ term converts log drift to arithmetic drift).
- Annualized volatility: $\hat{\sigma} = \sqrt{252}\,s_r$.
- **EWMA volatility** (RiskMetrics):
  $\hat{\sigma}^2_t = \lambda\hat{\sigma}^2_{t-1} + (1-\lambda)r_t^2$ with
  $\lambda = 0.94$.
- Student-$t$ degrees of freedom by maximum likelihood.
- OU speed from the AR(1) fit $r$-coefficient $\phi$:
  $\hat{\kappa} = -\ln\phi/\Delta$.

---

## 6. Portfolio mathematics

Correlated GBM assets are driven by $Z_{\text{corr}} = Z\,L^\top$ where $L$
is the Cholesky factor of the correlation matrix (repaired to the nearest
positive semi-definite matrix by eigenvalue clipping when needed).

Portfolio moments for weights $w$:
$$\mu_p = w^\top \mu, \qquad \sigma_p^2 = w^\top \Sigma\, w, \qquad
\Sigma = D\,C\,D,\ D = \mathrm{diag}(\sigma)$$

The efficient frontier is traced analytically (unconstrained minimum-variance
curve) and overlaid with a random long-only cloud (Dirichlet-sampled weights)
from which the maximum-Sharpe and minimum-variance portfolios are marked.

---

## 7. Option pricing benchmark

Under the risk-neutral measure ($\mu \to r$), a European call is
$$C = e^{-rT}\,\mathbb{E}^\mathbb{Q}\big[(S_T - K)^+\big]$$

The test suite verifies the Monte Carlo price against Black–Scholes:
$$C_{BS} = S_0\Phi(d_1) - Ke^{-rT}\Phi(d_2), \qquad
d_{1,2} = \frac{\ln(S_0/K) + (r \pm \sigma^2/2)T}{\sigma\sqrt{T}}$$

Agreement within Monte Carlo error is the strongest end-to-end correctness
check in the project: it exercises the RNG, the path engine, discounting,
and the estimator simultaneously.

---

## 8. Assumptions and limitations

- Parameters are constant within a run (except Heston variance and scenario
  overlays); real markets have regime changes.
- Daily discretization: barrier probabilities ignore intraday moves.
- The bootstrap assumes the future resembles the sampled history.
- No transaction costs, taxes, or liquidity effects.
- **This software is educational. It is not investment advice.**
