# Arbitrage-free residual models

A second family of residual producers alongside the original Nelson-Siegel fit.
Everything downstream — the residual cube, the correlation networks, the metrics,
the GUI — is unchanged; only how the residuals are produced is new.

## Why

`ycn`'s original residual is `observed yield − fitted Nelson-Siegel curve`. Plain
Nelson-Siegel is not arbitrage-free: it omits a convexity correction that grows
with the square of maturity. So a residual from that fit mixes two things — a
genuine deviation of one issuer's curve from its peers, and the curve model's own
missing term, which is largest at exactly the long end where the interesting
cross-issuer dispersion lives.

The arbitrage-free models supply that term. On a panel simulated with real
convexity, long-end residual RMSE falls from about 17bp to about 2bp.

## The models

| Model | Value | What it adds |
|---|---|---|
| Nelson-Siegel | `ns` | The original baseline. No adjustment. |
| AFNS | `afns` | Christensen–Diebold–Rudebusch (2011). Same loadings, plus the closed-form convexity term. |
| Discrete-time AFNS | `dtafns` | Eghbalzadeh, Godin & Gaillardetz (2024). The same idea derived on a period grid. |
| Neural HJM | `neural_hjm` | **Experimental.** A documented subset of Gao & Hyndman (2025). Needs the `neural` extra. |

Evaluated and deliberately not built: Kim–Wright's essentially-affine A₀(3)
(FEDS 2005-33), shadow-rate AFNS, and the five-factor AFGNS. The reasons are
recorded in `ycn.analysis.af_references`, which is the source of truth —
`format_bibliography()` prints it, and the tests keep it honest.

> One naming trap worth flagging: the local PDF `3factorArbitrageFree.pdf` is
> **Kim–Wright, not AFNS**. It has no Nelson-Siegel structure, no decay
> parameter and no yield-adjustment term, and it predates AFNS by two years.

## The idea that makes it cheap

In both AFNS and DTAFNS the correction is **state-independent** — a function of
`(decay, Sigma, maturity)` carrying no factor. So it is a fixed per-maturity
offset, and the fit stays ordinary least squares on the *existing* loadings:

```
y(tau) - adjustment(tau)  =  B(tau)' X
```

No new solver, no optimiser, no new numerical risk. And with `Sigma = 0` the
adjustment vanishes and AFNS reproduces Nelson-Siegel *exactly* —
`tests/test_afns_equivalence.py` asserts it, which is what stops the two
frameworks silently drifting apart.

## The result you should know before reading any output

**For any model whose adjustment is constant over time, the residual correlation
networks are provably identical to the Nelson-Siegel ones.**

Writing `P` for the least-squares residual maker,

```
resid_af = (y - adj) - X (X \ (y - adj)) = resid_ns - P(adj)
```

and `P(adj)` has no time index, so every issuer-tenor residual series is shifted
by a constant. Pearson correlation is invariant under adding a constant, so every
edge and every network metric survives unchanged. Verified to machine precision
in `tests/test_af_compare.py`.

So these models change residual *levels* substantially while leaving *who
deviates together* exactly as it was. Their value here is fit quality and
diagnostics, not network topology. What would move the networks is an adjustment
that varies over time — a rolling-window volatility estimate, or the neural
model's filtered residuals.

## Usage

```python
from ycn.analysis.af_models import ModelSpec, ResidualModel, SigmaScope
from ycn.analysis.residual_networks import residual_cube

# Unchanged: the default is still plain Nelson-Siegel.
issuers, dates, cube, terms, skipped = residual_cube(long, panel, "date")

# Arbitrage-free, with per-issuer volatility.
fits = {}
issuers, dates, cube, terms, skipped = residual_cube(
    long, panel, "date",
    model=ModelSpec(ResidualModel.AFNS, n_refit=1, sigma_scope=SigmaScope.POOLED),
    fits_out=fits,
)
```

Comparing several at once:

```bash
uv run python scripts/compare_af_models.py \
    --db data/ycs_fake.duckdb --table par_rates \
    --models ns,afns,dtafns --sigma-scope per_issuer,pooled \
    --decays 1.0,1.8 --out reports/afns
```

Generate a panel with known parameters to check against:

```bash
uv run python scripts/make_fake_afns_rates.py --out data/ycs_afns_fake.duckdb
```

## Estimation

**Two-step (default).** Fix the decay, recover factors by least squares per
date, fit a VAR to the factor series, map its innovation covariance to a
volatility, rebuild the offset, refit. The seed pass has zero volatility and so
is exactly plain Nelson-Siegel; `n_refit` controls how many passes follow.

The mean-reversion matrix never enters the cross-sectional fit — only the
volatility and the decay do — which keeps the estimator closed-form and the
matrix logarithm off the critical path.

**Kalman (opt-in).** `estimator=Estimator.KALMAN` treats the factors as latent
states and estimates every parameter jointly by maximum likelihood, seeded from
the two-step fit. Slower, and it can find a local optimum, but it avoids the
two-step's errors-in-variables bias: that estimator fits its VAR to *measured*
factors, so the volatility it infers inherits their estimation noise. That bias
is visible in the tests — a quiet panel's volatility is overstated because
measurement noise acts as a floor.

**Volatility scope.** `PER_ISSUER` gives each issuer its own correction;
`POOLED` estimates one across all of them, which is better identified and makes
cross-issuer residual correlation reflect genuine deviation rather than
differences in each issuer's estimated convexity.

## Two traps

**The decay convention is reciprocal.** This repo parameterises by a *scale*,
`exp(-tau/decay)`; the literature uses a *rate*, `exp(-lambda*tau)`. So
`lambda = 1/decay`. Every parameter crossing a module boundary is named `decay`
and is a scale; `lambda_` appears only inside `af_loadings`. Loadings keep the
scale convention deliberately — the reciprocal round-trip is not exact in
floating point (`1/(1/1.8) == 1.8000000000000003`), and that would break the
exactness guarantee above.

**Yield units matter enormously.** The correction scales with the *square* of
volatility, so quoting percent where decimals are expected inflates it ten
thousandfold. The two databases in this project disagree: `data/ycs_fake.duckdb`
is decimals, `ycs_data.duckdb` `par_rates` is percent. `ModelSpec.yield_scale`
defaults to `None`, which auto-detects; pass it explicitly to override.

## Adding a model

See `.mex/patterns/add-residual-model.md`. In short: two pure functions
(loadings and adjustment) in `af_loadings.py`, one `_Kernel` entry in
`af_fit.py`, one `FITTERS` entry in `af_models.py`, and a citation in
`af_references.py` — which the tests require.

## References

Printed by `uv run python scripts/compare_af_models.py --bibliography`, or:

```python
from ycn.analysis.af_references import format_bibliography
print(format_bibliography())
```
