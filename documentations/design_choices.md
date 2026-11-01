# Design Choices

Why things are done the way they are. For what runs when, see [flow.md](flow.md).

---

## 1. Implied vol

| Choice | Why |
|---|---|
| **Brenner-Subrahmanyam seed:** σ₀ ≈ price / (F·e^{−rT}·√(T/2π)) | Close enough for Newton to converge in a few steps on liquid strikes |
| **Newton with bisection fallback** | Newton is fast but fails when vega → 0. The solver falls back to bisection when vega < 1e-8 or a step leaves [1e-6, 5]. |
| **Bounds check first** (intrinsic < price < max) | Prices outside the bounds have no IV. Return NaN, never 0, so they can't contaminate the fit. |
| **Gap-fill in total variance w = σ²T, PCHIP, no extrapolation** | Shape-preserving, and only fills between real quotes, so no phantom wings |
| **Fit only out-of-the-money quotes with \|k\| ≤ 0.12** (`iv_valid_for_ssvi`) | In-the-money quotes are mostly intrinsic value and add noise; deep wings are illiquid and tick-sensitive |

## 2. SSVI

- **Model:** w(k, θ) = θ/2 · [1 + ρφk + √((φk + ρ)² + 1 − ρ²)], with one θ per expiry and global φ, ρ.
- **No-arbitrage (Gatheral-Jacquier 2014):** per expiry, θφ(1+|ρ|) < 4 (wing) and θφ²(1+|ρ|) ≤ 4 (butterfly). These are soft quadratic penalties (weight 1e4). Separately, any surface whose call prices aren't convex in strike on the grid is rejected outright.
- **Calendar by construction:** θᵢ = cumsum(softplus(δ)), so θ always increases with expiry, and the δ stay O(1) for the optimiser.
- **Parameter maps:** φ = softplus(x) + 0.1 stops a collapse to a flat smile; ρ = tanh(x) keeps ρ ∈ (−1, 1) smoothly.
- **Objective ×1e6:** the raw total-variance MSE (~1e-7) gives gradients below L-BFGS-B's tolerance, so without scaling the optimiser would stop at its starting point.
- **Error averaged per expiry, not pooled:** otherwise the front expiry's many strikes dominate the fit.
- **Known limit:** a single φ and ρ can't match BANKNIFTY's smile; ρ pins at −1. The next step is a power-law φ(θ).

## 3. Per-slice SVI

- **Why:** raw SVI per expiry, w(k) = a + b[ρ(k−m) + √((k−m)² + σ²)], separates skew (ρ) from curvature (σ). That lets it fit put skew and call smile together, which SSVI can't.
- **Objective in vol points:** total-variance error would under-weight short expiries, whose w is small.
- **Sequential fitting:** slices are fitted in expiry order, each penalised against the earlier ones, then checked for calendar arbitrage afterwards. The penalty is currently too weak to bind (see Corrections).
- **Smile-stress flag:** set when SSVI RMSE − SVI RMSE ≥ 1.2 vol points. This threshold is a heuristic; on the current data the gap is 0.8–17 vol points and the flag is set on 14 of 17 dates.

## 4. Dupire local vol

- **Gatheral (2006) log-moneyness form:** σ²_loc = w_T / D, with D = (1 − k·w_k/2w)² − (w_k²/4)(1/w + ¼) + w_kk/2. This avoids strike-space derivatives.
- **Derivatives:** w_k and w_kk are analytical for both models. w_T is analytical for SSVI (θ′(T) from PCHIP) and numerical for per-slice SVI.
- **D ≤ 0:** that is butterfly arbitrage, so local vol is undefined there and falls to the σ floor. D is the same g(k) that appears in the risk-neutral density.

## 5. Rates and dividends

- **NSS curve through MIBOR** (14-day, 1-month, 3-month): gives a smooth r(T) at any expiry. Linear interpolation would kink at each tenor.
- **q(T) = r(T) − ln(F/S)/T from futures**, so no dividend schedule is needed. It is interpolated linearly across expiries and held flat outside them.
  - q here is a residual that absorbs the gap between MIBOR and the funding rate futures imply. On most dates it is negative.

## 6. Data quality

- **Run after IV**, because the vol regime needs ATM IV: low < 12% ≤ medium < 16% ≤ high.
- **Stale = close *and* OI unchanged:** this matches NSE's carry-forward of untraded contracts. Either condition alone gives false positives.
- **Put-call-parity failure flags both legs:** we can't tell which leg is wrong.
- **High-vol regime:** the parity tolerance widens from 0.5% to 1.0%.
- The flags are stored and charted, but **don't currently filter the fit**.

## 7. Storage and dashboard

- **One Parquet file per symbol, dates as rows:** a history query is one read; an upsert replaces that date's rows.
- **`DEMO_MODE`** reads the committed Parquet, so the dashboard runs on Streamlit Cloud with no backend. Live mode shares all the chart code.
- **`_safe()`** wraps each chart, so one failing chart shows a warning instead of breaking the page. Checkboxes toggle each section.
- **Colours:**

  | Colour | Meaning |
  |---|---|
  | Blue `#2196F3` | Market BS IV |
  | Amber `#FF6F00` | SSVI |
  | Green `#2E7D32` | Per-slice SVI |
  | Purple `#7B1FA2` | Local vol |
  | Red `#E53935` | Risk / arbitrage |

  Open circles mark gap-filled IVs. Smile plots are clipped to |k| ≤ 0.15.

## 8. Out of scope

Heston and rough-vol models, intraday data, and cross-asset correlation. Wing extrapolation exists analytically but isn't shown, so model artefacts don't pass as market data.

---

## Corrections (Nov 2026)

| Bug | Effect | Fix |
|---|---|---|
| SSVI butterfly condition missing θ: φ²(1+\|ρ\|) ≤ 4 | Capped φ at ≈1.4 on every date; SSVI underfit | θφ²(1+\|ρ\|) ≤ 4; SSVI RMSE fell 15–32% |
| Dupire D: sign flipped on the w_k² term, w_kk/2 missing | Distorted local vol; D was always > 0, so arbitrage was invisible | Correct D; D ≤ 0 → floor |
| Risk-neutral density missing exp(−d₋²/2), spurious e^{−rT} | Wrong density shape | Now matches lognormal / Breeden-Litzenberger to < 0.02% |
| Forward variance computed W₂T₂ − W₁T₁ | Double-counted T; the calendar check was too weak | W₂ − W₁ |

**Still open:** the per-slice SVI calendar penalty (coefficient 50) is about 10⁴× too weak to bind, and simply raising it makes some dates worse. It needs a hard constraint or a joint fit. The SVI wing bound uses b(1+|ρ|) ≤ 1, but Lee's bound is 2.
