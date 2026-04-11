"""
Risk-Neutral Density (RND)
==========================
Computes the risk-neutral density from an implied volatility surface using the
Gatheral (2006) log-moneyness formula derived from Breeden-Litzenberger (1978).

Theory
------
The Breeden-Litzenberger result states:

    p(K, T) = e^{rT} · ∂²C / ∂K²

Re-expressed in terms of total variance  w(k) = σ²(k)·T  and log-forward
moneyness  k = log(K / F),  the density in k-space is:

    g(k) = [1 - k·w'/(2w)]²  -  (w'²/4)·(1/w + 1/4)  +  w''/2

where  w' = dw/dk  and  w'' = d²w/dk².

Converting to strike-space density:

    f(K) = g(k) / [K · √(2π·w(k))] · e^{-rT}

The columns ``dw_dk_svi`` and ``d2w_dk2_svi`` in the calibrated parquet
provide w' and w'' analytically from the SVI / SSVI surface, so no
numerical differentiation is needed.

Usage
-----
    from src.pricing_engines.risk_neutral_density import compute_rnd

    # surface_df is a single (symbol, trade_date, time_to_expiry) slice
    rnd_df = compute_rnd(surface_df, T=0.0822, r=0.0668)

Returns a DataFrame with columns:
    log_moneyness, strike, rnd_density, rnd_density_normalised

Note: This module is intentionally standalone — it reads only from the
calibrated surface DataFrame and has no dependencies on the rest of the
pricing engine.  It can be removed without affecting the pipeline.

References
----------
Gatheral, J. (2006). The Volatility Surface: A Practitioner's Guide.
Breeden, D.T. and Litzenberger, R.H. (1978). Prices of State-Contingent
  Claims Implicit in Option Prices. Journal of Business, 51(4), 621-651.
"""

import numpy as np
import pandas as pd


def compute_rnd(
    surface_df: pd.DataFrame,
    T: float,
    r: float,
    S: float = None,
    q: float = None,
    vol_col: str = "implied_variance_svi",
    dw_dk_col: str = "dw_dk_svi",
    d2w_dk2_col: str = "d2w_dk2_svi",
) -> pd.DataFrame:
    """
    Compute the risk-neutral density for one expiry slice.

    Parameters
    ----------
    surface_df  : DataFrame for a single expiry containing log_moneyness and
                  the vol/derivative columns (vol_col, dw_dk_col, d2w_dk2_col).
                  underlying_value and dividend are read from the DataFrame only
                  when S and q are not supplied explicitly.
    T           : Time to expiry in years.
    r           : Risk-free rate (annualised decimal, e.g. 0.068).
    S           : Spot price.  If None, read from surface_df["underlying_value"].
    q           : Dividend yield (annualised decimal).  If None, read from
                  surface_df["dividend"].  Defaults to 0.0 when missing.
    vol_col     : Column for total variance w = σ²·T.
    dw_dk_col   : Column for first derivative dw/dk.
    d2w_dk2_col : Column for second derivative d²w/dk².

    Returns
    -------
    pd.DataFrame with columns:
        log_moneyness, strike, rnd_density, rnd_density_normalised
    Returns empty DataFrame if fewer than 5 valid points.
    """
    required = {"log_moneyness", vol_col, dw_dk_col, d2w_dk2_col}
    missing = required - set(surface_df.columns)
    if missing:
        raise ValueError(f"compute_rnd: missing columns {missing}")

    sl = (
        surface_df[
            surface_df[vol_col].notna() & (surface_df[vol_col] > 1e-10)
            & surface_df["log_moneyness"].notna()
            & surface_df[dw_dk_col].notna()
            & surface_df[d2w_dk2_col].notna()
        ]
        .sort_values("log_moneyness")
        .copy()
    )
    if len(sl) < 5:
        return pd.DataFrame()

    if S is None:
        S = float(sl["underlying_value"].dropna().iloc[0])
    if q is None:
        q_col = sl["dividend"].dropna()
        q = float(q_col.iloc[0]) if not q_col.empty else 0.0
    F = S * np.exp((r - q) * T)              # forward price

    k    = sl["log_moneyness"].values
    w    = sl[vol_col].values                 # total variance  σ²·T
    w_k  = sl[dw_dk_col].values              # dw/dk
    w_kk = sl[d2w_dk2_col].values            # d²w/dk²

    # ── Gatheral (2006) density in log-moneyness space ────────────────────────
    # g(k) = [1 - k·w'/(2w)]² - (w'²/4)·(1/w + 1/4) + w''/2
    term1 = (1.0 - k * w_k / (2.0 * w)) ** 2
    term2 = (w_k ** 2 / 4.0) * (1.0 / w + 0.25)
    term3 = w_kk / 2.0
    g     = term1 - term2 + term3

    # ── Convert to strike-space density ──────────────────────────────────────
    # f(K) = g(k) / [K · √(2π·w)] · e^{-rT}
    K     = F * np.exp(k)
    denom = K * np.sqrt(np.maximum(2.0 * np.pi * w, 1e-20))
    f_K   = np.where(denom > 1e-20, g * np.exp(-r * T) / denom, np.nan)
    f_K   = np.where(f_K > 0, f_K, np.nan)   # density must be non-negative

    # ── Normalise: ∫ f(K) dK ≈ 1 (trapezoid) ─────────────────────────────────
    dK   = np.diff(K, prepend=K[0])
    area = float(np.nansum(f_K * dK))
    f_Kn = (f_K / area) if area > 1e-12 else f_K

    return pd.DataFrame({
        "log_moneyness":          k,
        "strike":                 np.where(np.isfinite(K), np.round(K), 0).astype(int),
        "rnd_density":            f_K,
        "rnd_density_normalised": f_Kn,
    })
