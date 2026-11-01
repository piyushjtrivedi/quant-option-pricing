"""
Forward Variance Surface
========================
Computes the implied forward total variance between consecutive expiry slices.

Theory
------
Calendar no-arbitrage requires total implied variance  W(k, T) = σ²(k, T)·T
to be non-decreasing in T for every k.  The **forward total variance** between
two expiries T₁ < T₂ at moneyness k is:

    f(k; T₁, T₂) = W(k, T₂) − W(k, T₁)

and the annualised forward vol is  σ_fwd = √(f / (T₂ − T₁)).

W is already total variance (σ²·T), so it must NOT be multiplied by T again.
For SSVI/SVI-pe surfaces ``W = implied_variance_svi`` (already in the parquet).

Properties
----------
- f ≥ 0 everywhere  ↔  no calendar arbitrage (at fixed log-forward moneyness)
- f / (T₂ − T₁) → ∂W/∂T  as T₂ → T₁  (the numerator of Dupire's formula)
- Flat term structure σ(k, T) = σ  ⇒  σ_fwd = σ
- Integrated across strikes, forward variance is what a forward-starting
  variance swap trades; the per-k value is the smile analogue

Usage
-----
    from src.pricing_engines.forward_variance import compute_forward_variance

    fv_df = compute_forward_variance(surface_df)
    # fv_df has columns: k, T1, T2, T1_days, T2_days, forward_variance,
    #                    forward_vol, calendar_arb_flag

Note: This module is intentionally standalone — it reads only from the
calibrated surface DataFrame and has no dependencies on other pricing modules.
It can be removed without affecting the pipeline.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def compute_forward_variance(
    surface_df: pd.DataFrame,
    vol_col: str = "implied_variance_svi",
    k_grid: np.ndarray = None,
) -> pd.DataFrame:
    """
    Compute implied forward total variance between all consecutive expiry pairs.

    Parameters
    ----------
    surface_df : DataFrame containing at least ``time_to_expiry``,
                 ``log_moneyness``, and the column named by ``vol_col``.
    vol_col    : Column for total variance W = σ²·T  (default: implied_variance_svi).
    k_grid     : Optional moneyness grid to interpolate onto.
                 Defaults to 61 points from -0.20 to +0.20.

    Returns
    -------
    pd.DataFrame with columns:
        log_moneyness      — k value
        T1, T2             — bounding expiries (years)
        T1_days, T2_days   — expiries in calendar days
        forward_variance   — f(k; T1, T2) in total-variance units
        forward_vol        — annualised forward vol  √(f / (T2-T1))  (decimal)
        forward_vol_pct    — forward_vol × 100  (per-cent, intuitive unit)
        calendar_arb_flag  — True if forward_variance < 0  (calendar arb)
    """
    required = {"time_to_expiry", "log_moneyness", vol_col}
    missing  = required - set(surface_df.columns)
    if missing:
        raise ValueError(f"compute_forward_variance: missing columns {missing}")

    if k_grid is None:
        k_grid = np.linspace(-0.20, 0.20, 61)

    expiries = sorted(surface_df["time_to_expiry"].unique())
    if len(expiries) < 2:
        return pd.DataFrame()

    # ── Interpolate W(k, T) onto common k_grid for each expiry ───────────────
    W_by_T: dict[float, np.ndarray] = {}
    for T in expiries:
        sl = (
            surface_df[np.isclose(surface_df["time_to_expiry"], T)]
            .dropna(subset=[vol_col])
            .groupby("log_moneyness", as_index=False)[vol_col].mean()
            .sort_values("log_moneyness")
        )
        if len(sl) < 3:
            continue
        pchip       = PchipInterpolator(sl["log_moneyness"].values,
                                        sl[vol_col].values,
                                        extrapolate=False)
        W_by_T[T]   = pchip(k_grid)

    valid_T = [T for T in expiries if T in W_by_T]
    if len(valid_T) < 2:
        return pd.DataFrame()

    # ── Forward variance between consecutive pairs ────────────────────────────
    rows = []
    for T1, T2 in zip(valid_T[:-1], valid_T[1:]):
        W1  = W_by_T[T1]
        W2  = W_by_T[T2]
        dT  = T2 - T1

        fv          = W2 - W1                     # forward total variance
        fv_floored  = np.where(fv > 0, fv, 0.0)   # floor at 0 for vol calc
        fwd_vol     = np.sqrt(fv_floored / dT)

        for i, k in enumerate(k_grid):
            if np.isnan(fv[i]):
                continue
            rows.append({
                "log_moneyness":     round(float(k), 4),
                "T1":                T1,
                "T2":                T2,
                "T1_days":           round(T1 * 365),
                "T2_days":           round(T2 * 365),
                "forward_variance":  float(fv[i]),
                "forward_vol":       float(fwd_vol[i]),
                "forward_vol_pct":   round(float(fwd_vol[i]) * 100, 2),
                "calendar_arb_flag": bool(fv[i] < 0),
            })

    return pd.DataFrame(rows)
