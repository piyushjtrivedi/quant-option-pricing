"""
Per-Slice SVI Fitter
====================
Fits the raw SVI parameterisation independently to each expiry slice.

Raw SVI (Gatheral 2004):
    w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))

Parameters per slice (5 free):
    a     : overall variance level  (> 0)
    b     : smile width / slope     (b > 0)
    rho   : skew asymmetry          (-1 < rho < 1)
    m     : smile centre in k-space (horizontal shift)
    sigma : ATM curvature           (sigma > 0, bowl depth independent of rho)

Why per-slice SVI vs SSVI
--------------------------
SSVI's ATM curvature is θ/2·φ²·(1−ρ²), which vanishes as ρ → −1 (strong put
skew, typical for BankNifty). Per-slice SVI decouples skew (ρ) from curvature
(σ), so it can fit put skew and call smile together.

No-arbitrage
------------
Calendar: soft penalty while fitting (slices fitted in T order), then a
post-fit check that w(k, T1) ≤ w(k, T2) on a k-grid, reported in the fit report.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator

from src.pricing_engines.iv_engines.svi_engine.svi_per_expiry_helpers import SSVIPEHelpers

logger = logging.getLogger(__name__)


# ── per-slice fit result ──────────────────────────────────────────────────

@dataclass
class SVISliceFit:
    T:     float
    a:     float
    b:     float
    rho:   float
    m:     float
    sigma: float
    rmse_vpts: float = np.nan
    converged: bool  = False

    def w(self, k: np.ndarray) -> np.ndarray:
        return SSVIPEHelpers.svi_w(k, self.a, self.b, self.rho, self.m, self.sigma)


# ── surface fit result ────────────────────────────────────────────────────

@dataclass
class SVISurfaceFitReport:
    slice_fits:           list[SVISliceFit]
    overall_rmse_vpts:    float
    calendar_violations:  list[str] = field(default_factory=list)
    fit_acceptable:       bool      = True


# ── main fitter ────────────────────────────────────────────────────────────

class SVIPerExpiryFitter:
    """
    Fits raw SVI independently to each expiry slice then checks calendar no-arb.

    Usage
    -----
    fitter = SVIPerExpiryFitter(iv_bs_df)
    report = fitter.fit()
    iv_svi_df = fitter.build_surface_df()
    """

    # Floor on sigma: sigma → 0 collapses the smile to a kinked linear skew
    SIGMA_MIN: float = 0.02

    # Post-fit calendar check grid
    _CAL_K_GRID = np.linspace(-0.15, 0.15, 121)

    def __init__(self, iv_df: pd.DataFrame):
        # Same rows as SSVI, for a like-for-like comparison
        if "iv_valid_for_ssvi" in iv_df.columns:
            df = iv_df[iv_df["iv_valid_for_ssvi"]].copy()
        else:
            df = iv_df[
                iv_df["implied_volatility"].notna() & (iv_df["implied_volatility"] > 0)
            ].copy()

        self.df             = df
        self.time_to_expiry = sorted(self.df["time_to_expiry"].unique().tolist())
        self.slice_fits:    list[SVISliceFit] = []

    # ── single-slice fit ──────────────────────────────────────────────────

    def _fit_slice(self, T: float, prev_fits: list = None) -> SVISliceFit:
        g     = self.df[self.df["time_to_expiry"] == T].sort_values("log_moneyness")
        k     = g["log_moneyness"].values
        w_obs  = g["bs_implied_variance"].values
        iv_obs = g["implied_volatility"].values   # in % units

        if len(k) < 5:
            # Too few points — return a flat surface at ATM level
            idx   = np.argmin(np.abs(k))
            w_atm = max(float(w_obs[idx]), 1e-6)
            return SVISliceFit(T=T, a=w_atm, b=1e-4, rho=0.0, m=0.0,
                               sigma=SVIPerExpiryFitter.SIGMA_MIN,
                               rmse_vpts=np.nan, converged=False)

        # ── initial guess ────────────────────────────────────────────────
        idx_atm = np.argmin(np.abs(k))
        a0      = max(float(w_obs[idx_atm]) * 0.9, 1e-6)
        # b0: put-wing slope of w vs k, clamped (polyfit is noisy)
        put_mask = k < -0.01
        if put_mask.sum() >= 2:
            slope = float(np.polyfit(k[put_mask], w_obs[put_mask], 1)[0])
            b0 = float(np.clip(slope, 1e-4, 0.5))
        else:
            b0 = 0.01
        rho0   = -0.3
        m0     = 0.0
        sigma0 = max(float(np.std(k)) * 0.5, SVIPerExpiryFitter.SIGMA_MIN)

        x0 = SSVIPEHelpers.pack(a0, b0, rho0, m0, sigma0)

        # ── objective: MSE in vol points (already O(1), weights expiries equally)
        k_cal = np.linspace(-0.15, 0.15, 61)

        def obj(x):
            a, b, rho, m, sigma = SSVIPEHelpers.unpack(x)
            w_model = SSVIPEHelpers.svi_w(k, a, b, rho, m, sigma)
            if np.any(w_model <= 0):
                return 1e20
            iv_model = np.sqrt(np.maximum(w_model / max(T, 1e-12), 0)) * 100
            mse = float(np.mean((iv_model - iv_obs) ** 2))
            pen = SSVIPEHelpers.butterfly_penalty(b, rho)
            pen += SSVIPEHelpers.calendar_penalty(k_cal, a, b, rho, m, sigma, prev_fits or [])
            # Negative w in the extended wings
            k_ext = np.linspace(-0.20, 0.20, 41)
            w_ext = SSVIPEHelpers.svi_w(k_ext, a, b, rho, m, sigma)
            if np.any(w_ext <= 0):
                pen += 1e4
            return mse + pen

        # Bounds in raw space: a, b, sigma ∈ [e^-30, e^10]; rho ∈ (-1, 1); m ∈ [-1, 1]
        _RC = SSVIPEHelpers.RAW_CLIP
        _MC = SSVIPEHelpers.M_CLIP
        bounds = [
            (-30,  _RC),          # x0 → a
            (-30,  _RC),          # x1 → b
            (-10,  10),           # x2 → rho
            (-_MC, _MC),          # x3 → m
            (-30,  _RC),          # x4 → sigma
        ]
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 3000, "ftol": 1e-14, "gtol": 1e-8})

        a, b, rho, m, sigma = SSVIPEHelpers.unpack(res.x)

        w_fitted  = SSVIPEHelpers.svi_w(k, a, b, rho, m, sigma)
        iv_fitted = np.sqrt(np.maximum(w_fitted / max(T, 1e-12), 0)) * 100
        rmse      = float(np.sqrt(np.mean((iv_fitted - iv_obs) ** 2)))

        logger.info(
            "[SVI slice] T=%.4f  a=%.6f  b=%.4f  rho=%.4f  m=%.4f  sigma=%.4f  "
            "RMSE=%.2f vpts  converged=%s",
            T, a, b, rho, m, sigma, rmse, res.success,
        )

        return SVISliceFit(
            T=T, a=a, b=b, rho=rho, m=m, sigma=sigma,
            rmse_vpts=rmse, converged=res.success,
        )

    # ── calendar no-arb check ─────────────────────────────────────────────

    def _check_calendar_arb(self) -> list[str]:
        """
        Check that w(k, T1) <= w(k, T2) for all k on CAL_K_GRID when T1 < T2.
        Returns list of violation strings (empty = no violations).
        """
        violations = []
        fits = sorted(self.slice_fits, key=lambda s: s.T)
        for i in range(len(fits) - 1):
            f1, f2 = fits[i], fits[i + 1]
            w1 = f1.w(self._CAL_K_GRID)
            w2 = f2.w(self._CAL_K_GRID)
            bad = self._CAL_K_GRID[w1 > w2 + 1e-8]
            if len(bad) > 0:
                msg = (
                    f"Calendar arb T={f1.T:.4f}→{f2.T:.4f}: "
                    f"w(T1)>w(T2) at {len(bad)} points, "
                    f"worst k={bad[np.argmax(w1[w1>w2+1e-8]-w2[w1>w2+1e-8])]:.4f}"
                )
                violations.append(msg)
                logger.warning("[SVI cal] %s", msg)
        if not violations:
            logger.info("[SVI cal] No calendar spread arbitrage detected")
        return violations

    # ── public fit ────────────────────────────────────────────────────────

    def fit(self) -> SVISurfaceFitReport:
        # Sequential in T so each slice is penalised against the earlier ones
        self.slice_fits = []
        for T in self.time_to_expiry:
            fit = self._fit_slice(T, prev_fits=list(self.slice_fits))
            self.slice_fits.append(fit)

        all_rmse = [s.rmse_vpts for s in self.slice_fits if not np.isnan(s.rmse_vpts)]
        overall  = float(np.mean(all_rmse)) if all_rmse else np.nan

        cal_viol = self._check_calendar_arb()

        logger.info("[SVI surface] Overall RMSE=%.2f vpts  calendar_violations=%d",
                    overall, len(cal_viol))

        report = SVISurfaceFitReport(
            slice_fits          = self.slice_fits,
            overall_rmse_vpts   = overall,
            calendar_violations = cal_viol,
            fit_acceptable      = (overall <= 1.5 and len(cal_viol) == 0),
        )
        self.report = report
        return report

    # ── surface builders ──────────────────────────────────────────────────

    def build_surface_df(
        self,
        n_expiry_grid: int   = 20,
        sigma_floor:   float = 1e-4,
    ) -> pd.DataFrame:
        """
        Build a dense (T, k) surface DataFrame from the fitted SVI slices.

        Columns: time_to_expiry, log_moneyness, implied_volatility_svi,
                 implied_variance_svi, dupire_local_volatility, dupire_local_variance,
                 dw_dk_svi, d2w_dk2_svi, dw_dT_svi
        """
        if not self.slice_fits:
            raise RuntimeError("Call fit() before build_surface_df()")

        expiries = [s.T for s in self.slice_fits]
        T_grid   = np.unique(np.concatenate((
            np.linspace(min(expiries), max(expiries), n_expiry_grid),
            expiries,
        )))

        k_grid = self.log_moneyness_grid

        # Interpolate each SVI parameter across T with PCHIP
        fits_sorted = sorted(self.slice_fits, key=lambda s: s.T)
        Ts     = np.array([s.T     for s in fits_sorted])
        as_    = np.array([s.a     for s in fits_sorted])
        bs_    = np.array([s.b     for s in fits_sorted])
        rhos   = np.array([s.rho   for s in fits_sorted])
        ms_    = np.array([s.m     for s in fits_sorted])
        sigmas = np.array([s.sigma for s in fits_sorted])

        a_fn     = PchipInterpolator(Ts, as_,    extrapolate=True)
        b_fn     = PchipInterpolator(Ts, bs_,    extrapolate=True)
        rho_fn   = PchipInterpolator(Ts, rhos,   extrapolate=True)
        m_fn     = PchipInterpolator(Ts, ms_,    extrapolate=True)
        sigma_fn = PchipInterpolator(Ts, sigmas, extrapolate=True)

        rows_T, rows_k = [], []
        rows_iv, rows_w = [], []
        rows_wk, rows_wkk, rows_wt = [], [], []
        rows_lv, rows_lvar = [], []

        nk = len(k_grid)

        for i, T in enumerate(T_grid):
            a     = float(np.clip(a_fn(T),     1e-8, None))
            b     = float(np.clip(b_fn(T),     1e-8, None))
            rho   = float(np.clip(rho_fn(T),  -0.999, 0.999))
            m     = float(m_fn(T))
            sigma = float(np.clip(sigma_fn(T), 1e-8, None))

            diff   = k_grid - m
            root   = np.sqrt(diff ** 2 + sigma ** 2)
            w_row  = np.maximum(a + b * (rho * diff + root), 1e-12)
            iv_row = np.sqrt(w_row / max(T, 1e-12)) * 100

            # Analytical: w_k = b(ρ + (k−m)/root),  w_kk = b·σ²/root³
            wk_row  = b * (rho + diff / root)
            wkk_row = b * sigma ** 2 / root ** 3

            rows_T.extend([T] * nk)
            rows_k.extend(k_grid.tolist())
            rows_w.extend(w_row.tolist())
            rows_iv.extend(iv_row.tolist())
            rows_wk.extend(wk_row.tolist())
            rows_wkk.extend(wkk_row.tolist())

        # w_T: numerical gradient over T_grid
        w_matrix = np.array(rows_w).reshape(len(T_grid), nk)
        dw_dT    = np.gradient(w_matrix, T_grid, axis=0)

        for i in range(len(T_grid)):
            w_row   = w_matrix[i]
            wt_row  = dw_dT[i]
            wk_row  = np.array(rows_wk[i*nk:(i+1)*nk])
            wkk_row = np.array(rows_wkk[i*nk:(i+1)*nk])

            # Gatheral (2006) eq. 1.10 — same D as the SSVI builder; D <= 0 → sigma_floor
            # D = (1 - k*w_k/(2w))^2 - (w_k^2/4)*(1/w + 1/4) + w_kk/2
            w_safe = np.maximum(w_row, 1e-12)
            term1  = 1.0 - (k_grid * wk_row) / (2.0 * w_safe)
            term2  = 0.25 * wk_row ** 2 * (1.0 / w_safe + 0.25)
            D      = term1 ** 2 - term2 + 0.5 * wkk_row

            lvar   = np.where(D > 1e-12, wt_row / np.maximum(D, 1e-12), np.nan)
            lvar   = np.where(np.isfinite(lvar) & (lvar > 0), lvar, sigma_floor ** 2)
            lvar   = np.maximum(lvar, sigma_floor ** 2)
            lv     = np.sqrt(lvar) * 100

            rows_wt.extend(wt_row.tolist())
            rows_lv.extend(lv.tolist())
            rows_lvar.extend(lvar.tolist())

        return pd.DataFrame({
            "time_to_expiry":           rows_T,
            "log_moneyness":            rows_k,
            "implied_volatility_svi":   rows_iv,
            "implied_variance_svi":     rows_w,
            "dw_dk_svi":                rows_wk,
            "d2w_dk2_svi":              rows_wkk,
            "dw_dT_svi":                rows_wt,
            "dupire_local_volatility":  rows_lv,
            "dupire_local_variance":    rows_lvar,
        })

    @property
    def log_moneyness_grid(self):
        k = self.df["log_moneyness"].values
        return np.unique(np.concatenate((
            np.linspace(k.min() - 0.5, k.max() + 0.5, 61), k
        )))
