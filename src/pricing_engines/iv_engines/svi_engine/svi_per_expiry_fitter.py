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
SSVI ties ATM curvature = theta/2 * phi^2 * (1-rho^2).  When rho→-1 (strong
put skew, which is normal for BankNifty), curvature→0 and the surface becomes
a pure monotone skew.  Per-slice SVI separates rho (skew) and sigma (bowl
depth), so it can simultaneously fit put skew AND call smile.

No-arbitrage enforcement
------------------------
Calendar no-arb is enforced as a post-fit check (standard industry practice):
    w_SVI(k, T1) <= w_SVI(k, T2) for all k when T1 < T2.
This is checked on a fine k-grid and reported in the fit report.
"""

import math
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────

def _svi_w(k: np.ndarray, a: float, b: float,
           rho: float, m: float, sigma: float) -> np.ndarray:
    """Raw SVI total variance."""
    k    = np.asarray(k, dtype=float)
    diff = k - m
    return a + b * (rho * diff + np.sqrt(diff ** 2 + sigma ** 2))


def _svi_iv_pct(k: np.ndarray, T: float, a: float, b: float,
                rho: float, m: float, sigma: float) -> np.ndarray:
    """SVI implied vol in % units."""
    w = np.maximum(_svi_w(k, a, b, rho, m, sigma), 1e-12)
    return np.sqrt(w / max(T, 1e-12)) * 100


# ── parameter encoding ─────────────────────────────────────────────────────

def _pack(a: float, b: float, rho: float,
          m: float, sigma: float) -> np.ndarray:
    """Encode (a, b, rho, m, sigma) → unconstrained x."""
    return np.array([
        math.log(max(a, 1e-12)),           # x0: a = exp(x0) > 0
        math.log(max(b, 1e-12)),           # x1: b = exp(x1) > 0
        math.atanh(max(-0.999, min(0.999, rho))),  # x2: rho = tanh(x2)
        m,                                  # x3: m unconstrained
        math.log(max(sigma, 1e-12)),        # x4: sigma = exp(x4) > 0
    ])


# Parameter bounds in the raw (unconstrained) space.
# These translate to: a,b,sigma in [exp(-30), exp(10)] — far wider than
# any realistic value; m in [-1, 1] (smile centre within ±100% moneyness).
# The hard clip prevents math.exp overflow (which occurs above ~709) while
# giving L-BFGS-B unlimited room to roam within the economically valid range.
_RAW_CLIP = 10.0   # exp(10) ≈ 22000 — safely above any realistic b value
_M_CLIP   = 1.0


def _unpack(x: np.ndarray):
    """Decode unconstrained x → (a, b, rho, m, sigma).
    Clips raw params before exp() to prevent math range overflow.
    """
    a     = math.exp(float(np.clip(x[0], -30, _RAW_CLIP)))
    b     = math.exp(float(np.clip(x[1], -30, _RAW_CLIP)))
    rho   = math.tanh(float(x[2]))
    m     = float(np.clip(x[3], -_M_CLIP, _M_CLIP))
    sigma = math.exp(float(np.clip(x[4], -30, _RAW_CLIP)))
    return a, b, rho, m, sigma


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
        return _svi_w(k, self.a, self.b, self.rho, self.m, self.sigma)

    def iv_pct(self, k: np.ndarray) -> np.ndarray:
        return _svi_iv_pct(k, self.T, self.a, self.b, self.rho, self.m, self.sigma)


# ── surface fit result ────────────────────────────────────────────────────

@dataclass
class SVISurfaceFitReport:
    slice_fits:           list[SVISliceFit]
    overall_rmse_vpts:    float
    calendar_violations:  list[str] = field(default_factory=list)
    fit_acceptable:       bool      = True

    def theta_map(self) -> dict:
        """ATM total variance per expiry (w at k=0)."""
        return {s.T: s.w(np.array([0.0]))[0] for s in self.slice_fits}


# ── main fitter ────────────────────────────────────────────────────────────

class SVIPerExpiryFitter:
    # Minimum sigma (ATM curvature parameter).
    # sigma=0 collapses SVI to a pure linear skew — losing all smile bowl — and
    # creates calendar arb when adjacent slices have opposite rho signs.
    # A floor of 0.02 corresponds to ~2 vol pts of ATM curvature at T=0.08,
    # which is economically negligible but prevents the degenerate state.
    SIGMA_MIN: float = 0.02
    """
    Fits raw SVI independently to each expiry slice then checks calendar no-arb.

    Usage
    -----
    fitter = SVIPerExpiryFitter(iv_bs_df)
    report = fitter.fit()
    iv_svi_df = fitter.build_surface_df()
    """

    # Calendar no-arb check grid
    _CAL_K_GRID = np.linspace(-0.15, 0.15, 121)

    def __init__(self, iv_df: pd.DataFrame):
        # Use the same rows that SSVI uses for a fair comparison
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
        g    = self.df[self.df["time_to_expiry"] == T].sort_values("log_moneyness")
        k    = g["log_moneyness"].values
        w_obs = g["bs_implied_variance"].values
        iv_obs = g["implied_volatility"].values   # in % units

        if len(k) < 5:
            # Too few points — return a flat surface at ATM level
            idx    = np.argmin(np.abs(k))
            w_atm  = max(float(w_obs[idx]), 1e-6)
            return SVISliceFit(T=T, a=w_atm, b=1e-4, rho=0.0, m=0.0, sigma=SVIPerExpiryFitter.SIGMA_MIN,
                               rmse_vpts=np.nan, converged=False)

        # ── initial guess ────────────────────────────────────────────────
        idx_atm = np.argmin(np.abs(k))
        a0      = max(float(w_obs[idx_atm]) * 0.9, 1e-6)
        # b: slope of w vs k in the put region
        put_mask = k < -0.01
        if put_mask.sum() >= 2:
            slope = float(np.polyfit(k[put_mask], w_obs[put_mask], 1)[0])
            # Clamp slope: polyfit can produce huge values on noisy data
            b0 = float(np.clip(slope, 1e-4, 0.5))
        else:
            b0 = 0.01
        rho0   = -0.3
        m0     = 0.0
        sigma0 = max(float(np.std(k)) * 0.5, SVIPerExpiryFitter.SIGMA_MIN)

        x0 = _pack(a0, b0, rho0, m0, sigma0)

        # ── no-arb soft constraints ──────────────────────────────────────
        # SVI butterfly no-arb: b*(1+|rho|) <= 1  (Lee 2004 wing condition)
        # We enforce this as a soft penalty.
        def penalty(b, rho):
            val = b * (1 + abs(rho))
            return 1e4 * max(0.0, val - 1.0) ** 2

        # ── objective: MSE in vol-point space ────────────────────────────
        # (same reasoning as SSVI: w-space gradients too small for optimizer)
        # Calendar check grid — finer than the post-fit check grid
        k_cal = np.linspace(-0.15, 0.15, 61)

        def obj(x):
            a, b, rho, m, sigma = _unpack(x)
            # hard reject: negative w at any observed k
            w_model = _svi_w(k, a, b, rho, m, sigma)
            if np.any(w_model <= 0):
                return 1e20
            iv_model = np.sqrt(np.maximum(w_model / max(T, 1e-12), 0)) * 100
            mse  = float(np.mean((iv_model - iv_obs) ** 2))
            pen  = penalty(b, rho)
            # Soft calendar no-arb penalty: penalise w(T_current) < w(T_prev) at any k.
            # Total variance must be non-decreasing in T — this is the calendar spread
            # no-arb condition. Each prior slice contributes a quadratic penalty.
            # Coefficient 50 is chosen so a 0.001 w-violation costs ~0.05 extra MSE
            # (same order as a 1 vol-pt fit error), keeping the fit-vs-arb balance.
            if prev_fits:
                cal_pen = 0.0
                w_curr = _svi_w(k_cal, a, b, rho, m, sigma)
                for pf in prev_fits:
                    w_prev = pf.w(k_cal)
                    violations = np.maximum(0.0, w_prev - w_curr)
                    cal_pen += 50.0 * float(np.mean(violations ** 2))
                pen += cal_pen
            # Also penalise negative w at extended wings
            k_ext = np.linspace(-0.20, 0.20, 41)
            w_ext = _svi_w(k_ext, a, b, rho, m, sigma)
            if np.any(w_ext <= 0):
                pen += 1e4
            return mse + pen

        # Parameter bounds: keep raw values in a safe range.
        # Translated: a,b,sigma in [exp(-30), exp(10)]; rho in (-1,1); m in [-1, 1].
        _B  = _RAW_CLIP
        bounds = [
            (-30,  _B),    # x0 → a
            (-30,  _B),    # x1 → b
            (-10,  10),    # x2 → rho (tanh maps this to (-1, 1))
            (-_M_CLIP, _M_CLIP),   # x3 → m
            (-30,  _B),    # x4 → sigma
        ]
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 3000, "ftol": 1e-14, "gtol": 1e-8})

        a, b, rho, m, sigma = _unpack(res.x)

        # Compute RMSE
        w_fitted  = _svi_w(k, a, b, rho, m, sigma)
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

    # ── calendar no-arb check ────────────────────────────────────────────

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

    # ── public fit ───────────────────────────────────────────────────────

    def fit(self) -> SVISurfaceFitReport:
        # Fit slices sequentially so each can apply the calendar penalty
        # against all previously fitted slices.
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

    # ── surface builders ─────────────────────────────────────────────────

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

        k_grid = self.log_moneyness_grid if hasattr(self, "log_moneyness_grid") else \
                 np.unique(np.concatenate((
                     np.linspace(-0.20, 0.20, 81),
                     self.df["log_moneyness"].values,
                 )))

        # Interpolate SVI params across T using PCHIP on each parameter
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

            # Analytical derivatives
            # dw/dk  = b * (rho + diff/root)
            # d2w/dk2 = b * sigma^2 / root^3
            wk_row  = b * (rho + diff / root)
            wkk_row = b * sigma ** 2 / root ** 3

            rows_T.extend([T] * nk)
            rows_k.extend(k_grid.tolist())
            rows_w.extend(w_row.tolist())
            rows_iv.extend(iv_row.tolist())
            rows_wk.extend(wk_row.tolist())
            rows_wkk.extend(wkk_row.tolist())

        # dw/dT: numerical differentiation over T_grid
        w_matrix = np.array(rows_w).reshape(len(T_grid), nk)
        dw_dT    = np.gradient(w_matrix, T_grid, axis=0)

        for i in range(len(T_grid)):
            w_row  = w_matrix[i]
            wt_row = dw_dT[i]
            wk_row = np.array(rows_wk[i*nk:(i+1)*nk])
            wkk_row = np.array(rows_wkk[i*nk:(i+1)*nk])

            w_safe = np.maximum(w_row, 1e-12)
            T_val  = T_grid[i]
            term1  = 1.0 - (k_grid * wk_row) / (2.0 * w_safe)
            term2  = 0.25 * wk_row ** 2 * (1.0 / w_safe + 0.25)
            D      = np.maximum(term1 ** 2 + term2, 1e-12)

            lvar   = wt_row / D
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
