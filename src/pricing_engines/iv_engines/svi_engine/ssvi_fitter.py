import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from scipy.interpolate import PchipInterpolator
from src.pricing_engines.iv_engines.svi_engine.ssvi_helpers import SSVIHelpers

class SSVIConstrainedFitter:

    def __init__(self, iv_df: pd.DataFrame, log_moneyness_grid=None, phi_min_override: float = None):

        self.ssvi_params = {}

        # iv_valid_for_ssvi (set in BSFitter) is the single source of truth for fit rows
        if "iv_valid_for_ssvi" in iv_df.columns:
            df = iv_df[iv_df["iv_valid_for_ssvi"]].copy()
        else:
            df = iv_df[
                iv_df["implied_volatility"].notna() & (iv_df["implied_volatility"] > 0)
            ].copy()

        self.df = df
        self.time_to_expiry = sorted(self.df["time_to_expiry"].unique().tolist())
        # Moneyness Grid
        log_moneyness = np.asarray(self.df["log_moneyness"].values)
        log_moneyness_min, log_moneyness_max = np.min(log_moneyness) - 0.5, np.max(log_moneyness) + 0.5
        log_moneyness_grid = np.linspace(log_moneyness_min, log_moneyness_max, 61) if log_moneyness_grid is None else np.asarray(log_moneyness_grid)
        self.log_moneyness_grid = np.unique(np.concatenate((log_moneyness_grid,log_moneyness)))
        # Data Map with underlying info corresponding to each expiry
        self.ips_by_expiry = {}
        for T in self.time_to_expiry:
            group_by_expiry = self.df[self.df["time_to_expiry"] == T]
            if len(group_by_expiry) == 0: continue
            S = float(group_by_expiry["underlying_value"].iloc[0])
            r = float(group_by_expiry["interest_rate"].iloc[0])
            F = float(group_by_expiry["futures_close"].iloc[0])
            q = float(group_by_expiry["dividend"].iloc[0])
            self.ips_by_expiry[T] = {"S":S, "r":r, "F" : F, "q" : q}

        # Floor on phi: stops the optimiser collapsing to a flat smile (phi → 0)
        self.phi_min = phi_min_override if phi_min_override is not None else 0.1

    def _unpack(self, x):
        n       = len(self.time_to_expiry)
        deltas  = x[:n]
        raw_phi = x[n]
        rho_un  = x[n + 1]
        theta   = SSVIHelpers.pack_to_theta(deltas)
        phi     = SSVIHelpers.softplus_np(raw_phi) + self.phi_min
        rho     = math.tanh(rho_un)
        return theta, phi, rho

    def _gj_penalty(self, theta, phi, rho) -> float:
        """
        Soft quadratic penalty for the Gatheral-Jacquier (2014, Thm 4.2) conditions,
        per expiry with ATM total variance θ:
            wing:       θ·φ·(1+|ρ|)  < 4
            butterfly:  θ·φ²·(1+|ρ|) ≤ 4
        """
        abs_rho = abs(rho)
        penalty = 0.0
        for th in theta:
            wing      = th * phi      * (1.0 + abs_rho)
            butterfly = th * phi ** 2 * (1.0 + abs_rho)
            if wing > 4.0:
                penalty += 1e4 * (wing - 4.0) ** 2
            if butterfly > 4.0:
                penalty += 1e4 * (butterfly - 4.0) ** 2
        return penalty

    def _feasible(self, theta, phi, rho):
        for i, T in enumerate(self.time_to_expiry):
            info   = self.ips_by_expiry[T]
            w_grid = SSVIHelpers.ssvi_w(self.log_moneyness_grid, theta[i], phi, rho)
            if not SSVIHelpers.positive_density_on_grid(
                self.log_moneyness_grid, w_grid, T,
                info['F'], info['S'], info['r'], info['q'], tol=-1e-12
            ):
                return False
        return True

    # Raw w-space MSE is O(1e-7) with gradients O(1e-9), below L-BFGS-B's
    # default gtol=1e-5, so the optimiser would stop at x0. Scaling by 1e6
    # lifts gradients to O(1e-3) without changing the minimiser.
    W_SCALE: float = 1e6

    def objective(self, x):
        """
        Scaled total-variance MSE, averaged per expiry (not pooled), so the
        front expiry's many strikes don't dominate the fit. Infeasible
        (negative-density) surfaces return 1e20.
        """
        theta, phi, rho = self._unpack(x)
        gj_pen = self._gj_penalty(theta, phi, rho)
        if not self._feasible(theta, phi, rho):
            return 1e20 + gj_pen

        ts    = self.df["time_to_expiry"].values
        ks    = self.df["log_moneyness"].values
        w_obs = self.df["bs_implied_variance"].values   # sigma^2 * T

        per_expiry_mse = []
        for i, T in enumerate(self.time_to_expiry):
            w_model = SSVIHelpers.ssvi_w(ks[ts == T], theta[i], phi, rho)
            w_obs_T = w_obs[ts == T]
            mask    = np.isfinite(w_model) & np.isfinite(w_obs_T)
            if mask.sum() == 0:
                continue
            err = (w_model[mask] - w_obs_T[mask]) ** 2
            per_expiry_mse.append(float(err.mean()))

        if not per_expiry_mse:
            return 1e20

        fit_cost = float(np.mean(per_expiry_mse)) * self.W_SCALE
        return fit_cost + gj_pen

    def fit(self, x0=None, verbose=True):
        if x0 is None:
            theta0 = []
            for T in self.time_to_expiry:
                g   = self.df[self.df["time_to_expiry"] == T]
                idx = np.argmin(np.abs(g["log_moneyness"].values))
                theta0.append(max(1e-6, float(g["bs_implied_variance"].values[idx])))
            deltas = []
            prev = 0.0
            for t in theta0:
                diff = max(1e-12, t - prev)
                deltas.append(math.log(max(1e-12, math.expm1(diff))))
                prev += SSVIHelpers.softplus_np(deltas[-1])
            raw_phi = math.log(math.expm1(0.5))
            rho_un  = math.atanh(-0.3)
            x0 = np.concatenate([np.array(deltas), np.array([raw_phi, rho_un])])

        res = minimize(self.objective, x0, method='L-BFGS-B',
                       options={'maxiter': 2000, 'ftol': 1e-12})
        theta, phi, rho = self._unpack(res.x)
        theta_map = {T: theta[i] for i, T in enumerate(self.time_to_expiry)}

        if verbose:
            import logging
            logging.getLogger(__name__).info(
                "[SSVI] success=%s  obj=%.6f  phi=%.4f  rho=%.4f",
                res.success, res.fun, float(phi), float(rho)
            )

        self.ssvi_params = {
            "theta_map": theta_map,
            "phi":       float(phi),
            "rho":       float(rho),
            "opt":       res,
        }
        return self.ssvi_params

    def build_ssviNdupire_vol_from_theta_map(self, sigma_floor = 1e-4,smooth_sigma = (0.6, 0.6)):
        """Build the SSVI surface on a (T, k) grid with analytical Dupire local vol."""

        k_grid = self.log_moneyness_grid
        expiries = self.time_to_expiry
        T_grid = np.unique(np.concatenate((np.linspace(np.min(expiries), np.max(expiries), 20),expiries)))
        nk, nT = len(k_grid), len(T_grid)

        w = np.zeros((nT,nk), dtype=float)
        iv = np.zeros_like(w)
        w_k = np.zeros_like(w)
        w_kk = np.zeros_like(w)
        w_t = np.zeros_like(w)

        theta_map = self.ssvi_params["theta_map"]
        phi       = self.ssvi_params["phi"]
        rho       = self.ssvi_params["rho"]

        Ts     = np.array(sorted(theta_map.keys()), dtype=float)
        thetas = np.array([theta_map[T] for T in Ts], dtype=float)
        thetas_interpolator     = PchipInterpolator(Ts, thetas, extrapolate=True)
        thetas_dot_interpolator = thetas_interpolator.derivative()

        theta_T     = thetas_interpolator(T_grid)
        theta_dot_T = thetas_dot_interpolator(T_grid)

        for i in range(nT):
            theta     = float(theta_T[i])
            theta_dot = float(theta_dot_T[i])
            w[i, :], iv[i, :], w_k[i, :], w_kk[i, :], w_t[i, :] = \
                SSVIHelpers.local_iv_from_ssvi_params(k_grid, T_grid[i], theta, theta_dot, phi, rho)

        # Gatheral (2006) eq. 1.10: σ²_loc = w_T / D,
        # D = (1 - k*w_k/(2w))^2 - (w_k^2/4)*(1/w + 1/4) + w_kk/2
        # D <= 0 is butterfly arb (local vol undefined) → sigma_floor.
        w_safe = np.maximum(w, 1e-12)
        term1 = 1.0 - ((k_grid[np.newaxis, :] * w_k) / (2.0*w_safe))
        term1_sq = term1*term1
        term2 = 0.25*(w_k*w_k)*(1.0 / w_safe + 0.25)
        D = term1_sq - term2 + 0.5*w_kk

        local_var = np.where(D > 1e-12, w_t / np.maximum(D, 1e-12), np.nan)
        local_var = np.where(np.isfinite(local_var) & (local_var > 0.0), local_var, sigma_floor**2)
        local_var = np.maximum(local_var, sigma_floor**2)

        if smooth_sigma is not None:
            local_var = gaussian_filter(local_var, sigma=smooth_sigma, mode='nearest')

        local_var = np.maximum(local_var, sigma_floor**2)
        local_vol = np.sqrt(local_var)

        iv_df = pd.DataFrame({
                'time_to_expiry': np.repeat(T_grid, nk),
                'log_moneyness': np.tile(k_grid, nT),
                'implied_volatility_svi': np.concatenate(100*iv),
                'implied_variance_svi' : np.concatenate(w),
                'dw_dT_svi' : np.concatenate(w_t),
                'dw_dk_svi' : np.concatenate(w_k),
                'd2w_dk2_svi' : np.concatenate(w_kk),
                'dupire_local_volatility' : np.concatenate(100*local_vol),
                'dupire_local_variance' : np.concatenate(local_var)
            })
        return iv_df