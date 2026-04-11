"""
Post-calibration validators for SSVI surface quality.

Two checks:
  1. SSVI fit quality  — RMSE in vol points, per-expiry breakdown
  2. Gatheral-Jacquier no-arbitrage conditions — analytical check on fitted params
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SSVIFitReport:
    """
    Returned by SSVIValidator.check_fit_quality().
    All vol figures in percentage points (e.g. 1.2 means 1.2 vol pts).
    """
    overall_rmse_vpts:  float                       # global RMSE in vol points
    per_expiry_rmse:    dict[float, float]          # {T: rmse_vpts}
    max_residual_vpts:  float                       # worst single point
    fit_acceptable:     bool                        # True if overall_rmse <= threshold
    gj_violations:      list[str] = field(default_factory=list)
    # SSVI−SVI gap: large gap = SSVI structurally failing = smile non-linear = stress indicator
    ssvi_svi_gap_vpts:  float = np.nan
    smile_stress_flag:  bool  = False   # True when gap >= SMILE_STRESS_THRESHOLD

    def log(self):
        status = "✅ PASS" if self.fit_acceptable else "⚠️  WARN"
        logger.info("[SSVI Fit] %s  RMSE=%.2f vol-pts  MaxRes=%.2f vol-pts",
                    status, self.overall_rmse_vpts, self.max_residual_vpts)
        for T, rmse in self.per_expiry_rmse.items():
            logger.info("  T=%.4f  RMSE=%.2f vol-pts", T, rmse)
        for msg in self.gj_violations:
            logger.warning("  [GJ violation] %s", msg)
        if not np.isnan(self.ssvi_svi_gap_vpts):
            flag = "⚠️  SMILE STRESS" if self.smile_stress_flag else "✅ Normal"
            logger.info("[Smile Stress] SSVI-SVI gap=%.2f vpts  %s",
                        self.ssvi_svi_gap_vpts, flag)


class SSVIValidator:

    RMSE_WARN_THRESHOLD_VPTS:  float = 1.5   # warn if overall RMSE > 1.5 vol points
    # SSVI−SVI gap threshold for smile stress flag.
    # From the multi-date analysis: gap is ~0.7 on normal days, ~1.9 on stress days.
    # A threshold of 1.2 sits between the two regimes.
    SMILE_STRESS_THRESHOLD:    float = 1.2   # vol points

    # ------------------------------------------------------------------ #
    @staticmethod
    def check_fit_quality(
        iv_bs_df:    pd.DataFrame,
        ssvi_params: dict,
        ssvi_fitter,
        svi_rmse_vpts: float = np.nan,   # pass SVI RMSE to compute gap
    ) -> SSVIFitReport:
        """
        Compare fitted SSVI IV against observed BS IV (raw, non-interpolated only).

        Args:
            iv_bs_df:    DataFrame with 'implied_volatility', 'log_moneyness',
                         'time_to_expiry', 'iv_valid_for_ssvi'
            ssvi_params: dict with theta_map, phi, rho
            ssvi_fitter: SSVIConstrainedFitter instance (for ssvi_w helper)

        Returns:
            SSVIFitReport
        """
        from src.pricing_engines.iv_engines.svi_engine.ssvi_helpers import SSVIHelpers

        theta_map  = ssvi_params["theta_map"]
        phi        = ssvi_params["phi"]
        rho        = ssvi_params["rho"]

        # Only validate on raw traded IV points
        if "iv_valid_for_ssvi" in iv_bs_df.columns:
            val_df = iv_bs_df[iv_bs_df["iv_valid_for_ssvi"]].copy()
        else:
            val_df = iv_bs_df[iv_bs_df["implied_volatility"].notna() &
                              (iv_bs_df["implied_volatility"] > 0)].copy()

        if val_df.empty:
            logger.warning("[SSVI Fit] No valid rows for quality check")
            return SSVIFitReport(
                overall_rmse_vpts=np.nan, per_expiry_rmse={},
                max_residual_vpts=np.nan, fit_acceptable=False
            )

        residuals_vpts = []
        per_expiry     = {}

        for T, group in val_df.groupby("time_to_expiry"):
            if T not in theta_map:
                continue

            theta   = theta_map[T]
            k       = group["log_moneyness"].values
            iv_obs  = group["implied_volatility"].values          # in %

            w_model  = SSVIHelpers.ssvi_w(k, theta, phi, rho)
            iv_model = np.sqrt(np.maximum(w_model / max(T, 1e-12), 0)) * 100  # in %

            res_vpts        = iv_model - iv_obs
            rmse_t          = float(np.sqrt(np.mean(res_vpts ** 2)))
            per_expiry[T]   = rmse_t
            residuals_vpts.extend(res_vpts.tolist())

        residuals_arr      = np.array(residuals_vpts)
        overall_rmse       = float(np.sqrt(np.mean(residuals_arr ** 2))) if len(residuals_arr) else np.nan
        max_res            = float(np.max(np.abs(residuals_arr)))        if len(residuals_arr) else np.nan
        fit_ok             = overall_rmse <= SSVIValidator.RMSE_WARN_THRESHOLD_VPTS

        # Gatheral-Jacquier conditions
        gj_violations = SSVIValidator._check_gj_conditions(ssvi_params)

        # SSVI−SVI gap: measures how much non-linearity the smile has.
        # Large gap = SSVI structurally failing = market in a fear/stress regime.
        gap        = (overall_rmse - svi_rmse_vpts
                      if not np.isnan(svi_rmse_vpts) else np.nan)
        stress_flag = (not np.isnan(gap) and
                       gap >= SSVIValidator.SMILE_STRESS_THRESHOLD)

        report = SSVIFitReport(
            overall_rmse_vpts = overall_rmse,
            per_expiry_rmse   = per_expiry,
            max_residual_vpts = max_res,
            fit_acceptable    = fit_ok,
            gj_violations     = gj_violations,
            ssvi_svi_gap_vpts = round(gap, 4) if not np.isnan(gap) else np.nan,
            smile_stress_flag = stress_flag,
        )
        report.log()
        return report

    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_gj_conditions(ssvi_params: dict) -> list[str]:
        """
        Gatheral-Jacquier (2014) no-arbitrage conditions.

        For each expiry with ATM total variance θ:
            (1)  φ·θ·(1+|ρ|) ≤ 4      [wing condition]
            (2)  φ²·(1+|ρ|)   ≤ 4      [butterfly condition]
        """
        theta_map = ssvi_params["theta_map"]
        phi       = ssvi_params["phi"]
        rho       = ssvi_params["rho"]
        abs_rho   = abs(rho)
        violations = []

        for T, theta in theta_map.items():
            wing_lhs      = phi * theta * (1 + abs_rho)
            butterfly_lhs = phi ** 2    * (1 + abs_rho)

            if wing_lhs > 4.0:
                msg = (f"T={T:.4f}: wing violated — φ·θ·(1+|ρ|) = {wing_lhs:.4f} > 4")
                violations.append(msg)
                logger.warning("[GJ] %s", msg)

            if butterfly_lhs > 4.0:
                msg = (f"T={T:.4f}: butterfly violated — φ²·(1+|ρ|) = {butterfly_lhs:.4f} > 4")
                violations.append(msg)
                logger.warning("[GJ] %s", msg)

        if not violations:
            logger.info("[GJ] All Gatheral-Jacquier conditions satisfied")

        return violations