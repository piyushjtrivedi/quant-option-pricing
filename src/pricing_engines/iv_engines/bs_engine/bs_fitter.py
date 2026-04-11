import logging
import math
import numpy as np
import pandas as pd
from src.pricing_engines.bs_pricer import BlackScholes
from src.pricing_engines.iv_engines.bs_engine.iv_bs import BSIV
from scipy.interpolate import PchipInterpolator

logger = logging.getLogger(__name__)

pd.set_option('display.float_format', '{:.6f}'.format)

# Module-level vectorised wrappers — created once, reused every call.
# np.vectorize avoids pandas .apply() overhead while keeping the scalar
# BlackScholes implementation unchanged (it has scalar if-guards).
_bs_price_vec = np.vectorize(BlackScholes.bs_price)
_bs_vega_vec  = np.vectorize(BlackScholes.bs_vega)


class BSFitter:

    @staticmethod
    def fit(df: pd.DataFrame, nss_fitted_curve, dividend_fitter):
        """
        Full BS IV calibration pipeline:
          1. Compute r(T) and q(T) vectorised per unique expiry
          2. Compute raw BS IV per row (Newton+bisection) → NaN for failures
          3. Interpolate NaN IVs within each (option_type, expiry) slice
             using PCHIP on the valid neighbours in log-moneyness space
          4. Mark iv_interpolated flag so downstream knows which are real vs filled
          5. Derive bs_vega, bs_implied_variance on all valid rows
          6. Add static arbitrage diagnostics (butterfly + calendar)
        """
        oc_df = df.copy()

        # ------------------------------------------------------------------
        # 1. Vectorised r(T) and q(T) — one call per unique expiry
        # ------------------------------------------------------------------
        unique_expiries = oc_df["time_to_expiry"].unique()
        r_map = {T: float(nss_fitted_curve(T)) / 100 for T in unique_expiries}
        q_map = {T: float(dividend_fitter.forecast(T))  for T in unique_expiries}

        oc_df["interest_rate"] = oc_df["time_to_expiry"].map(r_map)
        oc_df["dividend"]      = oc_df["time_to_expiry"].map(q_map)

        # ------------------------------------------------------------------
        # 2. Raw BS IV — array-based loop (avoids iterrows overhead)
        # ------------------------------------------------------------------
        prices  = oc_df["close"].values
        S_arr   = oc_df["underlying_value"].values
        K_arr   = oc_df["strike"].values
        r_arr   = oc_df["interest_rate"].values
        q_arr   = oc_df["dividend"].values
        T_arr   = oc_df["time_to_expiry"].values
        flags   = np.where(oc_df["option_type"].values == "CE", "c", "p")
        htv     = (oc_df["close"].values > oc_df["pay_off"].values)

        raw_iv = np.full(len(oc_df), np.nan)
        for i in range(len(oc_df)):
            if not htv[i]:
                continue
            iv = BSIV.compute(
                prices[i], S_arr[i], K_arr[i],
                r_arr[i], q_arr[i], T_arr[i], flags[i],
            )
            raw_iv[i] = iv * 100 if (iv is not None and not np.isnan(iv)) else np.nan

        oc_df["implied_volatility"] = raw_iv
        oc_df["iv_interpolated"]    = False

        # ------------------------------------------------------------------
        # 3. Interpolate NaN IVs within each (option_type, expiry) slice
        # ------------------------------------------------------------------
        oc_df = BSFitter._interpolate_iv_gaps(oc_df)

        # ------------------------------------------------------------------
        # 4. iv_valid_for_ssvi — raw, non-interpolated only
        # ------------------------------------------------------------------
        oc_df["iv_valid_for_ssvi"] = (
            ~oc_df["iv_interpolated"]
            & oc_df["implied_volatility"].notna()
            & (oc_df["implied_volatility"] > 0)
        )

        # ------------------------------------------------------------------
        # 5. Greeks and implied variance — vectorised via np.vectorize
        # ------------------------------------------------------------------
        valid      = oc_df["implied_volatility"].notna().values & (oc_df["implied_volatility"].values > 0)
        sigma_arr  = np.where(valid, oc_df["implied_volatility"].values / 100, np.nan)
        flag_arr   = np.where(oc_df["option_type"].values == "CE", "c", "p")

        bs_prices = np.where(
            valid,
            _bs_price_vec(S_arr, K_arr, r_arr, q_arr, T_arr, sigma_arr, flag_arr),
            oc_df["pay_off"].values,
        )
        bs_vegas = np.where(
            valid,
            _bs_vega_vec(S_arr, K_arr, r_arr, q_arr, T_arr, sigma_arr),
            0.0,
        )
        bs_var = np.where(
            valid,
            np.square(np.where(valid, oc_df["implied_volatility"].values / 100, 0.0)) * T_arr,
            0.0,
        )

        oc_df["bs_price"]           = bs_prices
        oc_df["bs_vega"]            = bs_vegas
        oc_df["bs_implied_variance"] = bs_var

        # ------------------------------------------------------------------
        # 6. OTM-only + moneyness cap for SSVI fitting
        # ------------------------------------------------------------------
        is_otm_call = (oc_df["option_type"] == "CE") & (oc_df["log_moneyness"] >= 0)
        is_otm_put  = (oc_df["option_type"] == "PE") & (oc_df["log_moneyness"] <= 0)
        K_MAX       = 0.12
        in_zone     = oc_df["log_moneyness"].abs() <= K_MAX

        oc_df["iv_valid_for_ssvi"] = (
            oc_df["iv_valid_for_ssvi"]
            & (is_otm_call | is_otm_put)
            & in_zone
        )

        # ------------------------------------------------------------------
        # 7. Static arbitrage diagnostics
        # ------------------------------------------------------------------
        return BSFitter.add_static_arbitrage_diagnostics(oc_df)

    # ------------------------------------------------------------------
    @staticmethod
    def interpolate_bs_iv_onto_svi_grid(
        iv_bs_df: pd.DataFrame, iv_svi_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Interpolate BS IV from traded strikes onto the denser SSVI k-grid.
        Uses PCHIP per (option_type, expiry) slice. No extrapolation.
        """
        iv_svi_df = iv_svi_df.copy()
        iv_svi_df["implied_volatility_bs_grid"] = np.nan
        n_filled = 0

        for (opt_type, T), group in iv_bs_df.groupby(["option_type", "time_to_expiry"]):
            g = group.dropna(subset=["implied_volatility"]).sort_values("log_moneyness")
            if len(g) < 3:
                continue
            pchip = PchipInterpolator(
                g["log_moneyness"].values,
                g["implied_volatility"].values,
                extrapolate=False,
            )
            svi_slice = iv_svi_df[np.isclose(iv_svi_df["time_to_expiry"], T)]
            k_svi     = svi_slice["log_moneyness"].values
            iv_interp = pchip(k_svi)
            valid     = ~np.isnan(iv_interp)
            iv_svi_df.loc[svi_slice.index[valid], "implied_volatility_bs_grid"] = iv_interp[valid]
            n_filled += int(valid.sum())

        logger.info(
            "[BSGrid] Interpolated BS IV onto SSVI grid: %d / %d points filled",
            n_filled, len(iv_svi_df),
        )
        return iv_svi_df

    # ------------------------------------------------------------------
    @staticmethod
    def _interpolate_iv_gaps(df: pd.DataFrame) -> pd.DataFrame:
        """
        For each (option_type, time_to_expiry) slice:
          - Sort by log_moneyness
          - Fit PCHIP on valid IV points in total-variance space
          - Fill NaN IVs that fall WITHIN the valid smile range (no extrapolation)
          - Tag filled rows with iv_interpolated = True
        """
        df = df.copy()

        for (opt_type, T), group in df.groupby(["option_type", "time_to_expiry"]):
            group = group.sort_values("log_moneyness")
            idx   = group.index

            k   = group["log_moneyness"].values
            iv  = group["implied_volatility"].values
            w   = np.where(~np.isnan(iv), np.square(iv / 100) * T, np.nan)

            valid_mask = ~np.isnan(w)
            if valid_mask.sum() < 3:
                continue

            k_valid = k[valid_mask]
            w_valid = w[valid_mask]

            pchip   = PchipInterpolator(k_valid, w_valid, extrapolate=False)
            k_min, k_max = k_valid.min(), k_valid.max()

            nan_mask      = np.isnan(w)
            in_range_mask = (k >= k_min) & (k <= k_max)
            fill_mask     = nan_mask & in_range_mask

            if fill_mask.sum() == 0:
                continue

            w_filled  = pchip(k[fill_mask])
            w_filled  = np.where(w_filled > 1e-8, w_filled, np.nan)
            iv_filled = np.where(
                ~np.isnan(w_filled),
                np.sqrt(np.maximum(w_filled / max(T, 1e-12), 0)) * 100,
                np.nan,
            )

            fill_indices = idx[fill_mask]
            df.loc[fill_indices, "implied_volatility"] = iv_filled
            df.loc[fill_indices, "iv_interpolated"]    = True

        return df

    @staticmethod
    def add_static_arbitrage_diagnostics(df: pd.DataFrame):

        df = df.copy()
        df = df.sort_values(["option_type", "time_to_expiry", "log_moneyness"])

        new_cols = ["dw_dk_bs", "d2w_dk2_bs", "dw_dT_bs"]
        for col in new_cols:
            df[col] = np.nan

        # BUTTERFLY ARBITRAGE
        for opt_type, df_type in df.groupby("option_type"):
            for maturity, group in df_type.groupby("time_to_expiry"):
                if len(group) < 3:
                    continue
                group = group.sort_values("log_moneyness")
                idx   = group.index
                k     = group["log_moneyness"].values
                w_bs  = group["bs_implied_variance"].values
                dw_dk_bs   = np.gradient(w_bs, k)
                d2w_dk2_bs = np.gradient(dw_dk_bs, k)
                df.loc[idx, "dw_dk_bs"]   = dw_dk_bs
                df.loc[idx, "d2w_dk2_bs"] = d2w_dk2_bs

        # CALENDAR ARBITRAGE
        for opt_type, df_type in df.groupby("option_type"):
            maturities = sorted(df_type["time_to_expiry"].unique())
            if len(maturities) < 2:
                continue
            k_min_list = [df_type[df_type["time_to_expiry"] == T]["log_moneyness"].min() for T in maturities]
            k_max_list = [df_type[df_type["time_to_expiry"] == T]["log_moneyness"].max() for T in maturities]
            k_min = max(k_min_list)
            k_max = min(k_max_list)
            if k_max <= k_min:
                continue
            k_grid   = np.linspace(k_min, k_max, 40)
            w_matrix = []
            for T in maturities:
                sub = df_type[df_type["time_to_expiry"] == T].sort_values("log_moneyness")
                if len(sub) < 3:
                    continue
                interp   = PchipInterpolator(sub["log_moneyness"].values,
                                             sub["bs_implied_variance"].values,
                                             extrapolate=False)
                w_matrix.append(interp(k_grid))
            w_matrix = np.array(w_matrix)
            if w_matrix.shape[0] < 2:
                continue
            T_array = np.array(maturities[:w_matrix.shape[0]])
            dw_dT   = np.gradient(w_matrix, T_array, axis=0)
            for i, T in enumerate(T_array):
                sub = df_type[df_type["time_to_expiry"] == T]
                for idx, row in sub.iterrows():
                    k_val = row["log_moneyness"]
                    if k_val < k_min or k_val > k_max:
                        continue
                    nearest = np.argmin(np.abs(k_grid - k_val))
                    df.loc[idx, "dw_dT_bs"] = dw_dT[i, nearest]

        return df
