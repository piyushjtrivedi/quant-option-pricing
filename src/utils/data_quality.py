"""
Data quality flags on the option chain (run after BS IV, which the regime needs).

  1. vol_regime      — ATM IV: low_vol < 12% ≤ medium_vol < 16% ≤ high_vol
  2. is_stale        — close and OI unchanged vs the previous day
  3. pcp_forward_ok  — put-call-parity forward within tolerance of futures
  4. slice_liquid    — enough liquid strikes in the (type, expiry) slice

is_clean is the intersection. Flags are informational: they are stored and
charted, but do not currently filter the rows used for SSVI/SVI fitting
(that is iv_valid_for_ssvi, set in BSFitter).

In high_vol the PCP tolerance widens (0.5% → 1.0%); the OI threshold drops to
Settings.MIN_OI_STOCK only when no explicit min_oi is passed.
"""

import logging
import numpy as np
import pandas as pd
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class VolRegime:
    LOW    = "low_vol"      # ATM IV < 12%
    MEDIUM = "medium_vol"   # ATM IV 12–16%
    HIGH   = "high_vol"     # ATM IV > 16%

    # Thresholds in % units (same as implied_volatility column)
    LOW_THRESHOLD:  float = 12.0
    HIGH_THRESHOLD: float = 16.0

    @staticmethod
    def classify(atm_iv_pct: float) -> str:
        if atm_iv_pct >= VolRegime.HIGH_THRESHOLD:
            return VolRegime.HIGH
        elif atm_iv_pct >= VolRegime.LOW_THRESHOLD:
            return VolRegime.MEDIUM
        return VolRegime.LOW

    @staticmethod
    def from_df(df: pd.DataFrame) -> str:
        """Infer regime from the nearest-ATM implied_volatility in the df."""
        if "implied_volatility" not in df.columns or df["implied_volatility"].isna().all():
            return VolRegime.LOW
        valid = df[df["implied_volatility"].notna() & (df["implied_volatility"] > 0)]
        if valid.empty:
            return VolRegime.LOW
        atm_row = valid.loc[valid["log_moneyness"].abs().idxmin()]
        return VolRegime.classify(float(atm_row["implied_volatility"]))


class DataQualityChecker:

    MIN_STRIKES_PER_EXPIRY: int   = 5
    # Max |F_implied − F_futures| / F_futures
    PCP_FORWARD_TOL:        float = 0.005   # normal regime
    PCP_FORWARD_TOL_HIGH:   float = 0.010   # high_vol regime

    # ------------------------------------------------------------------ #
    @staticmethod
    def run(
        df:       pd.DataFrame,
        prev_df:  pd.DataFrame | None = None,
        min_oi:   int          | None = None,
    ) -> pd.DataFrame:
        """
        Add vol_regime, is_stale, pcp_forward_ok, slice_liquid, is_clean.

        Args:
            df:       option chain with implied_volatility (i.e. after BSFitter)
            prev_df:  previous day's chain; if None the stale check is skipped
            min_oi:   explicit OI threshold; if None, regime-dependent
        """
        df = df.copy()

        regime = VolRegime.from_df(df)
        df["vol_regime"] = regime

        effective_pcp_tol = (DataQualityChecker.PCP_FORWARD_TOL_HIGH
                             if regime == VolRegime.HIGH
                             else DataQualityChecker.PCP_FORWARD_TOL)

        if min_oi is not None:
            effective_min_oi = min_oi
        elif regime == VolRegime.HIGH:
            effective_min_oi = Settings.MIN_OI_STOCK
        else:
            effective_min_oi = Settings.MIN_OI_INDEX

        logger.info("[DataQuality] Vol regime: %s  |  pcp_tol=%.1f%%  min_oi=%d",
                    regime, effective_pcp_tol * 100, effective_min_oi)

        df["is_stale"]       = DataQualityChecker._stale_check(df, prev_df)
        df["pcp_forward_ok"] = DataQualityChecker._pcp_check(df, tol=effective_pcp_tol)
        df["slice_liquid"]   = DataQualityChecker._liquidity_check(df, min_oi=effective_min_oi)
        df["is_clean"]       = ~df["is_stale"] & df["pcp_forward_ok"] & df["slice_liquid"]

        n_total = len(df)
        n_stale = int(df["is_stale"].sum())
        n_pcp   = int((~df["pcp_forward_ok"]).sum())
        n_illiq = int((~df["slice_liquid"]).sum())
        n_clean = int(df["is_clean"].sum())

        logger.info(
            "[DataQuality] %d total | %d stale | %d pcp-fail | %d illiquid-slice | %d clean",
            n_total, n_stale, n_pcp, n_illiq, n_clean,
        )
        if n_clean < 20:
            logger.warning(
                "[DataQuality] Only %d clean rows — calibration quality may be poor", n_clean
            )
        return df

    # ------------------------------------------------------------------ #
    @staticmethod
    def _stale_check(df: pd.DataFrame, prev_df: pd.DataFrame | None) -> pd.Series:
        """
        Stale = close AND OI identical to the previous day (NSE carry-forward
        with no trade). No prev_df → nothing marked stale.
        """
        stale = pd.Series(False, index=df.index)

        if prev_df is None:
            logger.debug("[DataQuality] No previous day data — stale check skipped")
            return stale

        key_cols    = ["expiry", "strike", "option_type"]
        prev_subset = prev_df[key_cols + ["close", "open_interest"]].rename(
            columns={"close": "prev_close", "open_interest": "prev_oi"}
        )

        merged = df.merge(prev_subset, on=key_cols, how="left")

        price_same = np.isclose(merged["close"],        merged["prev_close"], rtol=0, atol=0.01)
        oi_same    = np.isclose(merged["open_interest"], merged["prev_oi"],   rtol=0, atol=0)

        stale_mask = price_same & oi_same & merged["prev_close"].notna()
        stale      = pd.Series(stale_mask.values, index=df.index)

        n = stale.sum()
        if n > 0:
            logger.info("[DataQuality] %d stale contracts detected (price+OI unchanged)", n)

        return stale

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pcp_check(df: pd.DataFrame, tol: float = None) -> pd.Series:
        """
        Put-call parity: F_implied = K + (C − P)·e^{rT} per (expiry, strike).
        If |F_implied − F_futures| / F_futures > tol, flag both legs.
        Strikes with only one leg are left clean.
        """
        tol = tol if tol is not None else DataQualityChecker.PCP_FORWARD_TOL

        pcp_ok = pd.Series(True, index=df.index)

        has_rate = "interest_rate" in df.columns

        calls = df[df["option_type"] == "CE"][
            ["expiry", "strike", "close", "time_to_expiry",
             "futures_close"] + (["interest_rate"] if has_rate else [])
        ].copy()
        puts  = df[df["option_type"] == "PE"][
            ["expiry", "strike", "close"]
        ].copy()

        paired = calls.merge(
            puts.rename(columns={"close": "put_close"}),
            on=["expiry", "strike"],
            how="inner",
        )

        if paired.empty:
            return pcp_ok

        r = paired["interest_rate"].values if has_rate else 0.0
        T = paired["time_to_expiry"].values
        F = paired["futures_close"].values
        K = paired["strike"].values
        C = paired["close"].values
        P = paired["put_close"].values

        F_implied = K + (C - P) * np.exp(r * T)
        deviation = np.abs(F_implied - F) / np.maximum(F, 1e-6)

        bad_mask = deviation > tol

        if bad_mask.any():
            bad_strikes  = paired.loc[bad_mask, ["expiry", "strike"]].drop_duplicates()
            bad_key_set  = set(zip(bad_strikes["expiry"], bad_strikes["strike"]))
            df_key       = list(zip(df["expiry"], df["strike"]))
            flag_mask    = pd.Series([k in bad_key_set for k in df_key], index=df.index)
            pcp_ok[flag_mask] = False

            logger.info(
                "[DataQuality] %d strikes failed PCP forward check (tol=%.1f%%)",
                bad_mask.sum(), tol * 100,
            )

        return pcp_ok

    # ------------------------------------------------------------------ #
    @staticmethod
    def _liquidity_check(df: pd.DataFrame, min_oi: int = None) -> pd.Series:
        """Flag every row of an (option_type, expiry) slice with < MIN_STRIKES_PER_EXPIRY strikes at OI ≥ min_oi."""
        effective_oi = min_oi if min_oi is not None else Settings.MIN_OI_INDEX
        min_strikes  = DataQualityChecker.MIN_STRIKES_PER_EXPIRY
        liquid       = pd.Series(True, index=df.index)

        for (opt, exp), group in df.groupby(["option_type", "expiry"]):
            n_liquid = (group["open_interest"] >= effective_oi).sum()
            if n_liquid < min_strikes:
                liquid[group.index] = False
                logger.info(
                    "[DataQuality] Expiry %s %s: only %d liquid strikes — slice excluded",
                    exp, opt, n_liquid,
                )

        return liquid
