"""
Data quality checks applied to the raw option chain before IV calibration.

Checks applied in sequence:
  1. Vol regime detection   — classify ATM IV into low / medium / high regime
  2. Stale price detection  — flag options with no price or OI movement
  3. Put-call parity forward check — flag strikes with implied forward diverging from futures
  4. Minimum liquidity filter — drop expiry slices with too few valid strikes

Each check adds a boolean column; the final `is_clean` column is the intersection.
Rows where `is_clean = False` are excluded from IV calibration and SSVI fitting
but are kept in the DataFrame for diagnostic visibility.

Vol regime flags
----------------
  low_vol    ATM IV < 12%   — normal BankNifty conditions
  medium_vol ATM IV 12–16%  — elevated, consistent with 1–2% daily moves
  high_vol   ATM IV > 16%   — fear regime, spreads wide, liquidity thin

In high_vol regime the PCP tolerance is widened and the min-OI threshold is
lowered to avoid stripping too many contracts from an already thin market.
The high-vol fallback OI threshold reuses Settings.MIN_OI_STOCK (500) — the
same value used for single-stock options — because both represent thinner,
less-liquid market conditions.
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

    # Minimum number of valid strikes per expiry slice to attempt IV calibration
    MIN_STRIKES_PER_EXPIRY: int   = 5
    # Max fractional deviation of implied forward from futures forward
    # Widened automatically in high_vol regime (see run())
    PCP_FORWARD_TOL:        float = 0.005   # 0.5% — normal regime
    PCP_FORWARD_TOL_HIGH:   float = 0.010   # 1.0% — high_vol regime

    # OI thresholds are sourced from Settings to avoid duplication:
    #   Settings.MIN_OI_INDEX (1000) — default, used for all normal-regime runs
    #   Settings.MIN_OI_STOCK  (500) — high-vol fallback; same value because
    #       thin stress markets resemble single-stock liquidity profiles

    # ------------------------------------------------------------------ #
    @staticmethod
    def run(
        df:       pd.DataFrame,
        prev_df:  pd.DataFrame | None = None,
        min_oi:   int          | None = None,
    ) -> pd.DataFrame:
        """
        Run all quality checks. Returns df with added columns:
            vol_regime        str   — 'low_vol' | 'medium_vol' | 'high_vol'
            is_stale          bool  — price AND OI unchanged vs previous day
            pcp_forward_ok    bool  — implied forward within tolerance of futures
            slice_liquid      bool  — expiry slice has >= MIN_STRIKES_PER_EXPIRY valid rows
            is_clean          bool  — all three checks pass (use for calibration)

        Pass df with an `implied_volatility` column (i.e. after BSFitter) so
        that vol-regime detection is based on real calibrated IVs, not defaults.

        In high_vol regime: PCP tolerance is widened (0.5% → 1.0%) and the
        minimum OI threshold is lowered (1000 → 500) to retain contracts in
        thin markets without sacrificing data integrity.

        Args:
            df:       today's parsed option chain DataFrame (ideally post-BSFitter)
            prev_df:  yesterday's parsed DataFrame (optional; if None, stale skipped)
            min_oi:   override minimum OI (uses Settings.MIN_OI_INDEX if None)
        """
        df = df.copy()

        # ── Step 0: Vol regime ────────────────────────────────────────────
        regime = VolRegime.from_df(df)
        df["vol_regime"] = regime

        # Adaptive thresholds based on regime
        effective_pcp_tol = (DataQualityChecker.PCP_FORWARD_TOL_HIGH
                             if regime == VolRegime.HIGH
                             else DataQualityChecker.PCP_FORWARD_TOL)

        if min_oi is not None:
            effective_min_oi = min_oi
        elif regime == VolRegime.HIGH:
            effective_min_oi = Settings.MIN_OI_STOCK   # relaxed threshold for thin stress market
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
        A contract is stale if BOTH its close price AND open interest are
        identical to the previous day.  This catches NSE carry-forward fills
        where no actual trade occurred.

        If no previous-day data is provided, no rows are marked stale
        (can't know without a reference).
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
        Put-call parity forward check.

        For each (expiry, strike) pair with both a CE and PE:
            F_implied = K + (C - P) * e^(r*T)

        If |F_implied - F_futures| / F_futures > tol,
        both the call and put at that strike are flagged.

        tol is passed from run() and may be widened in high_vol regime.
        Strikes with only one leg (call or put only) are left clean.
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
        """
        Flag all rows in an expiry slice that has fewer than MIN_STRIKES_PER_EXPIRY
        distinct strikes with OI >= min_oi.

        We count per (option_type, expiry) since SSVI fitting is per-slice.
        """
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
