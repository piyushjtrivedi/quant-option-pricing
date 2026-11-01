"""
Main calibration pipeline.

Usage:
    python -m src.main_pipeline.run_calibration 2026-02-26
    python -m src.main_pipeline.run_calibration 2026-02-26 NIFTY ID
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.data_fetch.bhavcopy_client import BhavcopyClient
from src.data_fetch.extractor import BhavcopyExtractor
from src.data_fetch.spot_rate_client import SpotRateClient
from src.parsers.bhavcopy_parser import BhavcopyParser
from src.parsers.interest_rates_parser import InterestRateParser
from src.pricing_engines.nss_curve_fitting import NSSCurveFitter
from src.pricing_engines.dividend_fitting import DividendFitter
from src.pricing_engines.iv_engines.bs_engine.bs_fitter import BSFitter
from src.pricing_engines.iv_engines.svi_engine.ssvi_fitter import SSVIConstrainedFitter
from src.pricing_engines.iv_engines.svi_engine.svi_per_expiry_fitter import SVIPerExpiryFitter
from src.utils.data_quality import DataQualityChecker
from src.utils.calibration_validator import SSVIValidator
from src.utils.storage import ParquetSaver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _resolve_paths(date_str: str, symbol: str) -> dict:
    """Resolve file paths for a given date from Settings.DATA_DIR."""

    dt   = datetime.strptime(date_str, "%Y-%m-%d")
    base = Path(Settings.DATA_DIR)

    bhavcopy_candidates = [
        base / "raw" / f"BhavCopy_NSE_FO_0_0_0_{dt.strftime('%Y%m%d')}_F_0000.csv",
        base / "raw" / f"fo{dt.strftime('%d%b%Y').upper()}.csv",
    ]
    bhavcopy = next((p for p in bhavcopy_candidates if p.exists()), None)

    if not bhavcopy:
        logger.error(
            "[Paths]  Pre-existing Bhavcopy not found for %s. Expected in %s/raw/",
            date_str, Settings.DATA_DIR,
        )
        bhav_client    = BhavcopyClient()
        bhav_extractor = BhavcopyExtractor()
        zip_path = bhav_client.download(dt)
        bhavcopy = bhav_extractor.extract(zip_path)

    # Ensure the date is in mibor.csv (local file, then RBI Excel)
    SpotRateClient().get_rates_for_date(dt)
    single_mibor = base / "mibor" / "mibor.csv"
    mibor = single_mibor if single_mibor.exists() else None

    return {"bhavcopy": bhavcopy, "mibor": mibor}


def calibrate(
    date_str:   str,
    symbol:     str = None,
    instrument: str = None,
) -> str | None:
    """
    Run the full EOD calibration pipeline.

    Args:
        date_str:   Trade date in YYYY-MM-DD format
        symbol:     NSE ticker (defaults to Settings.DEFAULT_SYMBOL)
        instrument: 'ID' for index, 'ST' for stock

    Returns:
        Path to saved parquet, or None on failure.
    """
    symbol     = symbol     or Settings.DEFAULT_SYMBOL
    instrument = instrument or Settings.DEFAULT_INSTRUMENT_TYPE

    logger.info("=" * 60)
    logger.info("[Calibrate] Starting pipeline for %s  symbol=%s  instrument=%s",
                date_str, symbol, instrument)

    # ── 1. Resolve paths ──────────────────────────────────────────────────────
    paths = _resolve_paths(date_str, symbol)

    if not paths["bhavcopy"]:
        logger.error("[Paths] Bhavcopy not found for %s. Expected in %s/raw/",
                     date_str, Settings.DATA_DIR)
        return None
    if not paths["mibor"]:
        logger.error("[Paths] MIBOR rates unavailable for %s — all sources exhausted.",
                     date_str)
        return None

    logger.info("[Paths] Bhavcopy : %s", paths["bhavcopy"])
    logger.info("[Paths] MIBOR    : %s", paths["mibor"])

    # ── 2. Parse market data (parser caches the CSV, so it is read once) ─────
    bhav_parser = BhavcopyParser(symbol=symbol, instrument_type=instrument)
    oc_df       = bhav_parser.parse_to_dataframe(str(paths["bhavcopy"]))
    ir_df       = InterestRateParser.parse_to_dataframe(str(paths["mibor"]), date_str)
    futures_df  = bhav_parser.parse_to_futures_df(str(paths["bhavcopy"]))

    logger.info("[Parse] %d option contracts | %d futures rows | %d rate tenors",
                len(oc_df), len(futures_df), len(ir_df))

    # ── 3. Rates and dividends ────────────────────────────────────────────────
    nss_fitted_curve = NSSCurveFitter.fit_curve(ir_df["years"], ir_df["yield"])

    for i, row in futures_df.iterrows():
        ir = nss_fitted_curve(row.time_to_expiry) / 100
        futures_df.at[i, "dividend"] = ir - row.cost_of_carry

    dividend_fitter = DividendFitter(futures_df["time_to_expiry"], futures_df["dividend"])
    logger.info("[Rates] NSS curve fitted | %d dividend tenors", len(futures_df))

    # ── 4. BS IV calibration — before data quality, which needs IVs for the vol regime
    logger.info("[BS IV] Starting IV calibration ...")
    iv_bs_df = BSFitter.fit(oc_df, nss_fitted_curve, dividend_fitter)
    n_raw_iv = int(iv_bs_df.get("iv_valid_for_ssvi", iv_bs_df["implied_volatility"].notna()).sum())
    n_interp = int(iv_bs_df.get("iv_interpolated",   pd.Series([False])).sum())
    logger.info("[BS IV] %d raw IVs | %d interpolated gap-fills", n_raw_iv, n_interp)

    # ── 5. Data quality flags (regime-adaptive thresholds) ────────────────────
    iv_bs_df = DataQualityChecker.run(iv_bs_df, prev_df=None,
                                      min_oi=Settings.min_oi(instrument))
    vol_regime = iv_bs_df["vol_regime"].iloc[0] if "vol_regime" in iv_bs_df.columns else "low_vol"
    n_clean    = int(iv_bs_df["is_clean"].sum())
    logger.info("[Regime] Vol regime: %s", vol_regime)
    logger.info("[Quality] %d / %d contracts pass all quality checks", n_clean, len(iv_bs_df))
    if n_clean < 20:
        logger.warning("[Quality] Very few clean contracts — check data source")

    # ── 6. Surface fitting ────────────────────────────────────────────────────
    logger.info("[SSVI] Fitting surface ...")
    ssvi_fitter = SSVIConstrainedFitter(iv_bs_df)
    ssvi_params = ssvi_fitter.fit(verbose=False)
    theta_fmt   = {f"T={k:.3f}": f"{v:.4f}" for k, v in ssvi_params["theta_map"].items()}
    logger.info("[SSVI] phi=%.4f  rho=%.4f  theta_map=%s",
                ssvi_params["phi"], ssvi_params["rho"], theta_fmt)

    logger.info("[Dupire] Building SSVI local vol surface ...")
    iv_svi_df = ssvi_fitter.build_ssviNdupire_vol_from_theta_map()

    logger.info("[SVI] Fitting per-slice SVI surface ...")
    svi_fitter = SVIPerExpiryFitter(iv_bs_df)
    svi_report = svi_fitter.fit()
    logger.info("[SVI] Overall RMSE=%.2f vpts  calendar_violations=%d",
                svi_report.overall_rmse_vpts, len(svi_report.calendar_violations))

    for s in svi_report.slice_fits:
        logger.info(
            "[SVI slice] T=%.4f  a=%.6f  b=%.4f  rho=%.4f  m=%.4f  sigma=%.4f  RMSE=%.2f vpts",
            s.T, s.a, s.b, s.rho, s.m, s.sigma, s.rmse_vpts,
        )

    iv_svi_df_svi = svi_fitter.build_surface_df()

    # Merge SVI per-expiry columns with SSVI df for side-by-side comparison
    svi_rename = {
        "implied_volatility_svi":  "implied_volatility_svi_pe",
        "implied_variance_svi":    "implied_variance_svi_pe",
        "dw_dk_svi":               "dw_dk_svi_pe",
        "d2w_dk2_svi":             "d2w_dk2_svi_pe",
        "dw_dT_svi":               "dw_dT_svi_pe",
        "dupire_local_volatility": "dupire_local_volatility_pe",
        "dupire_local_variance":   "dupire_local_variance_pe",
    }
    iv_svi_df_svi = iv_svi_df_svi.rename(columns=svi_rename)

    iv_svi_df = iv_svi_df.merge(
        iv_svi_df_svi[["time_to_expiry", "log_moneyness"] + list(svi_rename.values())],
        on=["time_to_expiry", "log_moneyness"], how="left",
    )

    # ── 7. Back-compute strike from log-moneyness: K = S·e^k·e^{(r−q)T} ───────
    underlying_value = float(iv_bs_df["underlying_value"].iloc[0])
    T_vals    = iv_svi_df["time_to_expiry"].values
    k_vals    = iv_svi_df["log_moneyness"].values
    r_vals    = nss_fitted_curve(T_vals) / 100
    q_vals    = dividend_fitter.forecast(T_vals)
    strikes   = np.exp(k_vals) * underlying_value * np.exp((r_vals - q_vals) * T_vals)
    iv_svi_df["strike"] = np.round(strikes).astype(int)

    # ── 8. Merge market data onto grid points that match a traded contract ────
    iv_svi_df = iv_svi_df.merge(
        iv_bs_df, on=["time_to_expiry", "log_moneyness", "strike"], how="left"
    )

    # ── 9. Validate SSVI fit quality ──────────────────────────────────────────
    logger.info("[Validate] Running SSVI quality checks ...")
    report = SSVIValidator.check_fit_quality(
        iv_bs_df, ssvi_params,
        svi_rmse_vpts=svi_report.overall_rmse_vpts,
    )
    iv_svi_df["ssvi_rmse_vpts"]    = round(report.overall_rmse_vpts, 4)
    iv_svi_df["ssvi_fit_ok"]       = report.fit_acceptable
    iv_svi_df["gj_violations"]     = len(report.gj_violations)
    iv_svi_df["ssvi_svi_gap_vpts"] = (
        round(float(report.ssvi_svi_gap_vpts), 4)
        if not np.isnan(report.ssvi_svi_gap_vpts) else np.nan
    )
    iv_svi_df["smile_stress_flag"] = report.smile_stress_flag
    iv_svi_df["vol_regime"]        = vol_regime

    iv_svi_df["svi_rmse_vpts"]           = round(svi_report.overall_rmse_vpts, 4)
    iv_svi_df["svi_calendar_ok"]         = len(svi_report.calendar_violations) == 0
    iv_svi_df["svi_calendar_violations"] = len(svi_report.calendar_violations)

    # ── 10. Save ──────────────────────────────────────────────────────────────
    base_cols = [
        "time_to_expiry", "strike", "log_moneyness",
        # Market data
        "option_type", "open_interest", "close",
        "underlying_value", "interest_rate", "dividend",
        "futures_close",
        # BS IV
        "implied_volatility", "iv_interpolated", "iv_valid_for_ssvi",
        "bs_implied_variance", "dw_dk_bs", "d2w_dk2_bs", "dw_dT_bs",
        # SSVI surface
        "implied_volatility_svi", "implied_variance_svi",
        "dw_dT_svi", "dw_dk_svi", "d2w_dk2_svi",
        # Dupire local vol
        "dupire_local_volatility", "dupire_local_variance",
        # Fit quality metadata
        "ssvi_rmse_vpts", "ssvi_fit_ok", "gj_violations",
        "ssvi_svi_gap_vpts", "smile_stress_flag",
        # Data quality flags
        "is_clean", "pcp_forward_ok", "is_stale", "slice_liquid", "vol_regime",
    ]
    svi_pe_cols = [c for c in iv_svi_df.columns if c.endswith("_pe")]
    svi_meta    = [c for c in ["svi_rmse_vpts", "svi_calendar_ok", "svi_calendar_violations"]
                   if c in iv_svi_df.columns]
    save_cols   = [c for c in base_cols + svi_pe_cols + svi_meta if c in iv_svi_df.columns]

    path = ParquetSaver.upsert_impliedvolsurface(
        iv_svi_df[save_cols], date_str, symbol=symbol,
    )
    logger.info("[Save] Pipeline complete for %s  symbol=%s", date_str, symbol)
    logger.info("=" * 60)
    return path


if __name__ == "__main__":
    date_arg       = sys.argv[1] if len(sys.argv) > 1 else "2026-03-10"
    symbol_arg     = sys.argv[2] if len(sys.argv) > 2 else None
    instrument_arg = sys.argv[3] if len(sys.argv) > 3 else None
    calibrate(date_arg, symbol_arg, instrument_arg)
