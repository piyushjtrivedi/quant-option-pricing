import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

# ── DEMO MODE toggle ──────────────────────────────────────────────────────────
# Set DEMO_MODE = True  to serve a pre-baked parquet (no pipeline, no NSE downloads).
# Set DEMO_MODE = False to run the full live calibration pipeline.
DEMO_MODE = True

# ── DEMO MODE imports (active when DEMO_MODE = True) ─────────────────────────
# Path to the committed pre-baked parquet. Only used in demo mode.
DEMO_PARQUET = PROJECT_ROOT / "data" / "processed" / "iv_surfaces" / "BANKNIFTY.parquet"

# ── LIVE MODE imports (active when DEMO_MODE = False) ────────────────────────
# Comment these back in and set DEMO_MODE = False to restore the live pipeline.
# from src.api.iv_surface_service import ensure_iv_surface
# from src.api.ticker_service import fetch_ticker_data
# from src.config.settings import Settings

from frontend.components.plots import (
    _has_svi_pe,
    plot_iv_surface,
    plot_smile_comparison,
    plot_atm_term_structure,
    plot_iv_residual_heatmap,
    plot_smile_slices,
    plot_skew_profile,
    plot_lv_iv_ratio,
    plot_oi_weighted_smile,
    plot_butterfly_convexity,
    plot_calendar_slope,
    plot_data_quality_summary,
    plot_atm_iv_history,
    plot_skew_history,
    plot_vol_regime_history,
)

# ── session state ─────────────────────────────────────────────────────────────
for key, default in [("df", None), ("data_loaded", False), ("full_df", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── LIVE MODE helper (only used when DEMO_MODE = False) ──────────────────────
# @st.cache_data(show_spinner=False)
# def _load_ticker_data() -> pd.DataFrame:
#     return fetch_ticker_data()


def _safe(fig_fn, *args, **kwargs):
    try:
        st.plotly_chart(fig_fn(*args, **kwargs), use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Chart unavailable: {e}")


def _trading_dates(full_df: pd.DataFrame) -> list[str]:
    """Sorted list of unique trading dates present in the parquet."""
    if full_df is None or "trade_date" not in full_df.columns:
        return []
    return sorted(full_df["trade_date"].unique().tolist())


# ── dashboard ─────────────────────────────────────────────────────────────────
def render_dashboard():
    st.title("📈 Vol Surface Analytics Dashboard")
    st.caption("NSE EOD options  ·  BS IV → SSVI → Per-Slice SVI → Dupire Local Vol")

    if DEMO_MODE:
        # ── DEMO MODE controls ────────────────────────────────────────────────
        # Reads dates available in the pre-baked parquet; no pipeline run needed.
        st.info("📦 Demo mode — pre-baked surface snapshot. Set 'DEMO_MODE = False' to enable live calibration.")
        try:
            _demo_full = pd.read_parquet(DEMO_PARQUET)
        except FileNotFoundError:
            st.error(f"❌ Demo parquet not found at {DEMO_PARQUET}")
            st.stop()
        _demo_dates  = sorted(_demo_full["trade_date"].unique().tolist())
        _demo_symbol = _demo_full["trade_date"].iloc[0] and "BANKNIFTY"  # single symbol in demo file

        with st.container():
            c1, c2 = st.columns([1.2, 0.5])
            with c1: trade_date = st.selectbox("Trade Date", _demo_dates, index=len(_demo_dates) - 1)
            with c2: load_btn   = st.button("📥 Load", type="primary")

        if load_btn:
            date_key = str(trade_date)
            df = _demo_full[_demo_full["trade_date"] == date_key].reset_index(drop=True)
            if df.empty:
                st.warning("⚠️ No data for the selected date in the demo snapshot.")
                st.session_state.data_loaded = False
            else:
                st.session_state.df          = df
                st.session_state.full_df     = _demo_full
                st.session_state.data_loaded = True
                st.success(
                    f"✅ Surface loaded  ·  {len(df):,} rows  ·  "
                    f"{len(_demo_dates)} date(s) in snapshot"
                )

    else:
        # ── LIVE MODE controls ────────────────────────────────────────────────
        # Comment this block back in (and set DEMO_MODE = False) to enable the
        # full live calibration pipeline with NSE bhavcopy downloads.
        #
        # ticker_df  = _load_ticker_data()
        # symbol_list = ticker_df["TckrSymb"].tolist()
        # symbol_to_instrument = dict(zip(ticker_df["TckrSymb"], ticker_df["instrument_type"]))
        #
        # with st.container():
        #     c1, c2, c3, c4 = st.columns([1.2, 1.2, 0.7, 0.7])
        #     with c1: trade_date = st.date_input("Trade Date")
        #     with c2: symbol     = st.selectbox("Symbol", symbol_list)
        #     with c3: force      = st.checkbox("Recompute")
        #     with c4: load_btn   = st.button("📥 Load", type="primary")
        #
        # instrument = symbol_to_instrument.get(symbol, "ID")
        #
        # if load_btn:
        #     with st.spinner("Running calibration pipeline..."):
        #         try:
        #             path = ensure_iv_surface(
        #                 trade_date=str(trade_date), symbol=symbol,
        #                 instrument=instrument, force_recompute=force,
        #             )
        #             if path is None:
        #                 st.warning("⚠️ IV surface could not be generated.")
        #                 st.session_state.data_loaded = False
        #             else:
        #                 full_df  = pd.read_parquet(path)
        #                 date_key = str(trade_date)
        #                 if "trade_date" in full_df.columns:
        #                     df = full_df[full_df["trade_date"] == date_key].reset_index(drop=True)
        #                 else:
        #                     df = full_df
        #                 if df.empty:
        #                     st.warning("⚠️ IV surface data is empty for this date.")
        #                     st.session_state.data_loaded = False
        #                 else:
        #                     st.session_state.df       = df
        #                     st.session_state.full_df  = full_df
        #                     st.session_state.data_loaded = True
        #                     n_dates = full_df["trade_date"].nunique() if "trade_date" in full_df.columns else 1
        #                     st.success(
        #                         f"✅ Surface loaded  ·  {len(df):,} rows  ·  "
        #                         f"{n_dates} date(s) in store"
        #                     )
        #         except Exception as e:
        #             st.error(f"❌ {e}")
        #             st.session_state.data_loaded = False
        st.warning("DEMO_MODE is False but live pipeline code is commented out. Set DEMO_MODE = True or uncomment the live block above.")
        st.stop()

    if not st.session_state.data_loaded:
        st.info("Select a date above and click **Load** to begin.")
        st.stop()

    df      = st.session_state.df
    full_df = st.session_state.full_df

    # ── quality banner ────────────────────────────────────────────────────────
    cols = st.columns(5)
    if "ssvi_rmse_vpts" in df.columns:
        rmse   = float(df["ssvi_rmse_vpts"].iloc[0])
        fit_ok = bool(df["ssvi_fit_ok"].iloc[0])   if "ssvi_fit_ok"    in df.columns else True
        gj_n   = int(df["gj_violations"].iloc[0])  if "gj_violations"  in df.columns else 0
        cols[0].metric("SSVI RMSE", f"{rmse:.2f} vpts",
                       delta="✅ Good" if fit_ok else "⚠️ Check",
                       delta_color="normal" if fit_ok else "inverse")
        cols[1].metric("GJ No-Arb", "Pass" if gj_n==0 else f"Fail ({gj_n})",
                       delta="✅" if gj_n==0 else "⚠️ arb possible",
                       delta_color="normal" if gj_n==0 else "inverse")
    if "svi_rmse_vpts" in df.columns:
        svi_rmse = float(df["svi_rmse_vpts"].iloc[0])
        cal_ok   = bool(df["svi_calendar_ok"].iloc[0]) if "svi_calendar_ok" in df.columns else True
        cols[2].metric("SVI RMSE", f"{svi_rmse:.2f} vpts",
                       delta="✅ Good" if svi_rmse < 1.5 else "⚠️ Check",
                       delta_color="normal" if svi_rmse < 1.5 else "inverse")
        cols[3].metric("SVI Calendar", "Pass" if cal_ok else "Fail",
                       delta="✅" if cal_ok else "⚠️ check wings",
                       delta_color="normal" if cal_ok else "inverse")
    n_clean = int(df["is_clean"].sum()) if "is_clean" in df.columns else len(df)
    cols[4].metric("Clean Contracts", n_clean)

    has_pe = _has_svi_pe(df)
    if has_pe:
        st.success("✅ Per-slice SVI surface available — used for comparison plots")

    # ── visibility toggles ────────────────────────────────────────────────────
    with st.expander("📊 Show / Hide Sections", expanded=True):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        show_surfaces = r1c1.checkbox("3D Surfaces",        value=True)
        show_compare  = r1c2.checkbox("Smile Comparison",   value=True)
        show_atm      = r1c3.checkbox("ATM Term Structure", value=True)
        show_skew     = r1c4.checkbox("Skew Profile",       value=True)
        show_oi       = r2c1.checkbox("OI-Weighted Smile",  value=True)
        show_lv       = r2c2.checkbox("LV / IV Ratio",      value=True)
        show_slices   = r2c3.checkbox("Smile Slices",       value=True)
        show_arb      = r2c4.checkbox("Arb Diagnostics",    value=True)
        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        show_heatmap  = r3c1.checkbox("Residual Heatmap",   value=True)
        show_quality  = r3c2.checkbox("Data Quality",       value=True)
        # Cross-date analytics only visible if enough history
        trading_dates = _trading_dates(full_df)
        has_history   = len(trading_dates) >= 3
        show_history  = r3c4.checkbox("Vol History", value=has_history) if has_history else False

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 1. SMILE COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    if show_compare:
        st.subheader("1 · Smile Comparison — BS IV · SSVI · SVI per-slice")
        st.caption(
            "**Blue dots** = raw traded BS IV.  "
            "**Open circles** = PCHIP gap-fills.  "
            "**Amber** = SSVI.  **Green dashed** = per-slice SVI."
        )
        _safe(plot_smile_comparison, df)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. 3D SURFACES
    # ══════════════════════════════════════════════════════════════════════════
    if show_surfaces:
        st.subheader("2 · 3D Implied Volatility Surfaces")
        cols_3d = [
            ("implied_volatility",     "BS IV (traded)"),
            ("implied_volatility_svi", "SSVI"),
        ]
        if has_pe:
            cols_3d.append(("implied_volatility_svi_pe", "SVI per-slice"))
        cols_widgets = st.columns(len(cols_3d))
        for i, (col, lbl) in enumerate(cols_3d):
            with cols_widgets[i]:
                st.caption(lbl)
                _safe(plot_iv_surface, df, col)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. ATM TERM STRUCTURE
    # ══════════════════════════════════════════════════════════════════════════
    if show_atm:
        st.subheader("3 · ATM Volatility Term Structure")
        st.caption(
            "All surfaces at ATM (k≈0) across expiries.  "
            "Contango = near-dated vol cheap.  Backwardation = near-dated vol rich."
        )
        _safe(plot_atm_term_structure, df)

    # ══════════════════════════════════════════════════════════════════════════
    # 4. SKEW PROFILE
    # ══════════════════════════════════════════════════════════════════════════
    if show_skew:
        st.subheader("4 · Volatility Skew Profile")
        st.caption(
            "**Risk Reversal** (25Δ put − 25Δ call).  "
            "**Butterfly** (wing avg − ATM)."
        )
        _safe(plot_skew_profile, df)

    # ══════════════════════════════════════════════════════════════════════════
    # 5. OI-WEIGHTED SMILE
    # ══════════════════════════════════════════════════════════════════════════
    if show_oi:
        st.subheader("5 · OI-Weighted Smile")
        st.caption("Bubble size = open interest.")
        _safe(plot_oi_weighted_smile, df)

    # ══════════════════════════════════════════════════════════════════════════
    # 6. LV / IV RATIO
    # ══════════════════════════════════════════════════════════════════════════
    if show_lv:
        st.subheader("6 · Local Vol / Implied Vol Ratio")
        st.caption("LV/IV > 1 → forward vol premium.  Key for barrier pricing.")
        _safe(plot_lv_iv_ratio, df)

    # ══════════════════════════════════════════════════════════════════════════
    # 7. SMILE SLICES
    # ══════════════════════════════════════════════════════════════════════════
    if show_slices:
        st.subheader("7 · Smile Slices per Model")
        st.caption(
            "Row 1: BS IV.  Row 2: SSVI.  "
            + ("Row 3: SVI per-slice.  " if has_pe else "")
            + "Last row: Local Vol."
        )
        _safe(plot_smile_slices, df)

    # ══════════════════════════════════════════════════════════════════════════
    # 8. NO-ARB DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════════
    if show_arb:
        st.subheader("8 · No-Arbitrage Diagnostics")
        st.caption("d²w/dk² ≥ 0 (butterfly).  dw/dT ≥ 0 (calendar).")
        l, r = st.columns(2)
        with l: _safe(plot_butterfly_convexity, df)
        with r: _safe(plot_calendar_slope, df)

    # ══════════════════════════════════════════════════════════════════════════
    # 9. RESIDUAL HEATMAP
    # ══════════════════════════════════════════════════════════════════════════
    if show_heatmap:
        st.subheader("9 · IV Residual Heatmap  (BS − Model)")
        st.caption("White = perfect fit.  Systematic colour = model bias.")
        _safe(plot_iv_residual_heatmap, df)

    # ══════════════════════════════════════════════════════════════════════════
    # 10. DATA QUALITY
    # ══════════════════════════════════════════════════════════════════════════
    if show_quality:
        st.subheader("10 · Data Quality Summary")
        _safe(plot_data_quality_summary, df)
        with st.expander("🔍 Raw Data Snapshot", expanded=False):
            st.dataframe(df[df["implied_volatility"].notna()], use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 11. CROSS-DATE VOL HISTORY  (≥ 3 trading days required)
    # ══════════════════════════════════════════════════════════════════════════
    if show_history and full_df is not None:
        st.divider()
        st.subheader(f"11 · Vol History  ({len(trading_dates)} trading days)")
        st.caption(
            "All dates currently stored for this symbol.  "
            "Uses actual trading days in the parquet as the calendar — no external holiday list needed."
        )

        h1, h2, h3 = st.columns(3)
        with h1: st.metric("Trading Days Stored", len(trading_dates))
        with h2: st.metric("First Date", trading_dates[0])
        with h3: st.metric("Latest Date", trading_dates[-1])

        st.markdown("**ATM Implied Vol History**")
        _safe(plot_atm_iv_history, full_df)

        st.markdown("**25Δ Skew History**")
        _safe(plot_skew_history, full_df)

        st.markdown("**Vol Regime by Date**")
        _safe(plot_vol_regime_history, full_df)
