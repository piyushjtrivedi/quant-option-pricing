"""
All Plotly chart functions for the IV Surface & Vol Analytics Dashboard.

Key design principles:
  - TRADED expiries only for smile/slice/OI charts (not SSVI grid rows)
  - Moneyness clipped to |k| <= K_CLIP=0.15 for all smile plots
  - Traded rows identified by option_type not null
  - Grid rows (SSVI/SVI surface) used for smooth model lines only

Colour convention (consistent across all charts):
  Blue   (#2196F3) — raw BS IV traded points
  Amber  (#FF6F00) — SSVI surface
  Green  (#2E7D32) — Per-slice SVI surface
  Purple (#7B1FA2) — Dupire local vol
  Red    (#E53935) — sell signals / risk reversal
  Teal   (#00838F) — buy signals
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import PchipInterpolator

# ── colour palette ────────────────────────────────────────────────────────────
_C = dict(
    bs_raw  = "#2196F3",
    bs_grid = "#90CAF9",
    ssvi    = "#FF6F00",
    svi_pe  = "#2E7D32",
    lv      = "#7B1FA2",
    rr      = "#E53935",
    buy     = "#00838F",
    sell    = "#E53935",
    zero    = "#9E9E9E",
)

K_CLIP = 0.15   # clip moneyness axis to ±15% for all smile plots

# ── helpers ───────────────────────────────────────────────────────────────────
def _days(T):        return max(1, round(T * 365))
def _lbl(T):         return f"{_days(T)}d"
def _lbls(expiries): return [_lbl(T) for T in expiries]

def _has_svi_pe(df):
    return ("implied_volatility_svi_pe" in df.columns and
            df["implied_volatility_svi_pe"].notna().sum() > 5)

def _traded_expiries(df):
    """Only expiries that have actual traded contracts (option_type not null)."""
    if "option_type" not in df.columns:
        return sorted(df["time_to_expiry"].unique().tolist())
    traded = df[df["option_type"].notna()]
    return sorted(traded["time_to_expiry"].unique().tolist())

def _all_expiries(df):
    return sorted(df["time_to_expiry"].unique().tolist())

def _traded(df):
    """Rows with actual traded market data."""
    if "option_type" in df.columns:
        return df[df["option_type"].notna()]
    return df

def _clip_k(df, col="log_moneyness", limit=K_CLIP):
    return df[df[col].abs() <= limit]

def _delta_to_k(delta, T, atm_iv):
    from scipy.stats import norm
    return -atm_iv * np.sqrt(max(T, 1e-6)) * abs(norm.ppf(delta))


# ─────────────────────────────────────────────────────────────────────────────
# 1. THREE-MODEL SMILE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def plot_smile_comparison(df: pd.DataFrame) -> go.Figure:
    """
    One subplot per TRADED expiry.
    Blue dots = raw traded BS IV.  Open circles = PCHIP gap-fills.
    Amber line = SSVI.  Green dashed = per-slice SVI.
    X-axis clipped to |k| <= 0.15.
    """
    expiries = _traded_expiries(df)
    n        = len(expiries)
    has_pe   = _has_svi_pe(df)

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[_lbl(T) for T in expiries],
        shared_yaxes=True,
        horizontal_spacing=0.06,
        vertical_spacing=0.14,
    )

    def _rc(i):
        return (i // ncols + 1, i % ncols + 1)

    for i, T in enumerate(expiries):
        row, col = _rc(i)

        # Grid rows for smooth model lines (clipped)
        grid_sl = _clip_k(df[np.isclose(df["time_to_expiry"], T)]).sort_values("log_moneyness")

        # SSVI line
        ssvi = grid_sl[grid_sl["implied_volatility_svi"].notna()]
        if not ssvi.empty:
            fig.add_trace(go.Scatter(
                x=ssvi["log_moneyness"], y=ssvi["implied_volatility_svi"],
                mode="lines", name="SSVI", showlegend=(i == 0),
                line=dict(color=_C["ssvi"], width=2.5),
            ), row=row, col=col)

        # SVI per-slice line
        if has_pe:
            svi = grid_sl[grid_sl["implied_volatility_svi_pe"].notna()]
            if not svi.empty:
                fig.add_trace(go.Scatter(
                    x=svi["log_moneyness"], y=svi["implied_volatility_svi_pe"],
                    mode="lines", name="SVI (per-slice)", showlegend=(i == 0),
                    line=dict(color=_C["svi_pe"], width=2.5, dash="dash"),
                ), row=row, col=col)

        # Traded rows only for BS IV points
        trd_sl = _clip_k(_traded(df)[np.isclose(_traded(df)["time_to_expiry"], T)]).sort_values("log_moneyness")

        # Gap-filled
        if "iv_interpolated" in trd_sl.columns:
            interp = trd_sl[trd_sl["iv_interpolated"] == True]
            if not interp.empty:
                fig.add_trace(go.Scatter(
                    x=interp["log_moneyness"], y=interp["implied_volatility"],
                    mode="markers", name="Gap-filled", showlegend=(i == 0),
                    marker=dict(color=_C["bs_grid"], size=6, symbol="circle-open"),
                ), row=row, col=col)

        # Raw traded
        raw_col = "iv_valid_for_ssvi" if "iv_valid_for_ssvi" in trd_sl.columns else None
        raw = trd_sl[trd_sl[raw_col] == True] if raw_col else \
              trd_sl[trd_sl["implied_volatility"].notna()]
        if not raw.empty:
            fig.add_trace(go.Scatter(
                x=raw["log_moneyness"], y=raw["implied_volatility"],
                mode="markers", name="BS IV (traded)", showlegend=(i == 0),
                marker=dict(color=_C["bs_raw"], size=7, symbol="circle"),
            ), row=row, col=col)

        fig.add_vline(x=0, line_dash="dot", line_color=_C["zero"],
                      line_width=1, row=row, col=col)

    # Axis labels only on edge subplots
    for r in range(1, nrows + 1):
        fig.update_yaxes(title_text="IV (%)", row=r, col=1)
    for c in range(1, ncols + 1):
        fig.update_xaxes(title_text="log(K/F)", row=nrows, col=c)

    fig.update_layout(
        title="Smile Comparison — BS IV (traded) · SSVI · Per-Slice SVI",
        template="plotly_white",
        height=350 * nrows,
        legend=dict(orientation="h", y=-0.08 / nrows),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. THREE-MODEL ATM TERM STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
def plot_atm_term_structure(df: pd.DataFrame) -> go.Figure:
    """ATM IV for BS, SSVI, SVI per-slice and Local Vol across expiries."""
    expiries = _traded_expiries(df)
    records  = []

    for T in expiries:
        sl  = _traded(df)[np.isclose(_traded(df)["time_to_expiry"], T)]
        if sl.empty:
            continue
        idx = sl["log_moneyness"].abs().idxmin()
        row = sl.loc[idx]
        rec = {"T": T, "days": _days(T)}
        for c in ["implied_volatility", "implied_volatility_svi",
                  "implied_volatility_svi_pe", "dupire_local_volatility"]:
            rec[c] = float(row[c]) if c in sl.columns and pd.notna(row.get(c)) else np.nan
        records.append(rec)

    # Model IV at exactly k=0, interpolated from grid rows using PCHIP.
    # Using idxmin() on discretised k-grid can land on k=-0.02 or k=+0.02
    # which shifts ATM IV by up to 2 vol pts on a steep skew — misleading.
    # PchipInterpolator(k, iv)(0.0) gives the true ATM model value.
    def _atm_from_grid(sl, iv_col):
        d = sl[sl[iv_col].notna() & (sl[iv_col] > 0)].sort_values("log_moneyness") if iv_col in sl.columns else pd.DataFrame()
        if len(d) < 3:
            return np.nan
        k  = d["log_moneyness"].values
        iv = d[iv_col].values
        # deduplicate k
        _, idx = np.unique(k, return_index=True)
        k, iv = k[idx], iv[idx]
        if k.min() > 0 or k.max() < 0:
            return np.nan
        return float(PchipInterpolator(k, iv, extrapolate=False)(0.0))

    grid_records = []
    for T in _all_expiries(df):
        sl = df[np.isclose(df["time_to_expiry"], T)].copy()
        if sl.empty: continue
        grid_records.append({"T": T, "days": _days(T),
            "ssvi":   _atm_from_grid(sl, "implied_volatility_svi"),
            "svi_pe": _atm_from_grid(sl, "implied_volatility_svi_pe"),
            "lv":     _atm_from_grid(sl, "dupire_local_volatility"),
        })

    if not records:
        return go.Figure().update_layout(title="ATM Term Structure — no data")

    atm  = pd.DataFrame(records).sort_values("T")
    gatm = pd.DataFrame(grid_records).sort_values("T") if grid_records else pd.DataFrame()

    fig = go.Figure()

    # BS IV from traded rows (dots)
    if "implied_volatility" in atm.columns:
        fig.add_trace(go.Scatter(
            x=atm["days"], y=atm["implied_volatility"],
            mode="markers+lines", name="BS IV (ATM)",
            line=dict(color=_C["bs_raw"], width=2),
            marker=dict(size=9),
        ))

    # SSVI, SVI, LV from grid (smooth lines)
    if not gatm.empty:
        for col, name, color, dash in [
            ("ssvi",  "SSVI (ATM)",          _C["ssvi"],   "solid"),
            ("svi_pe","SVI per-slice (ATM)",  _C["svi_pe"], "dash"),
            ("lv",    "Local Vol (ATM)",       _C["lv"],     "dot"),
        ]:
            if col in gatm.columns and gatm[col].notna().any():
                fig.add_trace(go.Scatter(
                    x=gatm["days"], y=gatm[col],
                    mode="lines+markers", name=name,
                    line=dict(color=color, width=2.5, dash=dash),
                    marker=dict(size=7),
                ))

    fig.update_layout(
        title="ATM Volatility Term Structure — BS · SSVI · SVI · Local Vol",
        xaxis_title="Days to Expiry",
        yaxis_title="ATM Implied Vol (%)",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.18),
        height=420,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. RESIDUAL HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def plot_iv_residual_heatmap(df: pd.DataFrame,
                              iv_1: str = "implied_volatility",
                              iv_2: str = "implied_volatility_svi") -> go.Figure:
    """
    BS minus SSVI (left) and BS minus SVI per-slice (right).
    Uses traded rows only, clipped to |k| <= 0.15.
    """
    has_pe   = _has_svi_pe(df)
    n_panels = 2 if has_pe else 1
    titles   = ["BS − SSVI", "BS − SVI (per-slice)"] if has_pe else ["BS − SSVI"]

    fig = make_subplots(rows=1, cols=n_panels, subplot_titles=titles,
                        horizontal_spacing=0.10)

    def _heatmap(iv_col, panel):
        if iv_col not in df.columns:
            return
        trd = _traded(df)
        d = _clip_k(trd[
            trd["implied_volatility"].notna() &
            trd[iv_col].notna()
        ].copy())
        if d.empty:
            return
        d = d.copy()
        d["residual"] = d["implied_volatility"] - d[iv_col]
        pivot = d.pivot_table(index="time_to_expiry", columns="log_moneyness",
                              values="residual", aggfunc="mean")
        abs_max = max(np.nanmax(np.abs(pivot.values)), 0.1)
        y_labels = [f"{_days(T)}d" for T in pivot.index]
        fig.add_trace(go.Heatmap(
            x=pivot.columns, y=y_labels, z=pivot.values,
            colorscale="RdBu", zmid=0, zmin=-abs_max, zmax=abs_max,
            colorbar=dict(title="Δ vpts",
                          x=0.44 if panel == 1 else 1.01, len=0.85),
            hovertemplate="k=%{x:.3f}<br>T=%{y}<br>Res=%{z:.2f} vpts<extra></extra>",
        ), row=1, col=panel)

    _heatmap("implied_volatility_svi", 1)
    if has_pe:
        _heatmap("implied_volatility_svi_pe", 2)

    fig.update_xaxes(title_text="Log-Moneyness")
    fig.update_yaxes(title_text="Expiry", col=1)
    fig.update_layout(
        title="IV Residual Heatmap — BS minus Model  (white = perfect fit)",
        template="plotly_white", height=380,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. SMILE SLICES  (row per model, column per traded expiry)
# ─────────────────────────────────────────────────────────────────────────────
def plot_smile_slices(df: pd.DataFrame) -> go.Figure:
    """
    Rows: BS IV dots | SSVI | SVI per-slice | Local Vol
    Columns: traded expiries only, x-axis clipped to |k| <= 0.15.
    """
    has_pe   = _has_svi_pe(df)
    expiries = _traded_expiries(df)
    n_exp    = len(expiries)

    rows_cfg = [("implied_volatility", "BS IV", _C["bs_raw"], "markers")]
    rows_cfg.append(("implied_volatility_svi", "SSVI", _C["ssvi"], "lines"))
    if has_pe:
        rows_cfg.append(("implied_volatility_svi_pe", "SVI per-slice", _C["svi_pe"], "lines"))
    rows_cfg.append(("dupire_local_volatility", "Local Vol", _C["lv"], "lines"))

    n_rows = len(rows_cfg)

    # Grid: one row per model, one column per expiry
    fig = make_subplots(
        rows=n_rows, cols=n_exp,
        subplot_titles=[_lbl(T) for T in expiries],
        row_titles=[r[1] for r in rows_cfg],
        shared_xaxes=True,
        shared_yaxes="rows",
        vertical_spacing=0.05,
        horizontal_spacing=0.04,
    )

    for col_i, T in enumerate(expiries):
        # Grid slice for model lines
        grid_sl = _clip_k(df[np.isclose(df["time_to_expiry"], T)]).sort_values("log_moneyness")
        # Traded slice for BS IV dots
        trd_sl  = _clip_k(_traded(df)[np.isclose(_traded(df)["time_to_expiry"], T)]).sort_values("log_moneyness")

        for row_i, (col_name, name, color, mode) in enumerate(rows_cfg):
            src = trd_sl if mode == "markers" else grid_sl
            d   = src[src[col_name].notna() & (src[col_name] > 0)] if col_name in src.columns else pd.DataFrame()
            if d.empty:
                continue
            fig.add_trace(go.Scatter(
                x=d["log_moneyness"], y=d[col_name],
                mode=mode, name=name,
                showlegend=(col_i == 0),
                marker=dict(color=color, size=5),
                line=dict(color=color, width=2),
            ), row=row_i + 1, col=col_i + 1)

    for c in range(1, n_exp + 1):
        fig.update_xaxes(title_text="log(K/F)", row=n_rows, col=c)
    for r in range(1, n_rows + 1):
        fig.update_yaxes(title_text="IV (%)", col=1, row=r)

    fig.update_layout(
        title="Smile Slices — BS · SSVI · SVI per-slice · Local Vol",
        template="plotly_white",
        height=220 * n_rows,
        legend=dict(orientation="h", y=-0.04),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. SKEW PROFILE  (25Δ RR and Butterfly from SSVI and SVI)
# ─────────────────────────────────────────────────────────────────────────────
def plot_skew_profile(df: pd.DataFrame) -> go.Figure:
    has_pe   = _has_svi_pe(df)
    expiries = _traded_expiries(df)
    records  = []

    surfaces = [("implied_volatility_svi", "SSVI")]
    if has_pe:
        surfaces.append(("implied_volatility_svi_pe", "SVI"))

    for T in expiries:
        sl = _clip_k(df[np.isclose(df["time_to_expiry"], T)]).sort_values("log_moneyness")
        for iv_col, label in surfaces:
            d = sl[sl[iv_col].notna() & (sl[iv_col] > 0)] if iv_col in sl.columns else pd.DataFrame()
            if len(d) < 5:
                continue
            # Deduplicate k — grid rows can have CE+PE at same log_moneyness.
            # Average IV across duplicates before fitting PCHIP.
            smile = d.groupby("log_moneyness")[iv_col].mean().reset_index().sort_values("log_moneyness")
            k  = smile["log_moneyness"].values
            iv = smile[iv_col].values / 100
            if len(k) < 3:
                continue
            atm_iv = float(iv[np.argmin(np.abs(k))])
            k25p   = _delta_to_k(0.25, T, atm_iv)
            k25c   = -k25p
            pchip  = PchipInterpolator(k, iv, extrapolate=False)
            iv25p  = pchip(k25p)
            iv25c  = pchip(k25c)
            if np.isnan(iv25p) or np.isnan(iv25c):
                continue
            records.append({
                "T": T, "days": _days(T), "label": label,
                "rr": (iv25p - iv25c) * 100,
                "bf": (0.5 * (iv25p + iv25c) - atm_iv) * 100,
            })

    if not records:
        return go.Figure().update_layout(title="Skew Profile — insufficient data")

    sk    = pd.DataFrame(records)
    clrs  = {"SSVI": _C["ssvi"], "SVI": _C["svi_pe"]}

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["25Δ Risk Reversal  (put IV − call IV)",
                        "25Δ Butterfly  (wing avg − ATM)"],
        vertical_spacing=0.22, shared_xaxes=True,
    )
    for metric, row in [("rr", 1), ("bf", 2)]:
        for label in sk["label"].unique():
            sub = sk[sk["label"] == label].sort_values("days")
            fig.add_trace(go.Bar(
                x=[str(d) for d in sub["days"]], y=sub[metric],
                name=label, marker_color=clrs.get(label, _C["ssvi"]),
                showlegend=(metric == "rr"),
                text=[f"{v:+.2f}" for v in sub[metric]], textposition="outside",
                offsetgroup=label,
            ), row=row, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color=_C["zero"], row=row, col=1)

    fig.update_xaxes(title_text="Days to Expiry", row=2, col=1)
    fig.update_yaxes(title_text="Vol Pts", row=1, col=1)
    fig.update_yaxes(title_text="Vol Pts", row=2, col=1)
    fig.update_layout(
        title="Volatility Skew Profile — 25Δ Risk Reversal & Butterfly",
        template="plotly_white", height=520,
        barmode="group", legend=dict(orientation="h", y=-0.12),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. OI-WEIGHTED SMILE
# ─────────────────────────────────────────────────────────────────────────────
def plot_oi_weighted_smile(df: pd.DataFrame) -> go.Figure:
    """
    Uses TRADED rows only. X-axis clipped to |k| <= 0.15.
    CE circles, PE triangles. SSVI and SVI lines from grid rows.
    """
    has_pe   = _has_svi_pe(df)
    trd      = _clip_k(_traded(df)[
        df["implied_volatility"].notna() & (df["implied_volatility"] > 0) &
        df["open_interest"].notna()
    ].copy()) if "option_type" in df.columns else pd.DataFrame()

    expiries = _traded_expiries(df)
    n        = len(expiries)
    ncols    = min(n, 4)
    nrows    = (n + ncols - 1) // ncols

    if trd.empty:
        return go.Figure().update_layout(title="OI-Weighted Smile — no traded data")

    oi_max = max(trd["open_interest"].max(), 1)
    oi_min = trd["open_interest"].min()
    def _sz(oi): return 5 + 25 * (oi - oi_min) / max(oi_max - oi_min, 1)

    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=[_lbl(T) for T in expiries],
                        shared_yaxes=True,
                        horizontal_spacing=0.06, vertical_spacing=0.14)

    def _rc(i): return (i // ncols + 1, i % ncols + 1)

    for i, T in enumerate(expiries):
        row, col = _rc(i)
        sl_trd  = trd[np.isclose(trd["time_to_expiry"], T)]
        sl_grid = _clip_k(df[np.isclose(df["time_to_expiry"], T)]).sort_values("log_moneyness")

        for opt, sym, color in [("CE","circle",_C["bs_raw"]),
                                  ("PE","triangle-up",_C["rr"])]:
            leg = sl_trd[sl_trd["option_type"] == opt].sort_values("log_moneyness")
            if leg.empty: continue
            fig.add_trace(go.Scatter(
                x=leg["log_moneyness"], y=leg["implied_volatility"],
                mode="markers", name=opt, showlegend=(i == 0),
                marker=dict(color=color, size=[_sz(oi) for oi in leg["open_interest"]],
                            symbol=sym, line=dict(width=0.5, color="white")),
                customdata=np.stack([leg["strike"], leg["open_interest"]], axis=1),
                hovertemplate="K=%{customdata[0]:,.0f}<br>IV=%{y:.1f}%<br>OI=%{customdata[1]:,.0f}<extra></extra>",
            ), row=row, col=col)

        # Model lines from grid
        ssvi = sl_grid[sl_grid["implied_volatility_svi"].notna()]
        if not ssvi.empty:
            fig.add_trace(go.Scatter(
                x=ssvi["log_moneyness"], y=ssvi["implied_volatility_svi"],
                mode="lines", name="SSVI", showlegend=(i == 0),
                line=dict(color=_C["ssvi"], width=2, dash="dot"),
            ), row=row, col=col)

        if has_pe:
            svi = sl_grid[sl_grid["implied_volatility_svi_pe"].notna()]
            if not svi.empty:
                fig.add_trace(go.Scatter(
                    x=svi["log_moneyness"], y=svi["implied_volatility_svi_pe"],
                    mode="lines", name="SVI", showlegend=(i == 0),
                    line=dict(color=_C["svi_pe"], width=2, dash="dash"),
                ), row=row, col=col)

        fig.add_vline(x=0, line_dash="dot", line_color=_C["zero"],
                      line_width=1, row=row, col=col)

    for r in range(1, nrows + 1):
        fig.update_yaxes(title_text="BS IV (%)", row=r, col=1)
    for c in range(1, ncols + 1):
        fig.update_xaxes(title_text="log(K/F)", row=nrows, col=c)

    fig.update_layout(
        title="OI-Weighted Smile  (size = OI · ● CE · ▲ PE · model lines overlaid)",
        template="plotly_white", height=360 * nrows,
        legend=dict(orientation="h", y=-0.08 / nrows),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. LOCAL VOL / IMPLIED VOL RATIO
# ─────────────────────────────────────────────────────────────────────────────
def plot_lv_iv_ratio(df: pd.DataFrame) -> go.Figure:
    d = _clip_k(df[
        df["dupire_local_volatility"].notna() & (df["dupire_local_volatility"] > 0) &
        df["implied_volatility_svi"].notna()  & (df["implied_volatility_svi"] > 0)
    ].copy())
    d["ratio"] = (d["dupire_local_volatility"] / d["implied_volatility_svi"]).clip(0, 3)
    pivot = d.pivot_table(index="time_to_expiry", columns="log_moneyness",
                          values="ratio", aggfunc="mean")
    y_labels = [f"{_days(T)}d" for T in pivot.index]
    fig = go.Figure(go.Heatmap(
        x=pivot.columns, y=y_labels, z=pivot.values,
        colorscale=[[0.0,"#1565C0"],[0.33,"#90CAF9"],
                    [0.5,"#FFFFFF"],[0.67,"#FFCC80"],[1.0,"#BF360C"]],
        zmid=1.0, zmin=0, zmax=3,
        colorbar=dict(title="LV/IV", tickvals=[0,0.5,1,1.5,2,2.5,3]),
        hovertemplate="k=%{x:.3f}<br>T=%{y}<br>LV/IV=%{z:.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=_C["zero"], line_width=1)
    fig.update_layout(
        title="Local Vol / SSVI Implied Vol Ratio  (white=1, red>1 = forward vol premium)",
        xaxis_title="Log-Moneyness", yaxis_title="Expiry",
        template="plotly_white", height=380,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. NO-ARB DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
def plot_butterfly_convexity(df: pd.DataFrame) -> go.Figure:
    expiries = _traded_expiries(df)
    n = len(expiries)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(rows=2*nrows, cols=ncols,
                        subplot_titles=[_lbl(T) for T in expiries],
                        shared_yaxes=True, vertical_spacing=0.08)

    def _rc(i, model_row): return (model_row + i // ncols * 2, i % ncols + 1)

    for i, T in enumerate(expiries):
        sl = _clip_k(df[np.isclose(df["time_to_expiry"], T)]).sort_values("log_moneyness")
        for model_row, col_name, name, color in [
            (1, "d2w_dk2_bs",  "BS",   _C["bs_raw"]),
            (2, "d2w_dk2_svi", "SSVI", _C["ssvi"]),
        ]:
            r, c = _rc(i, model_row)
            if col_name not in sl.columns: continue
            fig.add_trace(go.Scatter(
                x=sl["log_moneyness"], y=sl[col_name],
                mode="lines", name=name, showlegend=(i == 0),
                line=dict(color=color, width=2),
            ), row=r, col=c)

    fig.add_hline(y=0, line_dash="dash", line_color=_C["zero"])
    fig.update_layout(
        title="Butterfly Convexity — d²w/dk²  (≥ 0 required)",
        template="plotly_white", height=420 * nrows,
    )
    return fig


def plot_calendar_slope(df: pd.DataFrame) -> go.Figure:
    trd       = _traded(df)
    quantiles = [0.25, 0.50, 0.75]
    labels    = [f"k≈{trd['log_moneyness'].quantile(q):.3f}" for q in quantiles]
    fig = make_subplots(rows=2, cols=len(quantiles), subplot_titles=labels,
                        row_titles=["BS dw/dT","SSVI dw/dT"],
                        vertical_spacing=0.16)
    for col_i, q in enumerate(quantiles):
        log_m = trd["log_moneyness"].quantile(q)
        sl    = df[np.isclose(df["log_moneyness"], log_m, atol=0.015)].sort_values("time_to_expiry")
        for row_i, (col_name, name, color) in enumerate([
            ("dw_dT_bs",  "BS",   _C["bs_raw"]),
            ("dw_dT_svi", "SSVI", _C["ssvi"]),
        ]):
            if col_name not in sl.columns or sl[col_name].isna().all(): continue
            fig.add_trace(go.Scatter(
                x=sl["time_to_expiry"], y=sl[col_name],
                mode="lines+markers", name=name, showlegend=(col_i == 0),
                line=dict(color=color, width=2),
            ), row=row_i + 1, col=col_i + 1)
    fig.add_hline(y=0, line_dash="dash", line_color=_C["zero"])
    fig.update_layout(title="Calendar Slope — dw/dT  (≥ 0 required)",
                      template="plotly_white", height=500)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. 3D SURFACE
# ─────────────────────────────────────────────────────────────────────────────
def plot_iv_surface(df: pd.DataFrame, iv_col: str) -> go.Figure:
    """Clip to |k| <= K_CLIP so the surface shows the traded region only."""
    d = _clip_k(df[(df[iv_col].notna()) & (df[iv_col] > 0) &
                   (df["time_to_expiry"] > 0)].copy())
    d["mk"] = d["log_moneyness"].round(2)
    d["mt"] = d["time_to_expiry"].round(3)
    surface = d.pivot_table(index="mt", columns="mk", values=iv_col, aggfunc="mean")
    if surface.isna().all().all():
        raise ValueError("Surface empty after clip")

    name_map = {
        "implied_volatility":        "BS IV",
        "implied_volatility_svi":    "SSVI",
        "implied_volatility_svi_pe": "SVI per-slice",
        "dupire_local_volatility":   "Local Vol",
    }
    cmap_map = {
        "implied_volatility":        "Blues",
        "implied_volatility_svi":    "Oranges",
        "implied_volatility_svi_pe": "Greens",
        "dupire_local_volatility":   "Purples",
    }
    label = name_map.get(iv_col, iv_col)
    cmap  = cmap_map.get(iv_col, "Viridis")

    fig = go.Figure(go.Surface(
        x=surface.columns.values, y=surface.index.values, z=surface.values,
        colorscale=cmap, showscale=True,
    ))
    fig.update_layout(
        title=f"{label} Surface",
        margin=dict(l=10, r=10, t=40, b=10),
        scene=dict(xaxis_title="Log-Moneyness",
                   yaxis_title="Time to Expiry",
                   zaxis_title="Implied Vol (%)"),
        height=400,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10. DATA QUALITY
# ─────────────────────────────────────────────────────────────────────────────
def plot_data_quality_summary(df: pd.DataFrame) -> go.Figure:
    d = _traded(df).copy() if "option_type" in df.columns else df.copy()
    expiries = _traded_expiries(df)
    labels   = _lbls(expiries)

    # Build boolean series safely: fill NaN with the "no-issue" default
    # so grid rows (which have NaN quality columns) don't pollute counts.
    def _bool_col(series, default: bool) -> pd.Series:
        return series.fillna(default).astype(bool)

    is_clean   = _bool_col(d.get("is_clean",       pd.Series(dtype=bool)), True)
    is_interp  = _bool_col(d.get("iv_interpolated", pd.Series(dtype=bool)), False)
    is_stale   = _bool_col(d.get("is_stale",        pd.Series(dtype=bool)), False)
    pcp_fail   = ~_bool_col(d.get("pcp_forward_ok", pd.Series(dtype=bool)), True)
    illiquid   = ~_bool_col(d.get("slice_liquid",   pd.Series(dtype=bool)), True)

    cats = {
        "Clean":          (is_clean,  "#43A047"),
        "IV Gap-filled":  (is_interp, "#90CAF9"),
        "Stale":          (is_stale,  "#E53935"),
        "PCP Fail":       (pcp_fail,  "#FF6F00"),
        "Illiquid Slice": (illiquid,  "#8E24AA"),
    }
    fig = go.Figure()
    for name, (mask, color) in cats.items():
        counts = []
        for T in expiries:
            sl = d[np.isclose(d["time_to_expiry"], T)]
            counts.append(int(mask.reindex(sl.index).fillna(False).sum()))
        fig.add_trace(go.Bar(x=labels, y=counts, name=name,
                             marker_color=color, text=counts, textposition="inside"))
    fig.update_layout(
        barmode="stack",
        title="Data Quality Summary — Contracts per Expiry",
        xaxis_title="Expiry", yaxis_title="Contracts",
        template="plotly_white", height=360,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-DATE: ATM IMPLIED VOL HISTORY
# Requires at least 3 trading days present in full_df.
# ─────────────────────────────────────────────────────────────────────────────
def plot_atm_iv_history(full_df: pd.DataFrame) -> go.Figure:
    """
    ATM IV for BS (traded), SSVI, and SVI per-slice over time.
    One line per (model × nearest-expiry-bucket). X-axis = trade_date.
    """
    if "trade_date" not in full_df.columns:
        return go.Figure().update_layout(title="ATM IV History — no trade_date column")

    dates      = sorted(full_df["trade_date"].unique())
    has_pe     = _has_svi_pe(full_df)
    atm_cols   = [
        ("implied_volatility",        "BS IV (ATM)",         _C["bs_raw"],  "markers+lines", None),
        ("implied_volatility_svi",    "SSVI (ATM)",          _C["ssvi"],    "lines",          None),
    ]
    if has_pe:
        atm_cols.append(("implied_volatility_svi_pe", "SVI pe (ATM)", _C["svi_pe"], "lines", "dash"))

    # For each date, find nearest-expiry ATM IV using PCHIP interpolation at k=0
    records = {col: {"dates": [], "iv": []} for col, *_ in atm_cols}

    for d in dates:
        day_df = full_df[full_df["trade_date"] == d]
        # pick nearest expiry (smallest T)
        expiries = sorted(day_df["time_to_expiry"].unique())
        if not expiries:
            continue
        T = expiries[0]
        sl = day_df[np.isclose(day_df["time_to_expiry"], T)].sort_values("log_moneyness")
        for col, *_ in atm_cols:
            if col not in sl.columns:
                continue
            d_col = sl[sl[col].notna() & (sl[col] > 0)]
            if len(d_col) < 3:
                continue
            k  = d_col["log_moneyness"].values
            iv = d_col[col].values
            _, ui = np.unique(k, return_index=True)
            k, iv = k[ui], iv[ui]
            if k.min() > 0 or k.max() < 0:
                continue
            atm_iv = float(PchipInterpolator(k, iv, extrapolate=False)(0.0))
            if not np.isnan(atm_iv):
                records[col]["dates"].append(d)
                records[col]["iv"].append(atm_iv)

    fig = go.Figure()
    for col, name, color, mode, dash in atm_cols:
        r = records[col]
        if not r["dates"]:
            continue
        fig.add_trace(go.Scatter(
            x=r["dates"], y=r["iv"],
            mode=mode, name=name,
            line=dict(color=color, width=2.5, **({} if dash is None else {"dash": dash})),
            marker=dict(size=8),
        ))

    fig.update_layout(
        title="ATM Implied Vol History — nearest expiry",
        xaxis_title="Trade Date", yaxis_title="ATM IV (%)",
        template="plotly_white", height=400,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-DATE: 25Δ SKEW HISTORY
# ─────────────────────────────────────────────────────────────────────────────
def plot_skew_history(full_df: pd.DataFrame) -> go.Figure:
    """
    25Δ Risk Reversal and Butterfly for SSVI (and SVI pe if available) across dates.
    Uses nearest expiry per date.
    """
    if "trade_date" not in full_df.columns:
        return go.Figure().update_layout(title="Skew History — no trade_date column")

    dates  = sorted(full_df["trade_date"].unique())
    has_pe = _has_svi_pe(full_df)

    surfaces = [("implied_volatility_svi", "SSVI", _C["ssvi"])]
    if has_pe:
        surfaces.append(("implied_volatility_svi_pe", "SVI pe", _C["svi_pe"]))

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["25Δ Risk Reversal  (put IV − call IV)",
                        "25Δ Butterfly  (wing avg − ATM)"],
        vertical_spacing=0.22, shared_xaxes=True,
    )

    for iv_col, label, color in surfaces:
        rr_dates, rr_vals = [], []
        bf_dates, bf_vals = [], []

        for d in dates:
            day_df   = full_df[full_df["trade_date"] == d]
            expiries = sorted(day_df["time_to_expiry"].unique())
            if not expiries:
                continue
            T  = expiries[0]
            sl = _clip_k(day_df[np.isclose(day_df["time_to_expiry"], T)].sort_values("log_moneyness"))
            if iv_col not in sl.columns:
                continue
            d_col = sl[sl[iv_col].notna() & (sl[iv_col] > 0)]
            smile  = d_col.groupby("log_moneyness")[iv_col].mean().reset_index().sort_values("log_moneyness")
            if len(smile) < 5:
                continue
            k  = smile["log_moneyness"].values
            iv = smile[iv_col].values / 100
            atm_iv = float(iv[np.argmin(np.abs(k))])
            k25p   = _delta_to_k(0.25, T, atm_iv)
            k25c   = -k25p
            pchip  = PchipInterpolator(k, iv, extrapolate=False)
            iv25p, iv25c = pchip(k25p), pchip(k25c)
            if np.isnan(iv25p) or np.isnan(iv25c):
                continue
            rr_dates.append(d);  rr_vals.append((iv25p - iv25c) * 100)
            bf_dates.append(d);  bf_vals.append((0.5 * (iv25p + iv25c) - atm_iv) * 100)

        for metric, r_dates, r_vals, row in [
            ("rr", rr_dates, rr_vals, 1),
            ("bf", bf_dates, bf_vals, 2),
        ]:
            if not r_dates:
                continue
            fig.add_trace(go.Scatter(
                x=r_dates, y=r_vals,
                mode="lines+markers", name=label,
                showlegend=(metric == "rr"),
                line=dict(color=color, width=2),
                marker=dict(size=6),
            ), row=row, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color=_C["zero"], row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=_C["zero"], row=2, col=1)
    fig.update_xaxes(title_text="Trade Date", row=2, col=1)
    fig.update_yaxes(title_text="Vol Pts", row=1, col=1)
    fig.update_yaxes(title_text="Vol Pts", row=2, col=1)
    fig.update_layout(
        title="25Δ Skew History — Risk Reversal & Butterfly",
        template="plotly_white", height=520,
        legend=dict(orientation="h", y=-0.12),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-DATE: VOL REGIME HISTORY
# ─────────────────────────────────────────────────────────────────────────────
def plot_vol_regime_history(full_df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart showing vol regime (low/medium/high) per trade date."""
    if "trade_date" not in full_df.columns or "vol_regime" not in full_df.columns:
        return go.Figure().update_layout(title="Vol Regime History — missing columns")

    regime_per_date = (
        full_df.dropna(subset=["vol_regime"])
        .groupby("trade_date")["vol_regime"]
        .first()
        .reset_index()
        .sort_values("trade_date")
    )

    regime_colors = {
        "low_vol":    "#43A047",
        "medium_vol": "#FF6F00",
        "high_vol":   "#E53935",
    }
    fig = go.Figure()
    for regime, color in regime_colors.items():
        sub = regime_per_date[regime_per_date["vol_regime"] == regime]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub["trade_date"], y=[1] * len(sub),
            name=regime.replace("_", " ").title(),
            marker_color=color,
        ))

    fig.update_layout(
        barmode="stack",
        title="Vol Regime by Trade Date",
        xaxis_title="Trade Date", yaxis_title="",
        yaxis=dict(showticklabels=False),
        template="plotly_white", height=220,
        legend=dict(orientation="h", y=-0.25),
    )
    return fig
