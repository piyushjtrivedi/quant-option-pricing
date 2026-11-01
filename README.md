# quantOptionPricing

End-of-day implied-volatility surfaces for NSE options (BANKNIFTY): Black-Scholes IV → SSVI and per-slice SVI → Dupire local vol, with a Streamlit dashboard.

```
NSE bhavcopy + RBI MIBOR → rates & dividends → BS IV → quality flags
    → SSVI ┐
    → SVI  ┴→ Dupire local vol → validate → Parquet → dashboard
```

## Quick start

```bash
python -m venv venv && source venv/bin/activate      # Python 3.12+
pip install -r requirements.txt

streamlit run frontend/app.py                         # demo dashboard (committed data)
python -m src.main_pipeline.run_calibration 2026-03-10 BANKNIFTY ID   # calibrate one date
```

From Python, `ensure_iv_surface(date, symbol, instrument)` in `src/api/iv_surface_service.py` does the same but skips dates already stored.

## Data

| Source | What | How |
|---|---|---|
| NSE F&O bhavcopy | Option and futures closes, OI | Downloaded automatically into `data/raw/` if missing |
| RBI MIBOR | 14-day, 1-month and 3-month rates | Excel downloaded manually into `data/raw/`; cached in `data/mibor/mibor.csv` |

The committed `data/processed/iv_surfaces/BANKNIFTY.parquet` holds 17 dates (29 Dec 2025 – 17 Mar 2026).

## Output

One row per (expiry, log-moneyness) grid point per date. Rows that match a traded contract also carry market data. Main columns:

| Column | Meaning |
|---|---|
| `implied_volatility` | Market (BS) IV, % |
| `implied_volatility_svi` / `_svi_pe` | SSVI / per-slice SVI IV, % |
| `dupire_local_volatility` / `_pe` | Local vol from each model, % |
| `implied_variance_svi`, `dw_dk_svi`, `d2w_dk2_svi`, `dw_dT_svi` | Total variance and its derivatives |
| `ssvi_rmse_vpts`, `svi_rmse_vpts`, `smile_stress_flag` | Date-level fit metrics |
| `vol_regime`, `is_clean`, `pcp_forward_ok`, … | Data-quality flags |

## Configuration

`src/config/settings.py`: `DEFAULT_SYMBOL` (`BANKNIFTY`), `MIN_OI_INDEX` (1000), `MIN_OI_STOCK` (500).
The fit-zone cap |k| ≤ 0.12 is `K_MAX` in `bs_fitter.py`. The dashboard mode is `DEMO_MODE` in `frontend/dashboard.py`.

## Layout

```
src/
  data_fetch/      NSE download, ZIP extract, MIBOR lookup
  parsers/         bhavcopy → option chain + futures; MIBOR rows
  pricing_engines/ BS pricer, NSS curve, dividends, risk-neutral density, forward variance
    iv_engines/    bs_engine (IV solver), svi_engine (SSVI, per-slice SVI)
  utils/           data quality, fit validation, Parquet storage
  main_pipeline/   run_calibration.py — the 10-step pipeline
  api/             cache-aware entry point, ticker list
frontend/          Streamlit app, dashboard, Plotly charts
documentations/    flow.md (what runs when) · design_choices.md (why)
```

## Known limitations

- **SSVI is structurally limited:** with one global φ and ρ, the fit pins ρ ≈ −1, and its RMSE runs 1.5–3 vol points on normal days.
- **Per-slice SVI arbitrage:** the calendar penalty is too weak to stop calendar arbitrage (present on 10 of 17 dates).
- **Quality flags don't filter the fit:** they are stored but not applied. The stale check never runs, because no previous day is passed in.
- **Last day before expiry:** an expiry 1 day away isn't excluded and wrecks the fit (29 Dec, 23 Feb).
- **Unwired modules:** `risk_neutral_density.py` and `forward_variance.py` are verified but not yet used by the pipeline or dashboard.
