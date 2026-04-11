import logging
import os
from src.config.settings import Settings
from src.utils.storage import ParquetSaver

logger = logging.getLogger(__name__)


def ensure_iv_surface(
    trade_date: str,
    symbol:     str  = None,
    instrument: str  = None,
    force_recompute: bool = False,
    model: str = "ssvi",
) -> str | None:
    """
    Return the path to the per-symbol IV surface parquet, computing and
    upserting the requested date if it is missing or force_recompute is set.

    Storage layout
    --------------
    data/processed/iv_surfaces/{symbol}.parquet

    One file per symbol; rows are distinguished by a `trade_date` column.
    The dashboard reads this file and filters to the requested date itself.

    Logic
    -----
    1. If the symbol parquet exists AND contains rows for trade_date
       AND force_recompute is False  →  return the path (cache hit).
    2. Otherwise run the calibration pipeline, which upserts the new rows
       (replacing any existing rows for that date when force=True).
    3. Return the path on success, None on failure.
    """
    sym  = symbol     or Settings.DEFAULT_SYMBOL
    inst = instrument or Settings.DEFAULT_INSTRUMENT_TYPE

    # ── Cache check ───────────────────────────────────────────────────────────
    if not force_recompute:
        existing = ParquetSaver.read_surface_for_date(sym, trade_date)
        if existing is not None:
            from src.utils.storage import _symbol_path
            path = _symbol_path(sym)
            logger.info(
                "[Service] Cache hit — %s rows for %s %s in %s",
                len(existing), sym, trade_date, path,
            )
            return path

    # ── Calibrate and upsert ──────────────────────────────────────────────────
    logger.info("[Service] Computing IV surface for %s %s %s ...", trade_date, sym, inst)
    try:
        from src.main_pipeline.run_calibration import calibrate
        path = calibrate(trade_date, sym, inst)
        return path
    except Exception as e:
        logger.error("[Service] Calibration failed: %s", e, exc_info=True)
        return None
