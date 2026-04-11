import logging
import os
import pandas as pd
from src.config.settings import Settings

logger = logging.getLogger(__name__)

# All symbol-level IV surface files live here
_IV_SURFACE_DIR = os.path.join(Settings.DATA_DIR, "processed", "iv_surfaces")


def _symbol_path(symbol: str) -> str:
    """Return the canonical parquet path for a symbol's IV surface store."""
    return os.path.join(_IV_SURFACE_DIR, f"{symbol}.parquet")


class ParquetSaver:

    @staticmethod
    def upsert_impliedvolsurface(
        df: pd.DataFrame,
        trade_date: str,
        symbol: str = None,
    ) -> str:
        """
        Upsert the calibrated surface for (symbol, trade_date) into the
        single per-symbol parquet at processed/iv_surfaces/{symbol}.parquet.

        Behaviour
        ---------
        - Stamps a `trade_date` column on every row before writing.
        - Drops any existing rows for that date (handles both new inserts and
          force-recompute overwrites — caller decides when to call).
        - Appends the new rows and re-sorts by (trade_date, time_to_expiry,
          log_moneyness) so the file stays in a predictable order.

        Returns
        -------
        str — path to the symbol parquet file.
        """
        sym = symbol or Settings.DEFAULT_SYMBOL
        os.makedirs(_IV_SURFACE_DIR, exist_ok=True)
        path = _symbol_path(sym)

        df = df.copy()
        df["trade_date"] = trade_date

        if os.path.exists(path):
            existing = pd.read_parquet(path)
            # Remove stale rows for this date before appending the fresh calibration
            if "trade_date" in existing.columns:
                existing = existing[existing["trade_date"] != trade_date]
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df

        combined = (
            combined
            .sort_values(["trade_date", "time_to_expiry", "log_moneyness"])
            .reset_index(drop=True)
        )
        combined.to_parquet(path, index=False)
        n_dates = combined["trade_date"].nunique() if "trade_date" in combined.columns else 1
        logger.info(
            "[Storage] IV surface upserted — symbol=%s  date=%s  "
            "total_dates_in_file=%d  rows=%d  path=%s",
            sym, trade_date, n_dates, len(combined), path,
        )
        return path

    @staticmethod
    def read_surface_for_date(
        symbol: str,
        trade_date: str,
    ) -> pd.DataFrame | None:
        """
        Return the surface rows for a single (symbol, trade_date) from the
        per-symbol parquet, or None if the file / date is not present.
        """
        path = _symbol_path(symbol or Settings.DEFAULT_SYMBOL)
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        if "trade_date" not in df.columns:
            return None
        slice_df = df[df["trade_date"] == trade_date].reset_index(drop=True)
        return slice_df if not slice_df.empty else None

    @staticmethod
    def save_impliedvols_to_parquet(
        df: pd.DataFrame, trade_date: str, symbol: str = None
    ) -> str:
        sym = symbol or Settings.DEFAULT_SYMBOL
        output_dir = os.path.join(Settings.DATA_DIR, f"processed/{sym}/{trade_date}")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "ivs.parquet")
        df.to_parquet(path, index=False)
        logger.info("[Storage] Implied vols parquet saved: %s/ivs.parquet", trade_date)
        return path
