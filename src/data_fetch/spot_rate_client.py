import logging
import os
import re
import pandas as pd
from datetime import datetime, date as date_type
from pathlib import Path
from src.config.settings import Settings

logger = logging.getLogger(__name__)

# ── RBI Excel filename ────────────────────────────────────────────────────────
# The RBI publishes a historical MIBOR/MIBID Excel file on data.rbi.org.in.
# Downloading it programmatically requires an authenticated SAP BusinessObjects
# session (enterprise login). Until that is implemented, place the manually
# downloaded file in data/raw/ with the exact filename below.
# Future: _download_rbi_excel() will use Playwright to log in and trigger the
# export, saving it to this same path automatically.
_RBI_EXCEL_FILENAME = (
    "Mumbai Inter-Bank Offer Rate (MIBOR) & Mumbai Inter-Bank Bid Rate (MIBID).xlsx"
)

# ── Excel layout constants ────────────────────────────────────────────────────
# Row indices (0-based) in the raw sheet
_HEADER_ROW_TENOR  = 3   # "OVERNIGHT", "14 DAYS", "1 MONTHS", "3 MONTHS"
_HEADER_ROW_TYPE   = 4   # "MIBID" / "MIBOR"
_HEADER_ROW_FIELD  = 5   # "Rate" / "Standard Deviation"
_DATA_START_ROW    = 6   # first data row
_DATE_COL          = 1   # column index holding the date


class SpotRateClient:
    """
    Provides MIBOR interest rates for a given date using a two-tier lookup:

      1. Local  — consolidated mibor.csv (data/mibor/mibor.csv).
                  All successful fetches are written here; most lookups hit here.
      2. RBI Excel — reads the RBI historical Excel from data/raw/ and bulk-loads
                     all MIBOR tenor rates into mibor.csv.  Place the manually
                     downloaded file at data/raw/<_RBI_EXCEL_FILENAME>.

    Columns stored: displayTime | tenor | years | yield
    Sorted by: displayTime ASC, years ASC
    """

    def __init__(self):
        self._mibor_dir  = os.path.join(Settings.DATA_DIR, "mibor")
        os.makedirs(self._mibor_dir, exist_ok=True)
        self._mibor_file = os.path.join(self._mibor_dir, "mibor.csv")
        self._raw_dir    = os.path.join(Settings.DATA_DIR, "raw")

    # ── public API ────────────────────────────────────────────────────────────

    def get_rates_for_date(self, target_date: datetime) -> pd.DataFrame:
        """
        Return MIBOR rates for *target_date* as a DataFrame with columns
        [displayTime, tenor, years, yield].

        Lookup order:
          1. Local mibor.csv  → return immediately if date present
          2. RBI Excel (bulk) → parse Excel, merge all into mibor.csv, re-check
        """
        key = self._as_date(target_date)

        # ── Tier 1: local file ────────────────────────────────────────────────
        result = self._lookup_local(key)
        if result is not None:
            logger.info("[SpotRateClient] Tier-1 hit for %s", key)
            return result

        # ── Tier 2: RBI Excel bulk load ───────────────────────────────────────
        logger.info("[SpotRateClient] Tier-1 miss — loading RBI Excel for %s", key)
        try:
            excel_df = self._load_rbi_excel()
            if not excel_df.empty:
                self._merge_into_file(excel_df)
                result = self._lookup_local(key)
                if result is not None:
                    logger.info("[SpotRateClient] Tier-2 hit — RBI Excel had %s", key)
                    return result
                logger.warning(
                    "[SpotRateClient] RBI Excel loaded (%d rows) but %s not present. "
                    "Ensure the Excel covers this date.",
                    len(excel_df), key,
                )
            else:
                logger.warning(
                    "[SpotRateClient] RBI Excel returned no data. "
                    "Ensure '%s' exists in data/raw/", _RBI_EXCEL_FILENAME
                )
        except Exception as exc:
            logger.warning("[SpotRateClient] RBI Excel load failed: %s", exc)

        logger.error("[SpotRateClient] No rates found for %s", key)
        return pd.DataFrame(columns=["displayTime", "tenor", "years", "yield"])

    def ingest_rbi_excel(self) -> int:
        """
        Public helper: parse the RBI Excel and merge all rows into mibor.csv.
        Returns the number of new rows added.
        Call this once after placing a freshly downloaded Excel in data/raw/.
        """
        excel_df = self._load_rbi_excel()
        if excel_df.empty:
            logger.warning("[SpotRateClient] ingest_rbi_excel: no data extracted")
            return 0

        before = len(self._read_file()) if os.path.exists(self._mibor_file) else 0
        self._merge_into_file(excel_df)
        after  = len(self._read_file())
        added  = after - before
        logger.info("[SpotRateClient] ingest_rbi_excel: +%d new rows (%d total)", added, after)
        return added

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _as_date(d):
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, date_type):
            return d
        return datetime.strptime(str(d), "%Y-%m-%d").date()

    def _lookup_local(self, key: date_type) -> pd.DataFrame | None:
        if not os.path.exists(self._mibor_file):
            return None
        df = self._read_file()
        mask = df["displayTime"] == key
        return df[mask].reset_index(drop=True) if mask.any() else None

    def _read_file(self) -> pd.DataFrame:
        df = pd.read_csv(self._mibor_file)
        df["displayTime"] = pd.to_datetime(df["displayTime"]).dt.date
        return df

    def _merge_into_file(self, new_df: pd.DataFrame) -> None:
        """
        Merge new_df into mibor.csv.
        Deduplicates on (displayTime, tenor), keeps last value.
        Final file is sorted by displayTime ASC, years ASC.
        """
        new_df = new_df[["displayTime", "tenor", "years", "yield"]].copy()
        new_df["displayTime"] = pd.to_datetime(new_df["displayTime"]).dt.date

        if os.path.exists(self._mibor_file):
            existing = self._read_file()
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["displayTime", "tenor"], keep="last")
        else:
            combined = new_df

        combined = combined.sort_values(
            ["displayTime", "years"],
            ascending=[True, True],
        ).reset_index(drop=True)

        combined.to_csv(self._mibor_file, index=False)
        logger.info("[SpotRateClient] mibor.csv updated — %d total rows", len(combined))

    def _excel_latest_date(self) -> date_type | None:
        """
        Return the latest date already ingested from the Excel into mibor.csv,
        without re-parsing the Excel.  Used to skip redundant re-parsing when
        the requested date is clearly beyond the Excel's coverage.
        Returns None if mibor.csv does not exist or is empty.
        """
        if not os.path.exists(self._mibor_file):
            return None
        df = self._read_file()
        return df["displayTime"].max() if not df.empty else None

    # ── Tier 2: RBI Excel parser ──────────────────────────────────────────────

    def _load_rbi_excel(self) -> pd.DataFrame:
        """
        Parse the RBI MIBOR/MIBID Excel file from data/raw/ and return a
        long-format DataFrame with columns [displayTime, tenor, years, yield]
        containing only MIBOR Rate values (MIBID is excluded).

        Excel layout (0-based row indices):
          Row 3 — tenor group headers: OVERNIGHT, 14 DAYS, 1 MONTHS, 3 MONTHS
          Row 4 — MIBID / MIBOR per tenor group
          Row 5 — Rate / Standard Deviation sub-headers
          Row 6+ — date (col 1) + rate data

        Only MIBOR Rate columns are extracted (MIBID and Std Dev are dropped).

        Future: replace this with _download_rbi_excel() once Playwright-based
        RBI portal login is implemented.
        """
        excel_path = Path(self._raw_dir) / _RBI_EXCEL_FILENAME
        if not excel_path.exists():
            logger.warning(
                "[SpotRateClient] RBI Excel not found at %s", excel_path
            )
            return pd.DataFrame(columns=["displayTime", "tenor", "years", "yield"])

        logger.info("[SpotRateClient] Parsing RBI Excel: %s", excel_path.name)

        raw = pd.read_excel(
            excel_path,
            sheet_name="Report 1",
            header=None,
            engine="openpyxl",
        )

        # ── Build column map: col_index → (tenor_label, field_type) ──────────
        # Row 3 has tenor group labels at the leftmost column of each group.
        # Row 4 has MIBID/MIBOR repeated within each group.
        # Row 5 has Rate/Standard Deviation alternating within each MIBID/MIBOR.
        # We propagate tenor and type labels forward to fill merged-cell gaps.

        tenor_row = raw.iloc[_HEADER_ROW_TENOR].tolist()
        type_row  = raw.iloc[_HEADER_ROW_TYPE].tolist()
        field_row = raw.iloc[_HEADER_ROW_FIELD].tolist()

        # Forward-fill tenor and type to cover merged cells (NaN = merged)
        current_tenor = None
        current_type  = None
        col_meta = {}   # col_idx → {"tenor": str, "type": str, "field": str}

        for col_idx in range(len(tenor_row)):
            t = tenor_row[col_idx]
            if pd.notna(t) and str(t).strip():
                current_tenor = str(t).strip()
            y = type_row[col_idx]
            if pd.notna(y) and str(y).strip():
                current_type = str(y).strip()
            f = field_row[col_idx]
            field_clean = str(f).strip().replace("\r\n", " ") if pd.notna(f) else ""

            if current_tenor and current_type and field_clean:
                col_meta[col_idx] = {
                    "tenor": current_tenor,
                    "type":  current_type,
                    "field": field_clean,
                }

        # ── Identify MIBOR Rate columns only ─────────────────────────────────
        mibor_rate_cols = {
            col_idx: meta
            for col_idx, meta in col_meta.items()
            if meta["type"].upper() == "MIBOR" and meta["field"].upper() == "RATE"
        }

        if not mibor_rate_cols:
            logger.error("[SpotRateClient] No MIBOR Rate columns found in Excel")
            return pd.DataFrame(columns=["displayTime", "tenor", "years", "yield"])

        logger.debug(
            "[SpotRateClient] MIBOR Rate columns: %s",
            {c: m["tenor"] for c, m in mibor_rate_cols.items()},
        )

        # ── Extract data rows ─────────────────────────────────────────────────
        data = raw.iloc[_DATA_START_ROW:].copy()
        rows_out = []

        for _, row in data.iterrows():
            raw_date = row.iloc[_DATE_COL]
            if pd.isna(raw_date):
                continue
            try:
                dt = pd.to_datetime(raw_date, errors="coerce")
                if pd.isna(dt):
                    continue
                display_date = dt.date()
            except Exception:
                continue

            for col_idx, meta in mibor_rate_cols.items():
                val = row.iloc[col_idx]
                if pd.isna(val):
                    continue
                try:
                    rate = float(val)
                except (ValueError, TypeError):
                    continue

                tenor_label = meta["tenor"]
                years       = self._tenor_to_years(tenor_label)

                rows_out.append({
                    "displayTime": display_date,
                    "tenor":       tenor_label,
                    "years":       years,
                    "yield":       rate,
                })

        if not rows_out:
            logger.warning("[SpotRateClient] RBI Excel parsed but no valid rows extracted")
            return pd.DataFrame(columns=["displayTime", "tenor", "years", "yield"])

        df = pd.DataFrame(rows_out)[["displayTime", "tenor", "years", "yield"]]
        df = df[df["tenor"].str.upper() != "OVERNIGHT"]
        df = df.sort_values(["displayTime", "years"], ascending=[True, True])
        logger.info(
            "[SpotRateClient] RBI Excel: extracted %d rows spanning %s → %s",
            len(df),
            df["displayTime"].min(),
            df["displayTime"].max(),
        )
        return df.reset_index(drop=True)

    # ── utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _tenor_to_years(tenor: str) -> float | None:
        """Convert a tenor string (e.g. '14 DAYS', '1 MONTHS') to year fraction."""
        if not isinstance(tenor, str):
            return None
        match = re.match(r"(\d+)\s*(DAY|DAYS|MONTH|MONTHS|YEAR|YEARS|OVERNIGHT)", tenor.upper().strip())
        if not match:
            # Handle bare "OVERNIGHT"
            if "OVERNIGHT" in tenor.upper():
                return 1 / 365.0
            return None
        value_str, unit = match.group(1), match.group(2)
        value = int(value_str)
        if "DAY"   in unit: return value / 365.0
        if "MONTH" in unit: return value / 12.0
        if "YEAR"  in unit: return float(value)
        return None
