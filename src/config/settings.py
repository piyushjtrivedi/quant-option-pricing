import os
from datetime import datetime


class Settings:
    # ── NSE data URLs ─────────────────────────────────────────────────────────
    BHAVCOPY_URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_"

    # ── Default symbol / instrument ───────────────────────────────────────────
    DEFAULT_SYMBOL          = "BANKNIFTY"
    DEFAULT_INSTRUMENT_TYPE = "ID"

    # ── Minimum open interest filter ─────────────────────────────────────────
    MIN_OI_INDEX = 1000
    MIN_OI_STOCK = 500

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
    )

    @staticmethod
    def min_oi(instrument_type: str) -> int:
        return Settings.MIN_OI_INDEX if instrument_type == "ID" else Settings.MIN_OI_STOCK

    @staticmethod
    def make_fo_filename(date: datetime) -> str:
        return f"fo{date.strftime('%d%b%Y').upper()}.zip"

    @staticmethod
    def make_fo_url(date: datetime) -> str:
        payload = f"{date.strftime('%Y%m%d')}_F_0000.csv.zip"
        return Settings.BHAVCOPY_URL + payload
