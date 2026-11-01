import logging
import pandas as pd
import numpy as np
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class BhavcopyParser:
    def __init__(self, symbol, instrument_type):
        self.symbol = symbol
        self.instrument_type = instrument_type
        # Raw CSV cache: one disk read per path
        self._csv_path: str | None = None
        self._raw_df: pd.DataFrame | None = None

    # ── internal helpers ──────────────────────────────────────────────────────

    def _read_raw(self, csv_path: str) -> pd.DataFrame:
        """Read and cache the raw BhavCopy CSV (one disk read per path)."""
        if self._csv_path != csv_path:
            logger.info("[BhavcopyParser] Reading: %s", csv_path)
            self._raw_df = pd.read_csv(csv_path)
            self._csv_path = csv_path
        return self._raw_df

    def _filter_futures(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Return rows for this symbol's futures contracts."""
        return raw[
            (raw["TckrSymb"] == self.symbol) &
            (raw["FinInstrmTp"] == f"{self.instrument_type}F")
        ]

    # ── public API ────────────────────────────────────────────────────────────

    def parse_to_dataframe(self, csv_path: str) -> pd.DataFrame:
        raw = self._read_raw(csv_path)

        # ── Futures close prices ──────────────────────────────────────────────
        f_df = self._filter_futures(raw)[["XpryDt", "ClsPric"]].rename(
            columns={"ClsPric": "futures_close"}
        )

        # ── Options chain ─────────────────────────────────────────────────────
        df = raw[
            (raw["TckrSymb"] == self.symbol) &
            (raw["FinInstrmTp"] == f"{self.instrument_type}O") &
            (raw["OpnIntrst"] >= Settings.min_oi(self.instrument_type))
        ].copy()

        df = df.merge(f_df, on=["XpryDt"], how="left")

        # No futures → no forward → log(K/F) undefined. Typically illiquid
        # quarterly expiries that would corrupt the surface fit.
        n_before = len(df)
        df = df[df["futures_close"].notna()]
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info(
                "[BhavcopyParser] Dropped %d contracts with no associated futures expiry",
                n_dropped,
            )

        df["time_to_expiry"] = (
            (pd.to_datetime(df["XpryDt"]) - pd.to_datetime(df["TradDt"])).dt.days / 365
        )
        df["log_moneyness"] = np.log(df["StrkPric"] / df["futures_close"])
        df["pay_off"] = np.where(
            df["OptnTp"] == "CE",
            np.maximum(0, df["UndrlygPric"].astype(float) - df["StrkPric"].astype(float)),
            np.maximum(0, df["StrkPric"].astype(float)    - df["UndrlygPric"].astype(float)),
        )
        df["cost_of_carry"] = (
            np.log(df["futures_close"] / df["UndrlygPric"]) / df["time_to_expiry"]
        )

        df = df[[
            "TckrSymb", "TradDt", "XpryDt", "StrkPric", "OptnTp",
            "ClsPric", "TtlTradgVol", "OpnIntrst", "UndrlygPric",
            "time_to_expiry", "log_moneyness",
            "pay_off", "futures_close", "cost_of_carry",
        ]].rename(columns={
            "TckrSymb":    "ticker",
            "TradDt":      "trade_date",
            "XpryDt":      "expiry",
            "StrkPric":    "strike",
            "OptnTp":      "option_type",
            "ClsPric":     "close",
            "TtlTradgVol": "volume",
            "OpnIntrst":   "open_interest",
            "UndrlygPric": "underlying_value",
        })

        df["strike"] = df["strike"].astype(float)

        logger.info("[BhavcopyParser] Parsed %d %s contracts.", len(df), self.symbol)
        return df

    def parse_to_futures_df(self, csv_path: str) -> pd.DataFrame:
        raw = self._read_raw(csv_path)  # uses cache if already read

        f_df = self._filter_futures(raw)[
            ["TradDt", "XpryDt", "ClsPric", "UndrlygPric"]
        ].copy()
        f_df["time_to_expiry"] = (
            (pd.to_datetime(f_df["XpryDt"]) - pd.to_datetime(f_df["TradDt"])).dt.days / 365
        )
        f_df["cost_of_carry"] = (
            np.log(f_df["ClsPric"] / f_df["UndrlygPric"]) / f_df["time_to_expiry"]
        )
        return f_df
