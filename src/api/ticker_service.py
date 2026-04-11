import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Fallback symbol list used when no BhavCopy file is available
_FALLBACK_SYMBOLS = pd.DataFrame(
    [
        {"TckrSymb": "BANKNIFTY", "FinInstrmTp": "IDO", "instrument_type": "ID"},
        {"TckrSymb": "NIFTY",     "FinInstrmTp": "IDO", "instrument_type": "ID"},
        {"TckrSymb": "FINNIFTY",  "FinInstrmTp": "IDO", "instrument_type": "ID"},
        {"TckrSymb": "RELIANCE",  "FinInstrmTp": "STO", "instrument_type": "ST"},
        {"TckrSymb": "INFY",      "FinInstrmTp": "STO", "instrument_type": "ST"},
    ]
)


def fetch_ticker_data() -> pd.DataFrame:
    """
    Returns a DataFrame of unique (TckrSymb, FinInstrmTp, instrument_type) pairs
    sourced from the latest available BhavCopy CSV in data/raw/.

    Falls back to a hardcoded list if no file is found or parsing fails,
    so the dashboard always has something to show.

    Columns
    -------
    TckrSymb        : NSE ticker symbol
    FinInstrmTp     : raw instrument type code ("IDO" or "STO")
    instrument_type : pipeline-ready code ("ID" or "ST")
    """
    try:
        from src.main_pipeline.generate_tickers import generate_ticker_df
        df = generate_ticker_df()
        logger.info("[TickerService] Loaded %d tickers from latest BhavCopy", len(df))
        return df
    except Exception as exc:
        logger.warning(
            "[TickerService] Could not load tickers from BhavCopy (%s). "
            "Using fallback symbol list.",
            exc,
        )
        return _FALLBACK_SYMBOLS.copy()
