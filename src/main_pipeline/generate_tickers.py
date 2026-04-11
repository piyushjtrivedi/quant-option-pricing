from pathlib import Path
import pandas as pd
from src.config.settings import Settings


def generate_ticker_df() -> pd.DataFrame:
    """
    Reads the latest BhavCopy CSV in data/raw/ and returns a DataFrame of
    unique (TckrSymb, FinInstrmTp) pairs where FinInstrmTp ends with 'O'
    (IDO = index options, STO = single-stock options).

    File selection: prefers the new-format filename
      BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv
    with fallback to the legacy foDD MON YYYY.csv pattern.
    The latest file is chosen by the date embedded in the filename;
    mtime is used as a tiebreaker when the date cannot be parsed.

    Returns
    -------
    pd.DataFrame
        Columns: TckrSymb (str), FinInstrmTp (str), instrument_type (str)
        instrument_type is FinInstrmTp with the trailing 'O' stripped
        ("IDO" → "ID", "STO" → "ST") — the value the pipeline expects.
        Sorted by TckrSymb ascending.
    """
    raw_dir = Path(Settings.DATA_DIR) / "raw"

    # New-format files carry a sortable YYYYMMDD date in the name
    candidates = sorted(
        raw_dir.glob("BhavCopy_NSE_FO_0_0_0_*_F_0000.csv"),
        key=lambda p: p.stem.split("_")[6] if len(p.stem.split("_")) > 6 else "0",
        reverse=True,
    )

    if not candidates:
        # Legacy format — sort by mtime
        candidates = sorted(
            raw_dir.glob("fo*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    if not candidates:
        raise FileNotFoundError(
            f"No BhavCopy CSV files found in {raw_dir}. "
            "Download at least one file before using the dynamic ticker list."
        )

    latest = candidates[0]

    df = pd.read_csv(latest, usecols=["TckrSymb", "FinInstrmTp"])
    df = df[df["FinInstrmTp"].str.endswith("O", na=False)].copy()
    df = df[["TckrSymb", "FinInstrmTp"]].drop_duplicates()

    # Derive the instrument type string the pipeline expects ("ID" or "ST")
    df["instrument_type"] = df["FinInstrmTp"].str[:-1]

    df = df.sort_values("TckrSymb").reset_index(drop=True)
    return df
