import logging
import pandas as pd

logger = logging.getLogger(__name__)


class InterestRateParser:

    @staticmethod
    def parse_to_dataframe(csv_path: str, date_str: str = None) -> pd.DataFrame:
        """
        Read the consolidated mibor.csv and return interest rate rows.

        Args:
            csv_path:  Path to mibor.csv (single consolidated file).
            date_str:  ISO date string (YYYY-MM-DD) to filter rows.
                       If None, all rows are returned.

        Returns:
            DataFrame with at least columns [displayTime, tenor, years, yield].
        """
        logger.info("[InterestRateParser] Parsing: %s", csv_path)
        df = pd.read_csv(csv_path)
        df["displayTime"] = pd.to_datetime(df["displayTime"]).dt.date

        if date_str is not None:
            from datetime import datetime
            key = datetime.strptime(date_str, "%Y-%m-%d").date()
            df = df[df["displayTime"] == key].reset_index(drop=True)

        logger.info("[InterestRateParser] Parsed %d interest rate rows.", len(df))
        return df
