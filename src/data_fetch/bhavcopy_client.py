import logging
import os
import requests
from datetime import datetime
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class BhavcopyClient:
    def __init__(self):
        self.data_dir = os.path.join(Settings.DATA_DIR, "raw")
        os.makedirs(self.data_dir, exist_ok=True)

    def download(self, date: datetime) -> str:
        url = Settings.make_fo_url(date)
        filename = Settings.make_fo_filename(date)
        filepath = os.path.join(self.data_dir, filename)

        logger.info("[BhavcopyClient] Downloading: %s", url)

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nseindia.com",
        }

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Failed to download FO Bhavcopy ({response.status_code}).")

        with open(filepath, "wb") as f:
            f.write(response.content)

        logger.info("[BhavcopyClient] Saved: %s", filepath)
        return filepath
