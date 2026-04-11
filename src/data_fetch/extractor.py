import logging
import zipfile
import os

logger = logging.getLogger(__name__)


class BhavcopyExtractor:
    def extract(self, zip_path: str) -> str:
        extract_dir = os.path.dirname(zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
            extracted_files = zf.namelist()

        csv_path = os.path.join(extract_dir, extracted_files[0])
        logger.info("[Extractor] Extracted CSV: %s", csv_path)
        return csv_path
