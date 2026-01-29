import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

# ======================================================
# LOG DIRECTORY - Moved to Home to avoid Dev Server Reloads
# ======================================================
LOG_DIR = Path.home() / "medverax_logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"medverax_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# ======================================================
# LOGGER CONFIG
# ======================================================
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=5_000_000,   # 5 MB
    backupCount=5
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# ======================================================
# LOGGER FACTORY
# ======================================================
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.propagate = False

    return logger
