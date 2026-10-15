"""Minimal local structured logging for operational troubleshooting."""

import logging
from logging.handlers import RotatingFileHandler
import config


def get_logger(name: str = "customer_support_assistant") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RotatingFileHandler(config.LOG_DIR / "application.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
