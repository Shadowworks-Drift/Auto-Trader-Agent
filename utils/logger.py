"""
Loguru-based structured logging setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_level: str = "INFO", log_file: str = "./logs/trader.log") -> None:
    """Configure loguru for console + rotating file output."""
    logger.remove()  # Remove default handler

    # Console — colourful, concise
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{extra[agent]}</cyan> | "
            "{message}"
        ),
        colorize=True,
        filter=lambda record: True,
    )

    # File — JSON-structured, rotating
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra} | {message}",
        rotation="100 MB",
        retention="30 days",
        compression="gz",
        serialize=False,
    )

    # Bind default extra fields
    logger.configure(extra={"agent": "main"})
    logger.info(f"Logger initialised | level={log_level} | file={log_file}")
