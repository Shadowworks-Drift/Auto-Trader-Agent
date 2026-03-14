"""
General utility helpers.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


def format_pct(value: float, decimals: int = 2) -> str:
    return f"{value * 100:+.{decimals}f}%"


def format_price(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@contextmanager
def timer(label: str = "") -> Generator[None, None, None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - t0) * 1000
        from loguru import logger
        logger.debug(f"{label} took {elapsed:.1f}ms")
