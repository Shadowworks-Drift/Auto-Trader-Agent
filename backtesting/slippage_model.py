"""
Realistic slippage and fee model for backtesting.

Key insight (Two Sigma / Jane Street methodology): a strategy that looks
profitable before costs often disappears after.  This model encodes:

  • Exchange taker/maker fees
  • Bid-ask spread estimate as a fraction of price
  • Market impact (Almgren-Chriss square-root law) for large orders
  • Slippage that scales with volatility (ATR) in illiquid conditions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class FeeModel:
    """Exchange fee schedule."""
    taker_pct: float = 0.001    # 0.10% taker
    maker_pct: float = 0.0005   # 0.05% maker
    funding_per_8h: float = 0.0001  # perpetual funding (default ~10bps/8h)

    PRESETS: Dict[str, "FeeModel"] = None  # populated below

    @classmethod
    def for_exchange(cls, name: str) -> "FeeModel":
        presets = {
            "binance":   cls(taker_pct=0.001,  maker_pct=0.0005),
            "coinbase":  cls(taker_pct=0.006,  maker_pct=0.004),
            "kraken":    cls(taker_pct=0.0026, maker_pct=0.0016),
            "bybit":     cls(taker_pct=0.00055, maker_pct=0.0002),
            "okx":       cls(taker_pct=0.001,  maker_pct=0.0008),
            "paper":     cls(taker_pct=0.001,  maker_pct=0.001),
        }
        return presets.get(name.lower(), cls())


@dataclass
class SlippageModel:
    """
    Realistic slippage model.

    Three components:
      1. Half bid-ask spread (fixed by asset class)
      2. Market-impact term: k * sqrt(order_size / avg_daily_volume)
         — Almgren-Chriss square-root law
      3. Volatility-scaled extra slippage for illiquid assets
    """
    spread_pct: float = 0.0003       # half spread estimate (0.03% for BTC)
    impact_coeff: float = 0.1         # Almgren-Chriss k
    vol_scaling: bool = True          # scale impact by realised vol

    ASSET_PRESETS: Dict[str, "SlippageModel"] = None  # populated below

    @classmethod
    def for_asset(cls, symbol: str) -> "SlippageModel":
        base = symbol.split("/")[0].upper()
        liquid  = {"BTC", "ETH"}
        midcap  = {"SOL", "BNB", "AVAX", "MATIC", "DOT", "ADA", "LINK"}
        if base in liquid:
            return cls(spread_pct=0.0002, impact_coeff=0.05)
        elif base in midcap:
            return cls(spread_pct=0.0008, impact_coeff=0.15)
        else:
            return cls(spread_pct=0.002, impact_coeff=0.30)

    def cost(
        self,
        price: float,
        order_value: float,
        avg_daily_volume_usd: float,
        atr_pct: float = 0.02,
        is_taker: bool = True,
    ) -> float:
        """Return total cost as a fraction of trade value."""
        # 1. Spread
        spread_cost = self.spread_pct

        # 2. Market impact (Almgren-Chriss sqrt law)
        participation = order_value / max(avg_daily_volume_usd * 0.20, abs(order_value))
        impact = self.impact_coeff * (max(0.0, participation) ** 0.5)

        # 3. Vol scaling
        vol_mult = (atr_pct / 0.02) if self.vol_scaling else 1.0
        impact *= min(vol_mult, 3.0)

        return spread_cost + impact

    def apply(
        self,
        price: float,
        direction: str,
        order_value: float,
        avg_daily_volume_usd: float = 10_000_000,
        atr_pct: float = 0.02,
    ) -> float:
        """Return the execution price after slippage."""
        cost_frac = self.cost(price, order_value, avg_daily_volume_usd, atr_pct)
        if direction == "long":
            return price * (1 + cost_frac)   # pay up to buy
        else:
            return price * (1 - cost_frac)   # accept less to sell
