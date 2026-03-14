"""
Configuration loader.
Merges default_config.yaml → config/config.yaml (if present) → environment vars.

Environment variable overrides follow the pattern:
  AUTO_TRADER_<SECTION>_<KEY>
e.g. AUTO_TRADER_LLM_MODEL=llama3:70b
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Sub-models ────────────────────────────────────────────────────────────────

class TradingConfig(BaseModel):
    mode: str = "paper"
    markets: str = "crypto"
    symbols: List[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    timeframes: Dict[str, str] = {"primary": "4h", "context": "1d", "entry": "1h"}
    loop_interval_seconds: int = 300
    max_open_positions: int = 3
    position_size_pct: float = 0.05


class RiskConfig(BaseModel):
    max_portfolio_drawdown_pct: float = 0.15
    max_daily_loss_pct: float = 0.05
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    trailing_stop: bool = False
    trailing_stop_pct: float = 0.02
    min_risk_reward_ratio: float = 1.5
    max_correlated_positions: int = 2


class LLMConfig(BaseModel):
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "deepseek-r1:7b"
    fallback_model: str = "llama3:8b"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout_seconds: int = 90
    chain_of_thought: bool = True
    structured_output: bool = True
    # Vision / chart agent
    vision_model: str = "llava:7b"       # ollama pull llava:7b  (or moondream:latest)
    vision_enabled: bool = True          # set false to skip chart rendering entirely


class ExchangeConfig(BaseModel):
    id: str = "binance"
    sandbox: bool = True
    api_key: str = ""
    api_secret: str = ""
    rate_limit_ms: int = 500


class DataConfig(BaseModel):
    ohlcv_limit: int = 200
    cache_ttl_seconds: int = 60
    news_enabled: bool = False
    news_api_key: str = ""
    sentiment_lookback_hours: int = 24


class SymbolSelectorConfig(BaseModel):
    min_volume_usdt_24h: float = 50_000_000
    min_atr_pct: float = 0.01
    max_atr_pct: float = 0.15
    top_n: int = 10


class QuantConfig(BaseModel):
    indicators: List[str] = ["rsi", "macd", "bbands", "ema_20", "ema_50", "ema_200", "atr", "volume_ma", "stochastic", "adx"]
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    adx_period: int = 14
    adx_trend_threshold: float = 25


class DecisionConfig(BaseModel):
    min_confidence: float = 0.65
    quant_weight: float = 0.35
    trend_weight: float = 0.15
    setup_weight: float = 0.20
    trigger_weight: float = 0.10
    sentiment_weight: float = 0.10
    vision_weight: float = 0.10          # chart vision agent (0.0 = disabled)
    adversarial_veto: bool = True
    consensus_threshold: float = 0.60


class MonitoringConfig(BaseModel):
    log_level: str = "INFO"
    log_file: str = "./logs/trader.log"
    db_path: str = "./data/trades.db"
    metrics_enabled: bool = True
    metrics_port: int = 8000
    dashboard_refresh_seconds: int = 10
    performance_window_days: int = 30


# ── Root settings ─────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AUTO_TRADER_",
        env_nested_delimiter="_",
        case_sensitive=False,
    )

    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    symbol_selector: SymbolSelectorConfig = Field(default_factory=SymbolSelectorConfig)
    quant: QuantConfig = Field(default_factory=QuantConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        raw: Dict[str, Any] = {}
        # Load defaults first
        defaults_path = Path(__file__).parent / "default_config.yaml"
        if defaults_path.exists():
            with open(defaults_path) as f:
                raw = yaml.safe_load(f) or {}
        # Layer user config on top
        user_path = Path(path)
        if user_path.exists():
            with open(user_path) as f:
                user_cfg = yaml.safe_load(f) or {}
            raw = _deep_merge(raw, user_cfg)
        return cls(**raw)


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings, reading config/config.yaml if present."""
    cfg_path = Path(__file__).parent / "config.yaml"
    return Settings.from_yaml(cfg_path)
