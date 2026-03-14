"""
Trading Gym Environment
════════════════════════
A Gymnasium-compatible environment for training RL position-sizing agents.

State space (11 features):
  [quant_score, trend_conf, setup_conf, trigger_conf, sentiment_score,
   vol_regime, funding_rate, fear_greed, portfolio_drawdown,
   daily_pnl_pct, open_position_count]

Action space: continuous [0, 1] representing position size fraction

Reward: iCVaR-shaped reward (SAPPO / MTS framework)
  r = PnL_pct / max(CVaR_95, 0.001)  — penalises tail losses

Based on:
  - SAPPO (NeurIPS 2025): Sentiment-Augmented PPO, Sharpe 2.07
  - MTS framework (arXiv 2503.04143): iCVaR reward shaping
  - FinRL Contest 2025: vectorised gym environments
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False


@dataclass
class TradingState:
    """Feature vector fed to the RL agent at each decision point."""
    quant_score: float         # −1 to +1 (QuantAnalyst composite)
    trend_confidence: float    # 0–1
    setup_confidence: float    # 0–1
    trigger_confidence: float  # 0–1
    sentiment_score: float     # −1 to +1
    vol_regime: float          # 0=low, 0.5=normal, 1=high, 1.5=extreme
    funding_rate: float        # normalised: funding_rate / 0.1
    fear_greed: float          # 0–1 (raw/100)
    portfolio_drawdown: float  # 0–1 (negative = loss)
    daily_pnl_pct: float       # −1 to +1
    open_positions: float      # 0–1 (positions / max_positions)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.quant_score,
            self.trend_confidence,
            self.setup_confidence,
            self.trigger_confidence,
            self.sentiment_score,
            self.vol_regime,
            np.clip(self.funding_rate, -3, 3),
            self.fear_greed,
            self.portfolio_drawdown,
            np.clip(self.daily_pnl_pct, -1, 1),
            self.open_positions,
        ], dtype=np.float32)

    STATE_DIM = 11

    @classmethod
    def from_agent_results(
        cls,
        quant_data: Dict[str, Any],
        trend_data: Dict[str, Any],
        setup_data: Dict[str, Any],
        trigger_data: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        portfolio_info: Dict[str, Any],
        vol_regime: float = 0.5,
    ) -> "TradingState":
        return cls(
            quant_score=float(quant_data.get("score", 0.0)),
            trend_confidence=float(trend_data.get("confidence", 0.0)),
            setup_confidence=float(setup_data.get("confidence", 0.0)),
            trigger_confidence=float(trigger_data.get("confidence", 0.0)),
            sentiment_score=float(sentiment_data.get("score", 0.0)),
            vol_regime=vol_regime,
            funding_rate=float(portfolio_info.get("funding_rate", 0.0)) / 0.1,
            fear_greed=float(portfolio_info.get("fear_greed", 50)) / 100.0,
            portfolio_drawdown=float(portfolio_info.get("drawdown", 0.0)),
            daily_pnl_pct=float(portfolio_info.get("daily_pnl_pct", 0.0)),
            open_positions=float(portfolio_info.get("open_count", 0)) / max(portfolio_info.get("max_positions", 3), 1),
        )


@dataclass
class _Episode:
    """Single training episode record."""
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    pnls: List[float] = field(default_factory=list)
    done: bool = False


class TradingEnv:
    """
    Gymnasium-compatible trading environment built on historical paper-trade records.

    The environment replays stored paper trade decisions from SQLite and asks
    the RL agent: 'given this state, what position size should I use?'

    Reward shaping (iCVaR):
      After each episode step:
        - Compute PnL for the action taken
        - reward = PnL_pct / CVaR_95(last N PnLs)
        - This forces the agent to care about tail risk, not just mean return
    """

    def __init__(
        self,
        episodes: List[Dict[str, Any]],
        max_position_size: float = 0.10,
        cvar_window: int = 20,
        cvar_lambda: float = 1.0,
    ) -> None:
        self.episodes = episodes
        self.max_position_size = max_position_size
        self.cvar_window = cvar_window
        self.cvar_lambda = cvar_lambda

        self._current_ep = 0
        self._step_idx   = 0
        self._pnl_history: List[float] = []
        self._current_episode: Optional[_Episode] = None

        if GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=-2.0, high=2.0,
                shape=(TradingState.STATE_DIM,),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )

    def reset(self) -> np.ndarray:
        self._step_idx = 0
        self._current_episode = _Episode()
        if self._current_ep >= len(self.episodes):
            self._current_ep = 0
        state = self._get_state()
        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action_val = float(np.clip(action[0] if hasattr(action, '__len__') else action, 0, 1))
        size_frac = action_val * self.max_position_size

        # Retrieve ground-truth PnL from this episode step
        ep = self.episodes[self._current_ep]
        steps = ep.get("steps", [])

        if self._step_idx >= len(steps):
            return self._get_state(), 0.0, True, {}

        step_data = steps[self._step_idx]
        raw_pnl_pct = float(step_data.get("pnl_pct", 0.0))

        # Scale by size fraction (larger position = more PnL and more risk)
        scaled_pnl = raw_pnl_pct * (size_frac / self.max_position_size)
        self._pnl_history.append(scaled_pnl)

        # iCVaR reward
        reward = self._icvar_reward(scaled_pnl)

        self._step_idx += 1
        done = self._step_idx >= len(steps)
        if done:
            self._current_ep += 1

        next_state = self._get_state() if not done else np.zeros(TradingState.STATE_DIM, dtype=np.float32)

        if self._current_episode:
            self._current_episode.rewards.append(reward)
            self._current_episode.pnls.append(scaled_pnl)

        return next_state, reward, done, {"pnl": scaled_pnl, "size": size_frac}

    def _icvar_reward(self, pnl: float) -> float:
        """iCVaR-shaped reward: r = PnL / max(CVaR_95, 0.001)"""
        if len(self._pnl_history) < 5:
            return pnl  # not enough history for CVaR
        window = self._pnl_history[-self.cvar_window:]
        sorted_pnls = sorted(window)
        n_tail = max(1, int(len(sorted_pnls) * 0.05))
        cvar = abs(np.mean(sorted_pnls[:n_tail]))
        reward = pnl / max(cvar, 0.001)
        return float(np.clip(reward * self.cvar_lambda, -10.0, 10.0))

    def _get_state(self) -> np.ndarray:
        ep = self.episodes[min(self._current_ep, len(self.episodes) - 1)]
        steps = ep.get("steps", [])
        idx = min(self._step_idx, len(steps) - 1)
        if not steps:
            return np.zeros(TradingState.STATE_DIM, dtype=np.float32)
        state_dict = steps[idx].get("state", {})
        state = TradingState(
            quant_score=state_dict.get("quant_score", 0.0),
            trend_confidence=state_dict.get("trend_confidence", 0.5),
            setup_confidence=state_dict.get("setup_confidence", 0.5),
            trigger_confidence=state_dict.get("trigger_confidence", 0.5),
            sentiment_score=state_dict.get("sentiment_score", 0.0),
            vol_regime=state_dict.get("vol_regime", 0.5),
            funding_rate=state_dict.get("funding_rate", 0.0),
            fear_greed=state_dict.get("fear_greed", 0.5),
            portfolio_drawdown=state_dict.get("drawdown", 0.0),
            daily_pnl_pct=state_dict.get("daily_pnl_pct", 0.0),
            open_positions=state_dict.get("open_positions", 0.0),
        )
        return state.to_array()

    @classmethod
    def from_sqlite(cls, db_path: str, **kwargs) -> "TradingEnv":
        """Build environment from historical paper trade SQLite database."""
        import asyncio
        import aiosqlite

        async def _load():
            async with aiosqlite.connect(db_path) as db:
                async with db.execute(
                    "SELECT pos_id, symbol, direction, entry_price, exit_price, "
                    "pnl, pnl_pct, exit_reason FROM trades ORDER BY closed_at"
                ) as cur:
                    rows = await cur.fetchall()
            return rows

        rows = asyncio.run(_load())
        if not rows:
            return cls(episodes=[], **kwargs)

        # Group into episode-like chunks of 10 trades
        chunk_size = 10
        episodes = []
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i + chunk_size]
            steps = []
            for row in chunk:
                steps.append({
                    "pnl_pct": float(row[7]),
                    "state": {
                        "quant_score": 0.5 if row[2] == "long" else -0.5,
                        "trend_confidence": 0.6,
                        "setup_confidence": 0.6,
                        "trigger_confidence": 0.6,
                        "sentiment_score": 0.0,
                        "vol_regime": 0.5,
                        "funding_rate": 0.0,
                        "fear_greed": 0.5,
                        "drawdown": 0.0,
                        "daily_pnl_pct": float(row[7]),
                        "open_positions": 0.3,
                    },
                })
            episodes.append({"steps": steps})

        return cls(episodes=episodes, **kwargs)
