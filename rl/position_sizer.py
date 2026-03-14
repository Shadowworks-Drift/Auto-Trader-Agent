"""
RL Position Sizer
═════════════════
Uses a trained PPO agent (or half-Kelly+CVaR fallback) to determine
the fraction of available capital to deploy for a given trade.

Two sizing strategies available:
  1. RLPositionSizer (PPO)    — primary, requires a trained model file
  2. HalfKellyCVaR (fallback) — analytic baseline, always available

References:
  - SAPPO (NeurIPS 2025): PPO + sentiment augmentation → Sharpe 2.07
  - arXiv 2508.16598: Half-Kelly with CVaR constraint
  - MTS framework (arXiv 2503.04143): iCVaR reward shaping
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from .environment import TradingState


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class SizerDecision:
    """Result from the position sizer."""
    position_size_pct: float      # fraction of capital, e.g. 0.05 = 5 %
    raw_action: float             # agent's raw [0,1] output before scaling
    method: str                   # "rl_ppo" | "half_kelly" | "fallback"
    cvar_95: float = 0.0          # CVaR-95 of recent PnL window
    kelly_fraction: float = 0.0   # Kelly fraction before halving
    confidence: float = 0.0       # agent confidence / certainty proxy
    elapsed_ms: float = 0.0


# ── Half-Kelly + CVaR analytic baseline ───────────────────────────────────────

class HalfKellyCVaR:
    """
    Analytic half-Kelly position sizing with CVaR floor constraint.

    Formula (from arXiv 2508.16598):
      kelly  = (win_rate * rr - (1 - win_rate)) / rr
      half_k = kelly / 2
      cvar_floor = max_drawdown_tolerance / cvar_95
      size = min(half_k, cvar_floor) * max_position_size
    """

    def __init__(
        self,
        max_position_size: float = 0.10,
        max_drawdown_tolerance: float = 0.02,
        cvar_window: int = 20,
        min_size: float = 0.005,
    ) -> None:
        self.max_position_size = max_position_size
        self.max_drawdown_tolerance = max_drawdown_tolerance
        self.cvar_window = cvar_window
        self.min_size = min_size
        self._pnl_history: List[float] = []

    def record_pnl(self, pnl_pct: float) -> None:
        self._pnl_history.append(pnl_pct)

    def size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        confidence: float = 0.5,
    ) -> SizerDecision:
        t0 = time.perf_counter()

        # Kelly formula
        rr = avg_win / max(avg_loss, 1e-6)
        kelly = (win_rate * rr - (1.0 - win_rate)) / max(rr, 1e-6)
        kelly = max(0.0, min(kelly, 1.0))
        half_kelly = kelly / 2.0

        # CVaR constraint
        cvar_95 = self._compute_cvar95()
        if cvar_95 > 0:
            cvar_floor = self.max_drawdown_tolerance / cvar_95
        else:
            cvar_floor = self.max_position_size

        raw_action = min(half_kelly, cvar_floor)
        size_pct = max(self.min_size, min(raw_action * self.max_position_size, self.max_position_size))

        return SizerDecision(
            position_size_pct=round(size_pct, 6),
            raw_action=round(raw_action, 4),
            method="half_kelly",
            cvar_95=round(cvar_95, 6),
            kelly_fraction=round(half_kelly, 4),
            confidence=confidence,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    def _compute_cvar95(self) -> float:
        if not self._pnl_history:
            return 0.0
        window = self._pnl_history[-self.cvar_window:]
        sorted_pnls = sorted(window)
        n_tail = max(1, int(len(sorted_pnls) * 0.05))
        return abs(float(np.mean(sorted_pnls[:n_tail])))


# ── PPO Agent (lightweight, no torch dependency required for inference) ────────

class _PolicyNetwork:
    """
    Simple 2-layer MLP policy stored as numpy arrays.
    Avoids hard torch dependency at inference time — weights saved as .npz.

    Architecture: Linear(11→64) → ReLU → Linear(64→32) → ReLU → Linear(32→1) → Sigmoid
    """

    def __init__(self, weights: Dict[str, np.ndarray]) -> None:
        self.W1 = weights["W1"]   # (64, 11)
        self.b1 = weights["b1"]   # (64,)
        self.W2 = weights["W2"]   # (32, 64)
        self.b2 = weights["b2"]   # (32,)
        self.W3 = weights["W3"]   # (1, 32)
        self.b3 = weights["b3"]   # (1,)

    @classmethod
    def random_init(cls, state_dim: int = 11) -> "_PolicyNetwork":
        rng = np.random.default_rng(42)
        weights = {
            "W1": rng.standard_normal((64, state_dim)).astype(np.float32) * 0.1,
            "b1": np.zeros(64, dtype=np.float32),
            "W2": rng.standard_normal((32, 64)).astype(np.float32) * 0.1,
            "b2": np.zeros(32, dtype=np.float32),
            "W3": rng.standard_normal((1, 32)).astype(np.float32) * 0.1,
            "b3": np.zeros(1, dtype=np.float32),
        }
        return cls(weights)

    def forward(self, x: np.ndarray) -> float:
        h1 = np.maximum(0, self.W1 @ x + self.b1)      # ReLU
        h2 = np.maximum(0, self.W2 @ h1 + self.b2)     # ReLU
        out = self.W3 @ h2 + self.b3
        return float(1.0 / (1.0 + np.exp(-out[0])))    # Sigmoid → [0,1]

    def save(self, path: Path) -> None:
        np.savez(
            str(path),
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
        )

    @classmethod
    def load(cls, path: Path) -> "_PolicyNetwork":
        data = np.load(str(path))
        return cls({k: data[k] for k in data.files})


# ── Main RL Position Sizer ────────────────────────────────────────────────────

class RLPositionSizer:
    """
    Position sizer backed by a trained PPO agent.

    Falls back to HalfKellyCVaR when:
      - No model file exists
      - State data is incomplete
      - Model produces an out-of-range action

    The PPO agent operates on TradingState (11 features) and outputs a
    continuous action in [0, 1] representing what fraction of max_position_size
    to deploy.

    Usage:
        sizer = RLPositionSizer.from_model(path="models/rl_sizer.npz")
        decision = sizer.decide(trading_state, portfolio_stats)
    """

    MODEL_DIR = Path("models")
    DEFAULT_MODEL_FILE = "rl_sizer.npz"
    METADATA_FILE = "rl_sizer_meta.json"

    def __init__(
        self,
        policy: Optional[_PolicyNetwork],
        max_position_size: float = 0.10,
        fallback: Optional[HalfKellyCVaR] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.policy = policy
        self.max_position_size = max_position_size
        self.fallback = fallback or HalfKellyCVaR(max_position_size=max_position_size)
        self.metadata = metadata or {}
        self._pnl_history: List[float] = []

    @classmethod
    def from_model(
        cls,
        path: Optional[str] = None,
        max_position_size: float = 0.10,
    ) -> "RLPositionSizer":
        model_path = Path(path) if path else cls.MODEL_DIR / cls.DEFAULT_MODEL_FILE
        meta_path = model_path.parent / cls.METADATA_FILE

        if not model_path.exists():
            logger.warning(
                f"RLPositionSizer: no model at {model_path}, using half-Kelly fallback"
            )
            return cls(policy=None, max_position_size=max_position_size)

        try:
            policy = _PolicyNetwork.load(model_path)
            metadata: Dict[str, Any] = {}
            if meta_path.exists():
                metadata = json.loads(meta_path.read_text())
            logger.info(
                f"RLPositionSizer: loaded model from {model_path} "
                f"(trained on {metadata.get('n_episodes', '?')} episodes, "
                f"avg_reward={metadata.get('avg_reward', '?')})"
            )
            return cls(policy=policy, max_position_size=max_position_size, metadata=metadata)
        except Exception as exc:
            logger.error(f"RLPositionSizer: failed to load model — {exc}")
            return cls(policy=None, max_position_size=max_position_size)

    @classmethod
    def untrained(cls, max_position_size: float = 0.10) -> "RLPositionSizer":
        """Create sizer with randomly-initialised weights (for bootstrapping)."""
        policy = _PolicyNetwork.random_init(TradingState.STATE_DIM)
        return cls(policy=policy, max_position_size=max_position_size)

    def record_pnl(self, pnl_pct: float) -> None:
        """Call after each trade closes to keep CVaR history current."""
        self._pnl_history.append(pnl_pct)
        self.fallback.record_pnl(pnl_pct)

    def decide(
        self,
        state: TradingState,
        portfolio_stats: Optional[Dict[str, Any]] = None,
    ) -> SizerDecision:
        """
        Compute position size from a TradingState.

        Args:
            state: 11-feature trading state
            portfolio_stats: optional dict with win_rate, avg_win, avg_loss
                             (used by fallback)
        """
        t0 = time.perf_counter()

        if self.policy is None:
            # No trained model — use half-Kelly
            return self._fallback_decide(state, portfolio_stats, t0)

        try:
            obs = state.to_array()
            raw_action = self.policy.forward(obs)

            # Sanity check
            if not (0.0 <= raw_action <= 1.0):
                raise ValueError(f"Policy returned out-of-range action: {raw_action}")

            size_pct = raw_action * self.max_position_size
            cvar_95 = self._compute_cvar95()

            return SizerDecision(
                position_size_pct=round(size_pct, 6),
                raw_action=round(raw_action, 4),
                method="rl_ppo",
                cvar_95=round(cvar_95, 6),
                kelly_fraction=0.0,
                confidence=round(raw_action, 4),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        except Exception as exc:
            logger.warning(f"RLPositionSizer.decide failed: {exc} — using fallback")
            return self._fallback_decide(state, portfolio_stats, t0)

    def decide_from_dict(
        self,
        quant_data: Dict[str, Any],
        trend_data: Dict[str, Any],
        setup_data: Dict[str, Any],
        trigger_data: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        portfolio_info: Dict[str, Any],
        vol_regime: float = 0.5,
    ) -> SizerDecision:
        """Convenience wrapper that builds TradingState from agent result dicts."""
        state = TradingState.from_agent_results(
            quant_data=quant_data,
            trend_data=trend_data,
            setup_data=setup_data,
            trigger_data=trigger_data,
            sentiment_data=sentiment_data,
            portfolio_info=portfolio_info,
            vol_regime=vol_regime,
        )
        return self.decide(state, portfolio_info)

    def _fallback_decide(
        self,
        state: TradingState,
        portfolio_stats: Optional[Dict[str, Any]],
        t0: float,
    ) -> SizerDecision:
        stats = portfolio_stats or {}
        win_rate = float(stats.get("win_rate", 0.5))
        avg_win = float(stats.get("avg_win_pct", 0.02))
        avg_loss = float(stats.get("avg_loss_pct", 0.01))
        confidence = abs(float(state.quant_score))

        decision = self.fallback.size(win_rate, avg_win, avg_loss, confidence)
        decision.elapsed_ms = (time.perf_counter() - t0) * 1000
        decision.method = "half_kelly"
        return decision

    def _compute_cvar95(self, window_size: int = 20) -> float:
        if not self._pnl_history:
            return 0.0
        window = self._pnl_history[-window_size:]
        sorted_pnls = sorted(window)
        n_tail = max(1, int(len(sorted_pnls) * 0.05))
        return abs(float(np.mean(sorted_pnls[:n_tail])))

    def save(self, directory: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Persist trained weights + metadata to disk."""
        save_dir = Path(directory) if directory else self.MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / self.DEFAULT_MODEL_FILE
        meta_path = save_dir / self.METADATA_FILE

        if self.policy is None:
            raise ValueError("Cannot save: no policy loaded.")

        self.policy.save(model_path)

        meta = {**(metadata or {}), **self.metadata}
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info(f"RLPositionSizer: saved model to {model_path}")
        return model_path
