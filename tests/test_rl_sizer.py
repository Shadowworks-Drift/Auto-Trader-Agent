"""
Tests for RL position sizer components:
  - TradingState
  - TradingEnv (step, reset, iCVaR reward)
  - HalfKellyCVaR
  - RLPositionSizer (with and without model)
  - RLTrainer (smoke test)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from rl.environment import TradingEnv, TradingState
from rl.position_sizer import HalfKellyCVaR, RLPositionSizer, SizerDecision, _PolicyNetwork
from rl.train import RLTrainer, TrainConfig, _make_synthetic_episodes


# ── TradingState ──────────────────────────────────────────────────────────────

class TestTradingState:
    def test_to_array_shape(self):
        state = TradingState(
            quant_score=0.5,
            trend_confidence=0.7,
            setup_confidence=0.6,
            trigger_confidence=0.8,
            sentiment_score=0.2,
            vol_regime=0.5,
            funding_rate=0.0,
            fear_greed=0.5,
            portfolio_drawdown=0.02,
            daily_pnl_pct=0.01,
            open_positions=0.33,
        )
        arr = state.to_array()
        assert arr.shape == (TradingState.STATE_DIM,)
        assert arr.dtype == np.float32

    def test_funding_rate_clip(self):
        state = TradingState(
            quant_score=0.0, trend_confidence=0.5, setup_confidence=0.5,
            trigger_confidence=0.5, sentiment_score=0.0, vol_regime=0.5,
            funding_rate=99.0,  # should be clipped to 3
            fear_greed=0.5, portfolio_drawdown=0.0,
            daily_pnl_pct=0.0, open_positions=0.0,
        )
        arr = state.to_array()
        assert arr[6] == pytest.approx(3.0)

    def test_from_agent_results(self):
        state = TradingState.from_agent_results(
            quant_data={"score": 0.7},
            trend_data={"confidence": 0.8},
            setup_data={"confidence": 0.6},
            trigger_data={"confidence": 0.9},
            sentiment_data={"score": 0.3},
            portfolio_info={
                "funding_rate": 0.05,
                "fear_greed": 60,
                "drawdown": 0.01,
                "daily_pnl_pct": 0.005,
                "open_count": 1,
                "max_positions": 3,
            },
            vol_regime=0.5,
        )
        assert state.quant_score == pytest.approx(0.7)
        assert state.trend_confidence == pytest.approx(0.8)
        assert state.fear_greed == pytest.approx(0.6)
        assert state.open_positions == pytest.approx(1 / 3)


# ── TradingEnv ────────────────────────────────────────────────────────────────

def _make_episodes(n: int = 3, steps: int = 5) -> List[Dict]:
    rng = np.random.default_rng(0)
    eps = []
    for _ in range(n):
        ep_steps = []
        for _ in range(steps):
            ep_steps.append({
                "pnl_pct": float(rng.normal(0.002, 0.01)),
                "state": {
                    "quant_score": 0.5,
                    "trend_confidence": 0.6,
                    "setup_confidence": 0.6,
                    "trigger_confidence": 0.7,
                    "sentiment_score": 0.1,
                    "vol_regime": 0.5,
                    "funding_rate": 0.0,
                    "fear_greed": 0.5,
                    "drawdown": 0.0,
                    "daily_pnl_pct": 0.001,
                    "open_positions": 0.33,
                },
            })
        eps.append({"steps": ep_steps})
    return eps


class TestTradingEnv:
    def test_reset_returns_state(self):
        env = TradingEnv(episodes=_make_episodes())
        obs = env.reset()
        assert obs.shape == (TradingState.STATE_DIM,)
        assert obs.dtype == np.float32

    def test_step_returns_tuple(self):
        env = TradingEnv(episodes=_make_episodes())
        env.reset()
        obs, reward, done, info = env.step(np.array([0.5]))
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "pnl" in info
        assert "size" in info

    def test_episode_terminates(self):
        env = TradingEnv(episodes=_make_episodes(n=1, steps=3))
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(np.array([0.5]))
            steps += 1
            if steps > 20:
                pytest.fail("Episode never terminated")
        assert steps == 3

    def test_episode_cycling(self):
        env = TradingEnv(episodes=_make_episodes(n=2, steps=3))
        for _ in range(3):
            env.reset()
            done = False
            while not done:
                _, _, done, _ = env.step(np.array([0.5]))
        # Should have cycled back without error
        assert env._current_ep <= 2

    def test_icvar_reward_clipped(self):
        env = TradingEnv(episodes=_make_episodes(n=1, steps=30), cvar_lambda=1.0)
        env.reset()
        done = False
        rewards = []
        while not done:
            _, r, done, _ = env.step(np.array([1.0]))
            rewards.append(r)
        assert all(-10.0 <= r <= 10.0 for r in rewards)

    def test_zero_action(self):
        env = TradingEnv(episodes=_make_episodes())
        env.reset()
        obs, reward, done, info = env.step(np.array([0.0]))
        assert info["size"] == pytest.approx(0.0)
        assert reward == pytest.approx(0.0)  # scaled_pnl = 0

    def test_empty_episodes(self):
        env = TradingEnv(episodes=[])
        obs = env.reset()
        assert obs.shape == (TradingState.STATE_DIM,)


# ── HalfKellyCVaR ─────────────────────────────────────────────────────────────

class TestHalfKellyCVaR:
    def test_basic_sizing(self):
        sizer = HalfKellyCVaR(max_position_size=0.10)
        decision = sizer.size(win_rate=0.6, avg_win=0.03, avg_loss=0.02)
        assert isinstance(decision, SizerDecision)
        assert 0.0 < decision.position_size_pct <= 0.10
        assert decision.method == "half_kelly"

    def test_zero_win_rate(self):
        sizer = HalfKellyCVaR(max_position_size=0.10)
        decision = sizer.size(win_rate=0.0, avg_win=0.01, avg_loss=0.01)
        assert decision.position_size_pct >= 0.0

    def test_cvar_with_history(self):
        sizer = HalfKellyCVaR(max_position_size=0.10)
        for pnl in [-0.05, -0.04, -0.03, 0.02, 0.03, 0.01, -0.02, 0.04]:
            sizer.record_pnl(pnl)
        decision = sizer.size(win_rate=0.55, avg_win=0.025, avg_loss=0.02)
        assert decision.cvar_95 > 0.0
        assert decision.position_size_pct <= 0.10


# ── _PolicyNetwork ─────────────────────────────────────────────────────────────

class TestPolicyNetwork:
    def test_forward_output_range(self):
        policy = _PolicyNetwork.random_init()
        obs = np.random.randn(TradingState.STATE_DIM).astype(np.float32)
        out = policy.forward(obs)
        assert 0.0 <= out <= 1.0

    def test_save_load_roundtrip(self):
        policy = _PolicyNetwork.random_init()
        obs = np.random.randn(TradingState.STATE_DIM).astype(np.float32)
        original_out = policy.forward(obs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_policy.npz"
            policy.save(path)
            loaded = _PolicyNetwork.load(path)
            loaded_out = loaded.forward(obs)

        assert abs(original_out - loaded_out) < 1e-6


# ── RLPositionSizer ───────────────────────────────────────────────────────────

class TestRLPositionSizer:
    def _make_state(self) -> TradingState:
        return TradingState(
            quant_score=0.6,
            trend_confidence=0.7,
            setup_confidence=0.65,
            trigger_confidence=0.75,
            sentiment_score=0.2,
            vol_regime=0.5,
            funding_rate=0.0,
            fear_greed=0.5,
            portfolio_drawdown=0.01,
            daily_pnl_pct=0.005,
            open_positions=0.33,
        )

    def test_untrained_decide(self):
        sizer = RLPositionSizer.untrained()
        state = self._make_state()
        decision = sizer.decide(state)
        assert isinstance(decision, SizerDecision)
        assert 0.0 <= decision.position_size_pct <= sizer.max_position_size
        assert decision.method == "rl_ppo"

    def test_fallback_when_no_policy(self):
        sizer = RLPositionSizer(policy=None, max_position_size=0.10)
        state = self._make_state()
        decision = sizer.decide(state, portfolio_stats={"win_rate": 0.55})
        assert decision.method == "half_kelly"
        assert decision.position_size_pct <= 0.10

    def test_no_model_file(self):
        sizer = RLPositionSizer.from_model(path="/nonexistent/path/model.npz")
        assert sizer.policy is None

    def test_decide_from_dict(self):
        sizer = RLPositionSizer.untrained()
        decision = sizer.decide_from_dict(
            quant_data={"score": 0.5},
            trend_data={"confidence": 0.7},
            setup_data={"confidence": 0.6},
            trigger_data={"confidence": 0.8},
            sentiment_data={"score": 0.3},
            portfolio_info={
                "funding_rate": 0.0, "fear_greed": 50,
                "drawdown": 0.0, "daily_pnl_pct": 0.0,
                "open_count": 1, "max_positions": 3,
            },
        )
        assert isinstance(decision, SizerDecision)

    def test_save_load(self):
        sizer = RLPositionSizer.untrained()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = sizer.save(directory=tmpdir, metadata={"test": True})
            assert path.exists()
            loaded = RLPositionSizer.from_model(path=str(path))
            assert loaded.policy is not None

        state = self._make_state()
        d1 = sizer.decide(state)
        d2 = loaded.decide(state)
        assert abs(d1.raw_action - d2.raw_action) < 1e-5

    def test_record_pnl(self):
        sizer = RLPositionSizer.untrained()
        for pnl in [-0.02, 0.03, -0.01, 0.04, 0.02]:
            sizer.record_pnl(pnl)
        assert len(sizer._pnl_history) == 5


# ── RLTrainer ─────────────────────────────────────────────────────────────────

class TestRLTrainer:
    def test_smoke_train(self):
        episodes = _make_synthetic_episodes(n_episodes=10, steps_per_ep=5)
        env = TradingEnv(episodes=episodes)
        cfg = TrainConfig(n_episodes=5, log_every=5)
        trainer = RLTrainer(env=env, cfg=cfg)
        stats = trainer.train()
        assert "n_episodes" in stats
        assert stats["n_episodes"] >= 5
        assert len(trainer.reward_history) >= 5

    def test_save_after_training(self):
        episodes = _make_synthetic_episodes(n_episodes=10, steps_per_ep=5)
        env = TradingEnv(episodes=episodes)
        cfg = TrainConfig(n_episodes=3, log_every=3)
        trainer = RLTrainer(env=env, cfg=cfg)
        trainer.train()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = trainer.save(directory=tmpdir)
            assert path.exists()

    def test_synthetic_episodes(self):
        eps = _make_synthetic_episodes(n_episodes=5, steps_per_ep=10)
        assert len(eps) == 5
        assert len(eps[0]["steps"]) == 10
        assert "pnl_pct" in eps[0]["steps"][0]
        assert "state" in eps[0]["steps"][0]
