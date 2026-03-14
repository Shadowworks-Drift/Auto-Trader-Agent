"""
RL Trainer
══════════
Trains a PPO-style policy on historical paper trade data using the
TradingEnv Gymnasium environment.

Training algorithm: REINFORCE with baseline (policy gradient).
For full PPO, swap in stable-baselines3 or cleanrl if available.

Usage:
    python rl/train.py --db data/trades.db --episodes 500
    python rl/train.py --episodes 200 --dummy   # synthetic data (no DB needed)
    python rl/train.py --help

References:
  - SAPPO (NeurIPS 2025): Sentiment-Augmented PPO
  - MTS iCVaR reward (arXiv 2503.04143)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from .environment import TradingEnv, TradingState
from .position_sizer import RLPositionSizer, _PolicyNetwork


# ── Training config ───────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    n_episodes: int = 500
    learning_rate: float = 3e-4
    gamma: float = 0.99           # discount factor
    baseline_decay: float = 0.95  # exponential baseline
    clip_grad: float = 0.5
    entropy_coeff: float = 0.01   # entropy bonus for exploration
    log_every: int = 50
    save_dir: str = "models"
    early_stop_patience: int = 100  # stop if no improvement for N episodes
    min_episodes_before_stop: int = 200


# ── REINFORCE trainer ─────────────────────────────────────────────────────────

class RLTrainer:
    """
    Policy-gradient trainer for the MLP position-sizing policy.

    Uses REINFORCE with an exponential moving average baseline to reduce
    variance.  The policy outputs a Gaussian action (mean = sigmoid(logit),
    std = 0.1) so we can compute log-probabilities and train with gradients.
    """

    def __init__(self, env: TradingEnv, cfg: Optional[TrainConfig] = None) -> None:
        self.env = env
        self.cfg = cfg or TrainConfig()
        self.policy = _PolicyNetwork.random_init(TradingState.STATE_DIM)
        self._baseline = 0.0
        self._best_avg_reward = -np.inf
        self._no_improvement = 0
        self.reward_history: List[float] = []

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> Dict[str, Any]:
        logger.info(
            f"RLTrainer: starting {self.cfg.n_episodes} episodes | "
            f"lr={self.cfg.learning_rate} γ={self.cfg.gamma}"
        )
        t_start = time.perf_counter()

        for ep in range(1, self.cfg.n_episodes + 1):
            ep_reward = self._run_episode()
            self.reward_history.append(ep_reward)

            # Logging
            if ep % self.cfg.log_every == 0 or ep == self.cfg.n_episodes:
                recent = self.reward_history[-self.cfg.log_every:]
                avg_r = float(np.mean(recent))
                logger.info(f"  Episode {ep:>5}/{self.cfg.n_episodes} | avg_reward={avg_r:.4f}")

                # Early stopping
                if ep >= self.cfg.min_episodes_before_stop:
                    if avg_r > self._best_avg_reward:
                        self._best_avg_reward = avg_r
                        self._no_improvement = 0
                    else:
                        self._no_improvement += self.cfg.log_every
                        if self._no_improvement >= self.cfg.early_stop_patience:
                            logger.info(f"Early stopping at episode {ep} (no improvement)")
                            break

        elapsed = time.perf_counter() - t_start
        final_avg = float(np.mean(self.reward_history[-50:])) if self.reward_history else 0.0
        logger.info(f"Training complete in {elapsed:.1f}s | final_avg_reward={final_avg:.4f}")

        return {
            "n_episodes": len(self.reward_history),
            "avg_reward": round(final_avg, 4),
            "best_avg_reward": round(self._best_avg_reward, 4),
            "training_time_s": round(elapsed, 2),
        }

    def save(self, directory: Optional[str] = None) -> Path:
        sizer = RLPositionSizer(policy=self.policy)
        meta = {
            "n_episodes": len(self.reward_history),
            "avg_reward": round(float(np.mean(self.reward_history[-50:])) if self.reward_history else 0.0, 4),
            "best_avg_reward": round(self._best_avg_reward, 4),
        }
        return sizer.save(directory=directory or self.cfg.save_dir, metadata=meta)

    # ── Episode runner ────────────────────────────────────────────────────────

    def _run_episode(self) -> float:
        obs = self.env.reset()
        states, actions, rewards, log_probs = [], [], [], []
        done = False

        while not done:
            action, log_prob = self._sample_action(obs)
            next_obs, reward, done, _ = self.env.step(np.array([action]))
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            obs = next_obs

        if not rewards:
            return 0.0

        ep_reward = float(np.sum(rewards))
        self._update_policy(states, actions, rewards, log_probs)
        return ep_reward

    def _sample_action(self, obs: np.ndarray) -> tuple:
        """Sample action from Gaussian policy."""
        mean = self.policy.forward(obs)
        std = 0.1
        # Clipped Gaussian sample
        action = float(np.clip(np.random.normal(mean, std), 0.0, 1.0))
        # Log-probability of the sample under N(mean, std)
        log_prob = -0.5 * ((action - mean) / std) ** 2 - np.log(std * np.sqrt(2 * np.pi))
        return action, float(log_prob)

    def _update_policy(
        self,
        states: List[np.ndarray],
        actions: List[float],
        rewards: List[float],
        log_probs: List[float],
    ) -> None:
        """REINFORCE gradient update with baseline."""
        # Discounted returns
        returns = self._compute_returns(rewards)
        if len(returns) == 0:
            return

        # Update baseline
        ep_mean = float(np.mean(returns))
        self._baseline = (
            self.cfg.baseline_decay * self._baseline
            + (1 - self.cfg.baseline_decay) * ep_mean
        )

        # Policy gradient update (numerical gradient via finite differences)
        lr = self.cfg.learning_rate
        eps = 1e-4

        for obs, action, ret, log_p in zip(states, actions, returns, log_probs):
            advantage = ret - self._baseline
            if abs(advantage) < 1e-9:
                continue

            # Finite-difference gradient for each weight matrix
            for attr in ["W1", "b1", "W2", "b2", "W3", "b3"]:
                w = getattr(self.policy, attr)
                flat = w.ravel()
                # Sample a few random indices to approximate gradient (for speed)
                n_sample = min(len(flat), 32)
                indices = np.random.choice(len(flat), n_sample, replace=False)
                for idx in indices:
                    orig = flat[idx]
                    flat[idx] = orig + eps
                    setattr(self.policy, attr, flat.reshape(w.shape))
                    lp_plus = self._log_prob(obs, action)

                    flat[idx] = orig - eps
                    setattr(self.policy, attr, flat.reshape(w.shape))
                    lp_minus = self._log_prob(obs, action)

                    flat[idx] = orig
                    setattr(self.policy, attr, flat.reshape(w.shape))

                    grad = (lp_plus - lp_minus) / (2 * eps) * advantage
                    flat[idx] += lr * grad

                setattr(self.policy, attr, flat.reshape(w.shape))

    def _log_prob(self, obs: np.ndarray, action: float) -> float:
        mean = self.policy.forward(obs)
        std = 0.1
        return float(-0.5 * ((action - mean) / std) ** 2)

    def _compute_returns(self, rewards: List[float]) -> np.ndarray:
        returns = np.zeros(len(rewards), dtype=np.float32)
        g = 0.0
        for i in reversed(range(len(rewards))):
            g = rewards[i] + self.cfg.gamma * g
            returns[i] = g
        return returns


# ── Synthetic episode generator (for quick testing without DB) ────────────────

def _make_synthetic_episodes(n_episodes: int = 50, steps_per_ep: int = 20) -> List[Dict]:
    """Generate synthetic paper-trade episodes for offline training."""
    rng = np.random.default_rng(42)
    episodes = []
    for _ in range(n_episodes):
        steps = []
        cum_pnl = 0.0
        for _ in range(steps_per_ep):
            pnl = float(rng.normal(0.003, 0.015))
            cum_pnl += pnl
            steps.append({
                "pnl_pct": pnl,
                "state": {
                    "quant_score": float(rng.uniform(-1, 1)),
                    "trend_confidence": float(rng.uniform(0, 1)),
                    "setup_confidence": float(rng.uniform(0, 1)),
                    "trigger_confidence": float(rng.uniform(0, 1)),
                    "sentiment_score": float(rng.uniform(-1, 1)),
                    "vol_regime": float(rng.choice([0.0, 0.5, 1.0, 1.5])),
                    "funding_rate": float(rng.uniform(-0.5, 0.5)),
                    "fear_greed": float(rng.uniform(0, 1)),
                    "drawdown": float(max(0, -cum_pnl)),
                    "daily_pnl_pct": float(rng.uniform(-0.1, 0.1)),
                    "open_positions": float(rng.uniform(0, 1)),
                },
            })
        episodes.append({"steps": steps})
    return episodes


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RL position-sizing agent on paper trade history"
    )
    parser.add_argument("--db", default=None, help="Path to paper trade SQLite database")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--max-pos", type=float, default=0.10, help="Max position size fraction")
    parser.add_argument("--save-dir", default="models", help="Directory to save trained model")
    parser.add_argument("--dummy", action="store_true", help="Use synthetic episodes (no DB)")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N episodes")
    args = parser.parse_args()

    # Load episodes
    if args.dummy or args.db is None:
        logger.info("Using synthetic episodes for training")
        episodes = _make_synthetic_episodes(n_episodes=max(args.episodes, 100))
    else:
        db_path = Path(args.db)
        if not db_path.exists():
            logger.error(f"Database not found: {db_path}")
            return
        env_temp = TradingEnv.from_sqlite(str(db_path), max_position_size=args.max_pos)
        episodes = env_temp.episodes
        logger.info(f"Loaded {len(episodes)} episodes from {db_path}")

    if not episodes:
        logger.error("No training episodes available — cannot train.")
        return

    env = TradingEnv(episodes=episodes, max_position_size=args.max_pos)
    cfg = TrainConfig(
        n_episodes=args.episodes,
        learning_rate=args.lr,
        gamma=args.gamma,
        log_every=args.log_every,
        save_dir=args.save_dir,
    )
    trainer = RLTrainer(env=env, cfg=cfg)
    stats = trainer.train()

    model_path = trainer.save()
    print(f"\nTraining stats: {json.dumps(stats, indent=2)}")
    print(f"Model saved: {model_path}")


if __name__ == "__main__":
    main()
