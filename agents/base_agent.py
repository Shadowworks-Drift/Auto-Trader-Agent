"""
Base agent ABC with shared helpers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger


@dataclass
class AgentResult:
    agent_name: str
    symbol: str
    timestamp: datetime
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    @property
    def confidence(self) -> float:
        return float(self.data.get("confidence", 0.0))

    @property
    def direction(self) -> str:
        return str(self.data.get("direction", "none"))


class BaseAgent(ABC):
    """Abstract base class for all trading agents."""

    name: str = "BaseAgent"

    def __init__(self) -> None:
        self.logger = logger.bind(agent=self.name)

    @abstractmethod
    async def analyse(self, *args: Any, **kwargs: Any) -> AgentResult:
        ...

    def _make_result(
        self,
        symbol: str,
        success: bool,
        data: Dict[str, Any],
        elapsed_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            success=success,
            data=data,
            error=error,
            elapsed_ms=elapsed_ms,
        )
