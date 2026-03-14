"""
Tests for FinBERTClient two-tier sentiment.
All tests run without downloading any models (mock-based).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm.finbert_client import FinBERTClient, SentimentScore


# ── SentimentScore unit tests ─────────────────────────────────────────────────

def test_sentiment_score_direction_positive() -> None:
    s = SentimentScore("positive", score=0.6, confidence=0.8, model_used="finbert2", elapsed_ms=20)
    assert s.direction == "bullish"


def test_sentiment_score_direction_negative() -> None:
    s = SentimentScore("negative", score=-0.7, confidence=0.85, model_used="finbert2", elapsed_ms=25)
    assert s.direction == "bearish"


def test_sentiment_score_neutral_zone() -> None:
    s = SentimentScore("neutral", score=0.05, confidence=0.6, model_used="finbert2", elapsed_ms=18)
    assert s.direction == "neutral"


def test_aggregate_weights_by_confidence() -> None:
    client = FinBERTClient()
    scores = [
        SentimentScore("positive", score=0.8, confidence=0.9, model_used="finbert2", elapsed_ms=10),
        SentimentScore("negative", score=-0.6, confidence=0.1, model_used="finbert2", elapsed_ms=10),
    ]
    agg = client.aggregate(scores)
    # High confidence positive score should dominate
    assert agg.score > 0


def test_aggregate_empty_returns_neutral() -> None:
    client = FinBERTClient()
    agg = client.aggregate([])
    assert agg.label == "neutral"
    assert agg.score == 0.0


def test_parse_pipeline_output_finbert_format() -> None:
    client = FinBERTClient()
    mock_output = [
        {"label": "positive", "score": 0.87},
        {"label": "negative", "score": 0.08},
        {"label": "neutral",  "score": 0.05},
    ]
    score, confidence, label = client._parse_pipeline_output(mock_output)
    assert score > 0.5
    assert label == "positive"
    assert confidence > 0.5


def test_parse_pipeline_output_negative() -> None:
    client = FinBERTClient()
    mock_output = [
        {"label": "positive", "score": 0.05},
        {"label": "negative", "score": 0.91},
        {"label": "neutral",  "score": 0.04},
    ]
    score, confidence, label = client._parse_pipeline_output(mock_output)
    assert score < 0
    assert label == "negative"


# ── Init / availability ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_init_without_transformers_returns_false() -> None:
    client = FinBERTClient()
    with patch("llm.finbert_client.FinBERTClient._load_model", side_effect=ImportError("no transformers")):
        result = await client.init()
    assert result is False
    assert not client.is_available()


# ── Fast-path tests ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fast_path_returns_result_when_confident() -> None:
    client = FinBERTClient(ollama_escalate=False)
    client._available = True
    client._pipeline = MagicMock()

    mock_output = [
        {"label": "positive", "score": 0.92},
        {"label": "negative", "score": 0.04},
        {"label": "neutral",  "score": 0.04},
    ]
    client._pipeline.return_value = mock_output

    with patch.object(client, "_run_pipeline", return_value=mock_output):
        result = await client.score_text("Bitcoin ETF approval expected this week")

    assert result.direction == "bullish"
    assert result.confidence > 0.5
    assert result.model_used == "finbert2"


@pytest.mark.asyncio
async def test_escalates_to_ollama_when_ambiguous() -> None:
    mock_ollama = AsyncMock()
    mock_ollama.chat = AsyncMock(return_value=MagicMock(
        parsed={"label": "negative", "score": -0.55, "confidence": 0.7},
        content=""
    ))

    client = FinBERTClient(ollama_escalate=True, ollama_client=mock_ollama)
    client._available = True

    # Ambiguous FinBERT output
    mock_output = [
        {"label": "positive", "score": 0.40},
        {"label": "negative", "score": 0.35},
        {"label": "neutral",  "score": 0.25},
    ]
    with patch.object(client, "_run_pipeline", return_value=mock_output):
        result = await client.score_text("Market conditions remain uncertain")

    # Should have escalated
    assert mock_ollama.chat.called
    assert result.model_used == "ollama"


# ── Graceful failure ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_returns_neutral_when_both_fail() -> None:
    client = FinBERTClient(ollama_escalate=False)
    client._available = False
    client.ollama_client = None

    result = await client.score_text("Any headline here")
    assert result.label == "neutral"
    assert result.score == 0.0
    assert result.model_used == "none"
