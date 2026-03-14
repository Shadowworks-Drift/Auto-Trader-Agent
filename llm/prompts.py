"""
Prompt library for all trading agents.
All prompts use chain-of-thought (CoT) structure and request structured JSON output.
"""

from __future__ import annotations

from string import Template
from typing import Dict


class PromptLibrary:
    """Central store of system and user prompt templates."""

    # ── System prompts ────────────────────────────────────────────────────────

    SYSTEM_TREND_AGENT = """\
You are an expert cryptocurrency and financial market trend analyst.
Your role is to assess the MACRO TREND of a given asset across multiple timeframes.
You must reason step-by-step using technical and contextual data provided.
Always output a valid JSON object matching the requested schema — no prose outside JSON.
"""

    SYSTEM_SETUP_AGENT = """\
You are an expert trading setup analyst specialising in identifying high-probability
trade setups based on price action, technical indicators and market structure.
Your role is to evaluate whether current conditions represent a favourable setup for entry.
Always output a valid JSON object matching the requested schema — no prose outside JSON.
"""

    SYSTEM_TRIGGER_AGENT = """\
You are an expert short-term entry trigger specialist.
Your role is to identify precise entry signals by analysing the lowest timeframe
in context of the established setup and trend.
Always output a valid JSON object matching the requested schema — no prose outside JSON.
"""

    SYSTEM_SENTIMENT_AGENT = """\
You are a financial sentiment analyst specialising in cryptocurrency and equity markets.
Your role is to interpret news headlines, social signals and on-chain data narratives
to produce a quantified sentiment score.
Always output a valid JSON object matching the requested schema — no prose outside JSON.
"""

    SYSTEM_RISK_AGENT = """\
You are a senior risk management officer at a quantitative hedge fund.
Your role is to critically evaluate proposed trades and veto any that breach
risk, regulatory or ethical guidelines.
Be conservative and sceptical. Only approve trades with a clear risk:reward edge.
Always output a valid JSON object matching the requested schema — no prose outside JSON.
"""

    SYSTEM_BEAR_ADVOCATE = """\
You are a devil's advocate bear analyst.  Your job is to find every reason why a
proposed LONG trade might fail, including macro risks, technical failures, and
sentiment traps.  Be thorough and sceptical.
Always output a valid JSON object matching the requested schema — no prose outside JSON.
"""

    SYSTEM_BULL_ADVOCATE = """\
You are a devil's advocate bull analyst.  Your job is to find every reason why a
proposed SHORT trade might fail, including short squeezes, positive catalysts, and
accumulation patterns.  Be thorough and sceptical.
Always output a valid JSON object matching the requested schema — no prose outside JSON.
"""

    # ── User prompt templates (use Template with $var substitution) ───────────

    TREND_ANALYSIS = Template("""\
Analyse the following market data for $symbol and determine the primary trend.

## Timeframe Data
- Context ($context_tf): $context_data
- Primary ($primary_tf): $primary_data

## Technical Indicators (Primary TF)
$indicators

## Task
1. Identify the primary trend direction (bullish / bearish / sideways)
2. Assess trend strength (1–10)
3. Identify key support and resistance levels
4. Note any trend change signals

Respond with this JSON schema:
{
  "trend": "bullish|bearish|sideways",
  "strength": <1-10>,
  "reasoning": "<concise chain-of-thought>",
  "key_levels": {"support": [<price>,...], "resistance": [<price>,...]},
  "confidence": <0.0-1.0>,
  "warnings": ["<any red flags>"]
}
""")

    SETUP_ANALYSIS = Template("""\
Evaluate the trade setup for $symbol given the trend context and current price action.

## Trend Context
$trend_summary

## Price Action (Primary TF: $primary_tf)
$primary_data

## Technical Indicators
$indicators

## Task
1. Identify the setup type (e.g., breakout, pullback, reversal, range)
2. Assess setup quality (1–10)
3. Confirm or deny alignment with the macro trend
4. Identify invalidation level

Respond with this JSON schema:
{
  "setup_type": "<string>",
  "direction": "long|short|none",
  "quality": <1-10>,
  "reasoning": "<concise chain-of-thought>",
  "invalidation_price": <float>,
  "target_price": <float>,
  "confidence": <0.0-1.0>
}
""")

    TRIGGER_ANALYSIS = Template("""\
Identify a precise entry trigger for $symbol on the $entry_tf timeframe.

## Setup Context
$setup_summary

## Entry TF Data ($entry_tf)
$entry_data

## Current Price: $current_price

## Task
1. Has an entry trigger fired? (candle close, level break, indicator cross)
2. Specify the exact trigger condition
3. Suggest entry price, stop-loss, and take-profit

Respond with this JSON schema:
{
  "trigger_fired": true|false,
  "trigger_type": "<string>",
  "entry_price": <float>,
  "stop_loss": <float>,
  "take_profit": <float>,
  "reasoning": "<concise chain-of-thought>",
  "confidence": <0.0-1.0>,
  "urgency": "immediate|next_candle|wait"
}
""")

    SENTIMENT_ANALYSIS = Template("""\
Analyse the following news headlines and social signals for $symbol over the last $hours hours.

## Headlines
$headlines

## Social Signals
$social_signals

## Task
1. Overall sentiment direction
2. Key catalysts (positive and negative)
3. Sentiment momentum (improving / deteriorating / stable)

Respond with this JSON schema:
{
  "sentiment": "bullish|bearish|neutral",
  "score": <-1.0 to 1.0>,
  "momentum": "improving|deteriorating|stable",
  "key_catalysts": {"positive": ["<str>",...], "negative": ["<str>",...]},
  "reasoning": "<concise chain-of-thought>",
  "confidence": <0.0-1.0>
}
""")

    RISK_AUDIT = Template("""\
Audit the following proposed trade and determine if it should be approved or vetoed.

## Trade Proposal
- Symbol: $symbol
- Direction: $direction
- Entry: $entry_price
- Stop-loss: $stop_loss  (risk: $risk_pct%)
- Take-profit: $take_profit  (reward: $reward_pct%)
- Risk:Reward ratio: $risk_reward
- Position size: $position_size_pct% of portfolio

## Portfolio Context
- Current portfolio value: $$portfolio_value
- Open positions: $open_positions
- Daily P&L so far: $daily_pnl_pct%
- Max drawdown limit: $max_drawdown_pct%

## Agent Signals
$agent_signals

## Adversarial Arguments
$adversarial_args

## Task
1. Verify stop-loss direction is correct relative to entry
2. Confirm R:R ratio meets minimum threshold ($min_rr)
3. Check daily loss and drawdown limits
4. Evaluate overall conviction vs risk

Respond with this JSON schema:
{
  "approved": true|false,
  "veto_reason": "<string or null>",
  "adjusted_stop_loss": <float or null>,
  "adjusted_take_profit": <float or null>,
  "adjusted_size_pct": <float or null>,
  "risk_score": <0.0-1.0>,
  "reasoning": "<concise chain-of-thought>"
}
""")

    ADVERSARIAL_BEAR = Template("""\
A long trade is proposed for $symbol at $entry_price (stop $stop_loss, target $take_profit).

## Bull Case Summary
$bull_summary

## Market Data
$market_data

## Your Task (Bear Advocate)
Challenge every bullish assumption. Find technical, fundamental and macro arguments
AGAINST this long trade. Be specific and data-driven.

Respond with this JSON schema:
{
  "bear_arguments": ["<argument 1>", "<argument 2>", ...],
  "risk_factors": ["<risk>", ...],
  "invalidation_probability": <0.0-1.0>,
  "overall_bearish_conviction": <0.0-1.0>
}
""")

    ADVERSARIAL_BULL = Template("""\
A short trade is proposed for $symbol at $entry_price (stop $stop_loss, target $take_profit).

## Bear Case Summary
$bear_summary

## Market Data
$market_data

## Your Task (Bull Advocate)
Challenge every bearish assumption. Find technical, fundamental and macro arguments
AGAINST this short trade. Be specific and data-driven.

Respond with this JSON schema:
{
  "bull_arguments": ["<argument 1>", "<argument 2>", ...],
  "risk_factors": ["<risk>", ...],
  "invalidation_probability": <0.0-1.0>,
  "overall_bullish_conviction": <0.0-1.0>
}
""")

    @classmethod
    def render(cls, template: Template, **kwargs: str) -> str:
        """Safely substitute template variables, leaving unset ones as-is."""
        return template.safe_substitute(**kwargs)
