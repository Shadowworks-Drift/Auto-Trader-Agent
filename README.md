# Auto-Trader-Agent

**Production-ready local LLM multi-agent trading system** for crypto and equities.

Runs entirely on your hardware — no cloud API keys required for the AI layer. Uses
[Ollama](https://ollama.ai) to run open-weight models (DeepSeek, Llama 3, Mistral, Qwen)
and a multi-agent Adversarial Decision Framework to produce high-quality, risk-managed
trade signals.

---

## Architecture

```
Universe of symbols
        │
        ▼
┌─────────────────┐
│  SymbolSelector │  Filters by volume, ATR volatility band, momentum
└────────┬────────┘
         │  selected symbols
         ▼
┌─────────────────┐
│    DataSync     │  Async CCXT → multi-timeframe OHLCV + news (CryptoPanic)
└────────┬────────┘
         │  MarketSnapshot
         ├─────────────────────────────────────┐
         ▼                                     ▼
┌─────────────────┐                  ┌──────────────────────┐
│  QuantAnalyst   │                  │   SentimentAgent     │
│  RSI/MACD/BB/   │                  │   LLM · news & social│
│  EMA/ATR/ADX/   │                  └──────────┬───────────┘
│  Stochastic     │                             │
└────────┬────────┘                             │
         │  QuantSignal                         │
         ▼                                      │
┌─────────────────┐                             │
│   TrendAgent    │  LLM · macro trend          │
└────────┬────────┘  (context + primary TF)     │
         │                                      │
         ▼                                      │
┌─────────────────┐                             │
│   SetupAgent    │  LLM · setup quality        │
└────────┬────────┘                             │
         │                                      │
         ▼                                      │
┌─────────────────┐                             │
│  TriggerAgent   │  LLM · entry trigger        │
└────────┬────────┘  (entry TF)                 │
         │                                      │
         └──────────────┬─────────────────────┘
                        ▼
              ┌──────────────────┐
              │  DecisionCore    │  Weighted fusion + Adversarial Debate
              │  (Bear/Bull LLM  │  (Bear advocate challenges long,
              │   Advocates)     │   Bull advocate challenges short)
              └────────┬─────────┘
                       │  TradeProposal
                       ▼
              ┌──────────────────┐
              │   RiskAudit      │  Hard rules + optional LLM narrative audit
              │  · SL direction  │  · Portfolio drawdown breaker
              │  · R:R check     │  · Daily loss breaker
              │  · Position size │  · Correlation limits
              └────────┬─────────┘
                       │  RiskDecision (approved/vetoed)
                       ▼
              ┌──────────────────┐
              │ ExecutionEngine  │  Paper (SQLite) or Live (CCXT)
              │  + SL/TP orders  │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │   Monitoring     │  Prometheus metrics + Rich dashboard
              └──────────────────┘
```

## Features

| Category | Feature |
|---|---|
| **LLM** | Ollama (local), OpenAI-compatible, fallback model, CoT prompting, structured JSON output |
| **Agents** | Symbol Selector, Quant Analyst, Trend/Setup/Trigger/Sentiment Semantic Agents |
| **Decision** | Adversarial Bear/Bull debate, weighted fusion, consensus threshold |
| **Risk** | SL direction correction, R:R gate, drawdown circuit-breaker, daily loss circuit-breaker, correlated position limit |
| **Indicators** | RSI, MACD, Bollinger Bands, EMA (20/50/200), ATR, Volume MA, Stochastic, ADX |
| **Execution** | Paper (SQLite-persisted), Live (CCXT), SL/TP bracket orders |
| **Monitoring** | Prometheus metrics, Rich terminal dashboard, structured loguru logs |
| **Config** | YAML + environment variable overrides |
| **Tests** | pytest-asyncio, full coverage of core modules |

---

## Quick Start

### 1. Install dependencies

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

### 2. Install Ollama and pull a model

```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended models
bash scripts/pull_models.sh

# Start Ollama server
ollama serve
```

**Minimum hardware:**
- 8 GB RAM → `deepseek-r1:7b` or `llama3:8b`
- 16 GB RAM → `deepseek-r1:14b`
- GPU (12 GB VRAM) → `llama3:70b`

### 3. Configure

```bash
cp config/default_config.yaml config/config.yaml
# Edit config/config.yaml
```

Key settings:

```yaml
trading:
  mode: paper          # paper | live
  symbols:
    - BTC/USDT
    - ETH/USDT

llm:
  model: deepseek-r1:7b
  base_url: http://localhost:11434

exchange:
  id: binance
  sandbox: true        # true = testnet
```

### 4. Health check

```bash
python scripts/healthcheck.py
```

### 5. Run

```bash
# Paper trading (default, safe)
python main.py

# Single cycle (test run)
python main.py --once

# Quant-only (no LLM, fastest)
python main.py --no-llm

# Custom symbols
python main.py --symbols BTC/USDT SOL/USDT

# Live trading (requires API keys in config)
python main.py --mode live
```

### 6. Run tests

```bash
pytest
```

---

## Project Structure

```
Auto-Trader-Agent/
├── main.py                      # Entry point / orchestrator
├── requirements.txt
├── pytest.ini
├── config/
│   ├── default_config.yaml      # Default settings (do not edit)
│   ├── config.yaml              # Your local config (git-ignored)
│   └── settings.py              # Pydantic settings loader
├── llm/
│   ├── ollama_client.py         # Async Ollama client with fallback
│   └── prompts.py               # All agent system/user prompt templates
├── data/
│   ├── market_data.py           # Domain models (OHLCV, Candle, Snapshot)
│   └── data_sync.py             # CCXT-based async data fetcher + cache
├── agents/
│   ├── base_agent.py            # Abstract base class
│   ├── symbol_selector.py       # Volume + ATR + momentum screener
│   ├── quant_analyst.py         # Technical indicator computation + voting
│   ├── semantic_agents.py       # TrendAgent, SetupAgent, TriggerAgent, SentimentAgent
│   ├── decision_core.py         # Signal fusion + adversarial debate
│   └── risk_audit.py            # Hard rules + LLM narrative risk audit
├── execution/
│   ├── paper_trading.py         # SQLite-persisted paper simulator
│   └── execution_engine.py      # Unified paper/live execution interface
├── monitoring/
│   ├── performance_tracker.py   # Prometheus metrics
│   └── dashboard.py             # Rich terminal UI
├── utils/
│   ├── logger.py                # Loguru setup
│   └── helpers.py               # Utility functions
├── scripts/
│   ├── setup.sh                 # One-command environment setup
│   ├── pull_models.sh           # Ollama model downloader
│   └── healthcheck.py           # Pre-flight checks
└── tests/
    ├── conftest.py
    ├── test_quant_analyst.py
    ├── test_decision_core.py
    ├── test_risk_audit.py
    └── test_paper_trading.py
```

---

## Configuration Reference

See `config/default_config.yaml` for all options. Key sections:

### `llm`
| Key | Default | Description |
|---|---|---|
| `provider` | `ollama` | `ollama` \| `openai` \| `anthropic` |
| `model` | `deepseek-r1:7b` | Primary model name |
| `fallback_model` | `llama3:8b` | Used if primary fails |
| `temperature` | `0.1` | Low = deterministic |
| `chain_of_thought` | `true` | Enable CoT prompting |

### `risk`
| Key | Default | Description |
|---|---|---|
| `max_portfolio_drawdown_pct` | `0.15` | Portfolio drawdown circuit-breaker |
| `max_daily_loss_pct` | `0.05` | Daily loss circuit-breaker |
| `stop_loss_pct` | `0.03` | Per-trade stop loss (3%) |
| `take_profit_pct` | `0.06` | Per-trade take profit (6%) |
| `min_risk_reward_ratio` | `1.5` | Minimum R:R to trade |

### `decision`
| Key | Default | Description |
|---|---|---|
| `min_confidence` | `0.65` | Minimum fused confidence |
| `quant_weight` | `0.40` | Quant analyst contribution |
| `adversarial_veto` | `true` | Enable bear/bull debate |

---

## Environment Variables

All config values can be overridden with `AUTO_TRADER_<SECTION>_<KEY>`:

```bash
AUTO_TRADER_LLM_MODEL=llama3:70b
AUTO_TRADER_EXCHANGE_API_KEY=your_key
AUTO_TRADER_EXCHANGE_API_SECRET=your_secret
AUTO_TRADER_TRADING_MODE=live
```

---

## Recommended Models

| Model | RAM | Quality | Use case |
|---|---|---|---|
| `deepseek-r1:7b` | 8 GB | ★★★★☆ | **Recommended default** |
| `llama3:8b` | 8 GB | ★★★★☆ | Reliable fallback |
| `mistral:7b` | 8 GB | ★★★☆☆ | Fast, lightweight |
| `deepseek-r1:14b` | 16 GB | ★★★★★ | Higher accuracy |
| `llama3:70b` | GPU 24 GB | ★★★★★ | Best quality |

---

## Metrics Dashboard

Prometheus metrics exposed at `http://localhost:8000/metrics` when enabled:

- `trader_portfolio_value_usd`
- `trader_total_return_pct`
- `trader_sharpe_ratio`
- `trader_max_drawdown_pct`
- `trader_win_rate`
- `trader_llm_latency_seconds{agent}`
- `trader_agent_confidence{agent, symbol}`
- `trader_decisions_total{direction, symbol}`

---

## Legal & Compliance

- **Paper mode by default** — no real money at risk until you explicitly set `mode: live`
- Review and comply with your exchange's ToS and local regulations
- This software does NOT constitute financial advice
- Maintain human oversight over all automated trading
- FINRA Rule 3110: document trading logic and maintain supervisory controls
- Avoid strategies that resemble spoofing, layering, or market manipulation
- See the scientific portfolio in the project docs for regulatory details (GENIUS Act, Digital Asset Market Clarity Act, FINRA Notice 24-09)

---

## Disclaimer

> This software is provided for educational and research purposes. Automated trading
> carries significant financial risk. Past performance of backtests or paper trades
> does not guarantee future results. Always start with paper trading, validate
> extensively, and only risk capital you can afford to lose. The authors accept no
> responsibility for financial losses.
