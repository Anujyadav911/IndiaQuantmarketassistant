# IndiaQuant MCP 4

Real-time Indian Stock Market AI Assistant using Model Context Protocol (MCP) + 100% Free APIs.

Give Claude (or any MCP-compatible AI agent) full Indian stock market intelligence without spending a rupee on APIs.

---

## Table of Contents

1. [What Is IndiaQuant MCP?](#what-is-indiaquant-mcp)
2. [Architecture Overview](#architecture-overview)
3. [Module Breakdown](#module-breakdown)
4. [Free API Stack](#free-api-stack)
5. [10 MCP Tools Reference](#10-mcp-tools-reference)
6. [Quick Start](#quick-start)
7. [Claude Desktop Integration](#claude-desktop-integration)
8. [Example Claude Conversations](#example-claude-conversations)
9. [Running Tests](#running-tests)
10. [Design Decisions & Trade-offs](#design-decisions--trade-offs)
11. [Known Limitations](#known-limitations)

---

## What Is IndiaQuant MCP?

**Without this MCP**: Claude has no awareness of live NIFTY levels, your portfolio positions, RSI values, or options activity.

**With this MCP**: Claude becomes a fully-equipped quant assistant for Indian markets:

- "Should I buy HDFC Bank right now?" — get RSI, MACD, Bollinger, sentiment combined into one BUY/SELL/HOLD signal.
- "What's the max pain for Nifty this expiry?" — live options chain → max-pain strike.
- "Show my portfolio P&L with risk breakdown." — real-time unrealised P&L, stop-loss alerts, volatility scores.
- "Find oversold IT stocks with RSI below 30." — market scanner across sectors.

---

## Architecture Overview

```
+-------------------------------------------------------------+
|                        Claude Desktop                       |
|                    (or any MCP client)                      |
+-------------------------------------------------------------+
                  |  MCP Protocol (stdio / JSON-RPC)
                  v
+-------------------------------------------------------------+
|                    main.py  --  MCP Server                  |
|   Tool registry (10 tools) + input validation + routing     |
+-------------------------------------------------------------+
     |        |          |              |          |
     v        v          v              v          v
 market_   signals.  options.py   portfolio.  sector_
 data.py    py                      py       heatmap.py
     |        |          |              |          |
     v        v          v              v          v
 yfinance  pure-numpy  Black-        SQLite    yfinance
 (Yahoo)   /pandas     Scholes       (local)   (batch)
              |
              v
          NewsAPI.org
```

### Request Lifecycle

1. Claude emits a tool call (JSON-RPC over stdio).
2. main.py validates arguments and routes to the correct module.
3. The module checks its in-memory TTL cache; on miss, fetches live data.
4. Data is enriched (indicators computed, Greeks calculated) and returned as structured JSON.
5. Claude receives clean JSON and synthesises a natural-language response.

---

## Module Breakdown

### Module 1 -- src/market_data.py -- Market Data Engine

| Function | Description |
|---|---|
| get_live_price(symbol) | Live quote: price, change %, volume, 52w hi/lo |
| get_historical_data(symbol, period, interval) | Full OHLCV history via yfinance |
| get_nifty50_prices() | Batch-fetch all 50 Nifty constituents |
| normalise_symbol(symbol) | Maps NIFTY -> ^NSEI, RELIANCE -> RELIANCE.NS |

Caching: Price TTL = 30 s, History TTL = 5 min (configurable in src/config.py).

---

### Module 2 -- src/signals.py -- AI Trade Signal Generator

Indicators computed (pure numpy/pandas -- no TA-Lib dependency):

| Indicator | Scoring Logic |
|---|---|
| RSI (14) | < 30 -> bullish (+up to 40), > 70 -> bearish (-up to 40) |
| MACD histogram | Positive -> +15, Negative -> -15 |
| MACD cross | Line > Signal -> +10, Line < Signal -> -10 |
| Bollinger Bands | Near lower band -> +20, near upper -> -20 |
| EMA 9/21 cross | EMA9 > EMA21 -> +10 |
| SMA 50 trend | Price > SMA50 -> +10 |
| SMA 200 trend | Price > SMA200 -> +10 |
| Chart patterns | +/-15 per detected pattern |
| Sentiment | Weighted NewsAPI score -> +/-20 |

Signal mapping:
- Composite >= +20 -> BUY, confidence = 50 + composite
- Composite <= -20 -> SELL, confidence = 50 + |composite|
- -20 < composite < +20 -> HOLD

Chart patterns detected: Double Top/Bottom, Golden/Death Cross, Bullish/Bearish Engulfing, RSI Divergence.

---

### Module 3 -- src/options.py -- Options Chain Analyzer

#### Black-Scholes Greeks (from scratch -- zero library dependencies)

Full implementation of the Black-Scholes-Merton model:

  d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma * sqrt(T))
  d2 = d1 - sigma * sqrt(T)

| Greek | Formula |
|---|---|
| Call Price | S*N(d1) - K*exp(-rT)*N(d2) |
| Put Price | K*exp(-rT)*N(-d2) - S*N(-d1) |
| Delta (Call) | N(d1) |
| Delta (Put) | N(d1) - 1 |
| Gamma | N'(d1) / (S * sigma * sqrt(T)) |
| Theta | -[S*N'(d1)*sigma / (2*sqrt(T))] - r*K*exp(-rT)*N(d2), per day |
| Vega | S*N'(d1)*sqrt(T), per 1% IV change |
| Rho | K*T*exp(-rT)*N(d2), per 1% rate change |

The normal CDF uses the Abramowitz & Stegun rational polynomial approximation (error < 7.5e-8) -- no scipy needed.

Implied Volatility: Estimated via bisection method (100 iterations, converges to < 0.01 tolerance).

#### Max Pain Calculation

  Max Pain = argmin over K* of:
    [sum(calls) OI_c * max(K* - K_c, 0)] + [sum(puts) OI_p * max(K_p - K*, 0)]

Iterates over all strikes in the chain.

#### Unusual Activity Detection

Three detection criteria:
1. Volume z-score > 3 (spike vs. chain average)
2. Volume/OI ratio > 0.5 (large intraday positioning relative to open interest)
3. Absolute: volume > 2000 AND OI > 10,000 (whale-level activity)

---

### Module 4 -- src/portfolio.py -- Portfolio Risk Manager

- Storage: SQLite (local file, no server needed)
- Tables: portfolio (positions), trades (history), cash (balance singleton)
- P&L: Real-time using live prices at query time
- Risk Score: Annualised volatility of 3-month daily returns (252 trading days)
  - < 20% -> LOW, 20-35% -> MEDIUM, > 35% -> HIGH
- Auto-alerts: Flags positions where SL/target has been breached

---

### Module 5 -- src/sector_heatmap.py -- Sector Heatmap

Covers 8 sectors with 30+ constituent stocks. Computes average % change and classifies:
- avg change > +0.3% -> BULLISH
- avg change < -0.3% -> BEARISH
- otherwise -> NEUTRAL

---

## Free API Stack

| Purpose | API | Limit | Notes |
|---|---|---|---|
| Live NSE/BSE prices | yfinance (Yahoo Finance) | Unlimited | No key needed |
| Historical OHLCV | yfinance | Full history | All intervals supported |
| Options chain | yfinance .options | Free, NSE supported | Greeks computed locally |
| News & sentiment | NewsAPI.org | 100 req/day (free) | Key in .env |
| Macro indicators | Alpha Vantage | 25 req/day (free) | Key in .env |
| Technical analysis | Custom numpy/pandas | Unlimited | Pure Python, no TA-Lib |
| Greeks | Custom Black-Scholes | Unlimited | From scratch |
| Portfolio storage | SQLite | Unlimited | Local file |

Zero paid APIs. Zero broker account required.

---

## 10 MCP Tools Reference

| # | Tool Name | Key Inputs | Key Outputs |
|---|---|---|---|
| 1 | get_live_price | symbol | price, change%, volume, 52w hi/lo |
| 2 | get_options_chain | symbol, expiry? | CE/PE OI, volume, LTP, Greeks, max pain, PCR |
| 3 | analyze_sentiment | symbol | score (-1 to +1), signal, top headlines |
| 4 | generate_signal | symbol, timeframe? | BUY/SELL/HOLD, confidence 0-100, all indicators |
| 5 | get_portfolio_pnl | (none) | positions, unrealised P&L, risk scores, alerts |
| 6 | place_virtual_trade | symbol, qty, side, stop_loss?, target? | order_id, status, cash balance |
| 7 | calculate_greeks | symbol, strike, expiry, option_type, custom_iv? | delta, gamma, theta, vega, rho, BS price |
| 8 | detect_unusual_activity | symbol | flagged contracts, z-scores, anomaly descriptions |
| 9 | scan_market | universe?, rsi_below?, signal?, pattern?, ... | matching symbols with full analysis |
| 10 | get_sector_heatmap | (none) | per-sector change%, heat, top gainer/loser |

---

## Quick Start

### Prerequisites

- Python 3.11+
- (Optional) NewsAPI.org free key -- for sentiment analysis
- (Optional) Alpha Vantage free key -- for macro indicators

### 1. Clone & install

```bash
git clone https://github.com/Anujyadav911/IndiaQuantmarketassistant.git
cd IndiaQuantmarketassistant

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your free API keys:
# NEWS_API_KEY=abc123...
# ALPHA_VANTAGE_KEY=xyz789...
```

### 3. Run the server (test)

```bash
python main.py
```

You should see: `Starting IndiaQuant MCP server...`
The server listens on stdio -- it is ready for Claude Desktop.

---

## Claude Desktop Integration

Add this block to your Claude Desktop config file:

- macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
- Windows: %APPDATA%\Claude\claude_desktop_config.json

```json
{
  "mcpServers": {
    "indiaquant": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/IndiaQuantmarketassistant/main.py"],
      "env": {
        "NEWS_API_KEY": "your_key_here",
        "ALPHA_VANTAGE_KEY": "your_key_here"
      }
    }
  }
}
```

Restart Claude Desktop. You will see **indiaquant** appear as a connected MCP server.

---

## Example Claude Conversations

```
You: What's the max pain for Nifty this expiry?
Claude: [calls get_options_chain("NIFTY")]
        Max pain for Nifty expiring 2026-03-27 is at 22,200.
        Total Call OI: 4.2 Cr | Total Put OI: 3.8 Cr | PCR: 0.90

You: Is there unusual options activity on Infosys today?
Claude: [calls detect_unusual_activity("INFY")]
        Unusual activity detected on INFY:
        - 1500 CE @ Rs.2.3: Volume 15,420 (z-score: 4.2) -- volume spike
        - 1400 PE @ Rs.8.1: Vol/OI ratio 0.72 -- aggressive put buying

You: Show me my portfolio P&L with risk breakdown
Claude: [calls get_portfolio_pnl()]
        Portfolio Value: Rs.12,34,500 | Cash: Rs.4,50,000
        - HDFC Bank (10 shares): +Rs.2,300 (+1.8%) | Risk: LOW
        - Tata Motors (50 shares): -Rs.4,500 (-3.2%) | Risk: HIGH
        STOP-LOSS ALERT: Tata Motors has breached SL of Rs.720

You: Find oversold IT stocks with RSI below 30
Claude: [calls scan_market(universe="IT", rsi_below=30)]
        Found 2 oversold IT stocks:
        - WIPRO: RSI 27.4, Signal: BUY, Confidence: 68%
        - TECHM: RSI 28.9, Signal: BUY, Confidence: 72%
```

---

## Running Tests

```bash
# All unit tests (no network required)
pytest tests/ -v

# Only unit tests (skip live-data tests)
pytest tests/ -v -m "not live"

# Specific test file
pytest tests/test_black_scholes.py -v

# With coverage
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing
```

### Test Coverage

| Test File | What It Tests | Network? |
|---|---|---|
| test_black_scholes.py | Full Black-Scholes math, IV estimation, put-call parity | No -- Pure math |
| test_signals.py | RSI, MACD, Bollinger, EMA, ATR, sentiment scoring, signal generation | No -- Mocked |
| test_market_data.py | Symbol normalisation, live price parsing, caching | No -- Mocked |
| test_portfolio.py | Virtual trades, P&L, stop-loss alerts, trade history | No -- Mocked |
| test_options_analysis.py | Max-pain calculation, Greeks integration | No -- Pure math |

---

## Design Decisions & Trade-offs

### 1. Pure-Python Black-Scholes (No scipy)

Decision: Implement the normal CDF using the Abramowitz & Stegun rational approximation.
Rationale: Eliminates a heavy dependency, keeps the package lightweight. Error < 7.5e-8 is sufficient for options pricing.
Trade-off: Slightly less accurate than scipy for extreme tails, but negligible for realistic strikes.

### 2. yfinance for All Market Data

Decision: Use yfinance as the sole market-data source.
Rationale: Free, no API key, covers NSE/BSE symbols and indices, full options chain support.
Trade-off: Yahoo Finance has occasional data gaps. The code handles this with multiple fallbacks.

### 3. In-Memory TTL Cache

Decision: Simple dict-based TTL cache instead of Redis.
Rationale: Minimal deployment footprint, suitable for single-user Claude Desktop use.
Trade-off: Not shared across processes. Concurrent multi-user deployments should use Redis.

### 4. Rule-Based Sentiment (No LLM)

Decision: Keyword-based sentiment scoring on headlines.
Rationale: No additional API cost, deterministic, fast, works offline.
Trade-off: Less nuanced than FinBERT. Can be upgraded by replacing _score_headline().

### 5. SQLite for Portfolio Storage

Decision: Single-file SQLite database.
Rationale: Zero configuration, ACID compliant, Python built-in.
Trade-off: Not suitable for concurrent writes. Replace with PostgreSQL for production.

### 6. Technical Indicators from Scratch (No TA-Lib)

Decision: Implement RSI, MACD, Bollinger Bands, ATR using pure pandas/numpy.
Rationale: TA-Lib has C compilation requirements that make installation fragile on many systems.
Trade-off: Slightly slower than TA-Lib C implementation at scale, but negligible for single-symbol analysis.

### 7. Confidence Score Calibration

Decision: Linear mapping of composite score (-100 to +100) -> confidence (0 to 100).
Rationale: Simple, transparent, explainable. The rationale list shows exactly which indicators contributed.
Trade-off: Not a probabilistic model. A backtested ML model would provide more reliable estimates.

---

## Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| yfinance 15-min delay in some free tiers | Price may be delayed outside market hours | Use .NS suffix; real-time during market hours |
| No options data for all NSE stocks | Some mid-caps have no option chains | Tool returns clear error message |
| NewsAPI 100 req/day limit | Sentiment may be stale after limit hit | Results cached 10 min; upgrade for heavy use |
| Historical data gaps on holidays | Some indicators may be NaN | Handled via .dropna() with fallback defaults |
| No real broker connectivity | Virtual trades only | By design -- connect Zerodha/Upstox Kite SDK for live |

---

## Project Structure

```
IndiaQuantmarketassistant/
+-- main.py                    MCP server entry point (10 tools)
+-- src/
|   +-- __init__.py
|   +-- config.py              All constants, API keys, symbol maps
|   +-- market_data.py         Module 1: Live prices, historical OHLCV
|   +-- signals.py             Module 2: RSI/MACD/BB signals + sentiment
|   +-- options.py             Module 3: Black-Scholes, options chain, max pain
|   +-- portfolio.py           Module 4: Virtual portfolio + SQLite P&L
|   +-- sector_heatmap.py      Module 5: Sector % change heatmap
+-- tests/
|   +-- test_black_scholes.py      20 unit tests -- BS math
|   +-- test_signals.py            15 unit tests -- indicators & signals
|   +-- test_market_data.py        10 unit tests -- price fetching & caching
|   +-- test_portfolio.py          12 unit tests -- virtual trading
|   +-- test_options_analysis.py    8 unit tests -- max pain & Greeks
+-- requirements.txt
+-- pyproject.toml
+-- .env.example
+-- README.md
```

---

## Cloud Deployment (24/7 Free Hosting)

No frontend needed — Claude Desktop connects directly to the hosted SSE endpoint.

### Architecture

```
Claude Desktop  ──SSE──►  https://your-app.fly.dev/sse  ──►  IndiaQuant MCP Server
```

Three free platforms are supported. Pick one:

| Platform | Always-on? | Best for | Config file |
|---|---|---|---|
| **Fly.io** ✅ Recommended | Yes (3 free VMs) | India latency (Singapore region) | `fly.toml` |
| **Railway** | Yes ($5/mo free credit) | Simplest setup | `railway.toml` |
| **Render** | No (spins down; fix with UptimeRobot) | CI/CD via GitHub push | `render.yaml` |

---

### Option A — Fly.io (Recommended)

```bash
# 1. Install flyctl
curl -L https://fly.io/install.sh | sh

# 2. Sign up / log in
fly auth signup          # or: fly auth login

# 3. Launch (first time — picks up fly.toml automatically)
fly launch --no-deploy

# 4. Create a volume for the SQLite DB
fly volumes create indiaquant_data --region sin --size 1

# 5. Set secrets (replace with real values)
fly secrets set \
  MCP_AUTH_TOKEN=your-secret-token \
  NEWS_API_KEY=your-newsapi-key \
  ALPHA_VANTAGE_KEY=your-av-key

# 6. Deploy
fly deploy
```

Your SSE endpoint: `https://indiaquant-mcp.fly.dev/sse`

---

### Option B — Railway

```bash
# 1. Push this repo to GitHub

# 2. Go to https://railway.app → New Project → Deploy from GitHub repo

# 3. Railway auto-detects the Dockerfile and railway.toml

# 4. Add environment variables in the Railway dashboard:
#    MCP_AUTH_TOKEN  =  your-secret-token
#    NEWS_API_KEY    =  your-newsapi-key
#    ALPHA_VANTAGE_KEY = your-av-key

# 5. Click Deploy
```

Your SSE endpoint: `https://<auto-generated>.up.railway.app/sse`

---

### Option C — Render (with UptimeRobot keep-alive)

```bash
# 1. Push this repo to GitHub

# 2. Go to https://render.com → New → Web Service → Connect GitHub repo

# 3. Render detects render.yaml automatically

# 4. Set environment variables (marked sync:false in render.yaml):
#    MCP_AUTH_TOKEN, NEWS_API_KEY, ALPHA_VANTAGE_KEY

# 5. Deploy

# 6. Prevent cold starts — set up a free UptimeRobot monitor:
#    → https://uptimerobot.com → New Monitor
#    → Type: HTTP(S)
#    → URL: https://your-app.onrender.com/health
#    → Interval: every 5 minutes
```

Your SSE endpoint: `https://indiaquant-mcp.onrender.com/sse`

---

### Connect Claude Desktop to the deployed server

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "indiaquant": {
      "transport": "sse",
      "url": "https://YOUR-APP-URL/sse",
      "headers": {
        "Authorization": "Bearer YOUR_MCP_AUTH_TOKEN"
      }
    }
  }
}
```

Replace `YOUR-APP-URL` with your actual deployment URL and `YOUR_MCP_AUTH_TOKEN`
with the same token you set as an environment variable.

> **Tip:** If you leave `MCP_AUTH_TOKEN` empty in the server env vars, the
> `Authorization` header is not required.

---

### Local development (stdio transport)

For local use with Claude Desktop, use the original stdio transport — no server
needed, zero latency:

```json
{
  "mcpServers": {
    "indiaquant": {
      "command": "python",
      "args": ["/absolute/path/to/main.py"],
      "env": {
        "NEWS_API_KEY": "your-key",
        "ALPHA_VANTAGE_KEY": "your-key"
      }
    }
  }
}
```

### Health check endpoint

```bash
curl https://YOUR-APP-URL/health
# {"status":"ok","service":"indiaquant-mcp","tools":10,"uptime_seconds":...}
```

---

## Project Structure

```
IndiaQuantmarketassistant/
+-- main.py                    MCP server entry point (10 tools, stdio transport)
+-- server_http.py             Cloud entry point (SSE/HTTP transport)
+-- Dockerfile                 Multi-stage Docker build
+-- .dockerignore
+-- render.yaml                Render.com deployment config
+-- fly.toml                   Fly.io deployment config
+-- railway.toml               Railway.app deployment config
+-- src/
|   +-- __init__.py
|   +-- config.py              All constants, API keys, symbol maps
|   +-- market_data.py         Module 1: Live prices, historical OHLCV
|   +-- signals.py             Module 2: RSI/MACD/BB signals + sentiment
|   +-- options.py             Module 3: Black-Scholes, options chain, max pain
|   +-- portfolio.py           Module 4: Virtual portfolio + SQLite P&L
|   +-- sector_heatmap.py      Module 5: Sector % change heatmap
+-- tests/
|   +-- test_black_scholes.py      26 unit tests -- BS math
|   +-- test_signals.py            19 unit tests -- indicators & signals
|   +-- test_market_data.py        11 unit tests -- price fetching & caching
|   +-- test_portfolio.py          12 unit tests -- virtual trading
|   +-- test_options_analysis.py    7 unit tests -- max pain & Greeks
+-- requirements.txt
+-- pyproject.toml
+-- .env.example
+-- README.md
```

---

## License

MIT -- use freely, build on top, give credit.
