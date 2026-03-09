"""IndiaQuant MCP Server — 10 tools over stdio for Claude Desktop."""

from __future__ import annotations

import json
import logging
import os

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from src.config import NIFTY50_SYMBOLS, SECTOR_SYMBOLS
from src.market_data import get_live_price, get_nifty50_prices
from src.options import (
    calculate_greeks,
    detect_unusual_activity,
    get_options_chain,
)
from src.portfolio import (
    get_portfolio_pnl,
    get_trade_history,
    init_db,
    place_virtual_trade,
)
from src.sector_heatmap import get_sector_heatmap
from src.signals import analyse_sentiment, generate_signal, scan_market

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO"), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("indiaquant.mcp")

server = Server("indiaquant-mcp")


# ══════════════════════════════════════════════════════════════════════════════
# Tool Definitions
# ══════════════════════════════════════════════════════════════════════════════

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        # ── 1. get_live_price ────────────────────────────────────────────────
        types.Tool(
            name="get_live_price",
            description=(
                "Fetch real-time price, % change, and volume for any NSE/BSE stock "
                "or index (NIFTY, BANKNIFTY, SENSEX, etc.). "
                "Uses yfinance – live data, no cache delay > 30 s."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": (
                            "NSE/BSE ticker (e.g. RELIANCE, HDFCBANK, TCS) "
                            "or index alias (NIFTY, BANKNIFTY, SENSEX)."
                        ),
                    }
                },
                "required": ["symbol"],
            },
        ),

        # ── 2. get_options_chain ─────────────────────────────────────────────
        types.Tool(
            name="get_options_chain",
            description=(
                "Fetch live options chain for a symbol with CE/PE OI, volume, "
                "LTP, IV, and full Greeks (Delta, Gamma, Theta, Vega). "
                "Also returns Max Pain and PCR for the expiry."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Underlying symbol (e.g. NIFTY, BANKNIFTY, RELIANCE).",
                    },
                    "expiry": {
                        "type": "string",
                        "description": "Expiry date as YYYY-MM-DD. Omit for nearest expiry.",
                    },
                },
                "required": ["symbol"],
            },
        ),

        # ── 3. analyze_sentiment ─────────────────────────────────────────────
        types.Tool(
            name="analyze_sentiment",
            description=(
                "Fetch recent news headlines for a stock and compute a rule-based "
                "sentiment score (–1 = very bearish → +1 = very bullish). "
                "Returns top headlines with individual scores."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "NSE ticker (e.g. INFY, RELIANCE, SBIN).",
                    }
                },
                "required": ["symbol"],
            },
        ),

        # ── 4. generate_signal ───────────────────────────────────────────────
        types.Tool(
            name="generate_signal",
            description=(
                "Generate a BUY / SELL / HOLD signal with a 0-100 confidence score "
                "by combining: RSI, MACD, Bollinger Bands, EMA crossovers, SMA trend, "
                "chart-pattern detection, and news sentiment. "
                "Supports multiple timeframes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "NSE/BSE ticker.",
                    },
                    "timeframe": {
                        "type": "string",
                        "description": (
                            "Analysis timeframe: '15m', '1h', '1d' (default), '1w', '1mo'."
                        ),
                        "default": "1d",
                    },
                },
                "required": ["symbol"],
            },
        ),

        # ── 5. get_portfolio_pnl ─────────────────────────────────────────────
        types.Tool(
            name="get_portfolio_pnl",
            description=(
                "Show real-time P&L for all open virtual positions. "
                "Includes unrealised P&L per position, risk scores, "
                "stop-loss / target alerts, and overall portfolio value."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),

        # ── 6. place_virtual_trade ───────────────────────────────────────────
        types.Tool(
            name="place_virtual_trade",
            description=(
                "Place a simulated BUY or SELL order at the current live price. "
                "Manages cash balance, opens/closes positions, and records P&L. "
                "Optional stop-loss and target prices for auto-management."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "NSE/BSE ticker to trade.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of shares.",
                    },
                    "side": {
                        "type": "string",
                        "description": "'BUY' or 'SELL'.",
                        "enum": ["BUY", "SELL"],
                    },
                    "stop_loss": {
                        "type": "number",
                        "description": "Optional stop-loss price.",
                    },
                    "target": {
                        "type": "number",
                        "description": "Optional target price.",
                    },
                },
                "required": ["symbol", "quantity", "side"],
            },
        ),

        # ── 7. calculate_greeks ──────────────────────────────────────────────
        types.Tool(
            name="calculate_greeks",
            description=(
                "Calculate full Black-Scholes Greeks (Delta, Gamma, Theta, Vega, Rho) "
                "for a specific options contract. IV is derived from the live options "
                "chain or can be overridden. Pure mathematical implementation — no library."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Underlying symbol (e.g. NIFTY, RELIANCE).",
                    },
                    "strike": {
                        "type": "number",
                        "description": "Strike price of the option.",
                    },
                    "expiry": {
                        "type": "string",
                        "description": "Expiry date as YYYY-MM-DD.",
                    },
                    "option_type": {
                        "type": "string",
                        "description": "'CE' for Call, 'PE' for Put.",
                        "enum": ["CE", "PE"],
                    },
                    "custom_iv": {
                        "type": "number",
                        "description": "Optional: override implied volatility (decimal, e.g. 0.18 for 18%).",
                    },
                },
                "required": ["symbol", "strike", "expiry", "option_type"],
            },
        ),

        # ── 8. detect_unusual_activity ───────────────────────────────────────
        types.Tool(
            name="detect_unusual_activity",
            description=(
                "Scan the live options chain for unusual volume/OI spikes that "
                "may indicate large institutional activity. Returns flagged contracts "
                "with z-scores and anomaly descriptions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "NSE/BSE ticker (e.g. INFY, NIFTY).",
                    }
                },
                "required": ["symbol"],
            },
        ),

        # ── 9. scan_market ───────────────────────────────────────────────────
        types.Tool(
            name="scan_market",
            description=(
                "Scan the market (Nifty 50 or a specific sector) using custom filters "
                "like RSI range, signal type, confidence threshold, or chart pattern. "
                "Returns all matching stocks with full analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "universe": {
                        "type": "string",
                        "description": (
                            "Stock universe to scan: 'nifty50' (default) or a sector name "
                            "like 'IT', 'Banking', 'Auto', 'Pharma', 'Energy', 'FMCG', "
                            "'Metals', 'Infrastructure'."
                        ),
                        "default": "nifty50",
                    },
                    "rsi_below": {
                        "type": "number",
                        "description": "Filter: RSI must be below this value (e.g. 30 for oversold).",
                    },
                    "rsi_above": {
                        "type": "number",
                        "description": "Filter: RSI must be above this value (e.g. 70 for overbought).",
                    },
                    "signal": {
                        "type": "string",
                        "description": "Filter: required signal type — 'BUY', 'SELL', or 'HOLD'.",
                        "enum": ["BUY", "SELL", "HOLD"],
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Filter: minimum confidence score (0-100).",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Filter: chart pattern substring (e.g. 'Bullish', 'Golden Cross').",
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Analysis timeframe. Default '1d'.",
                        "default": "1d",
                    },
                },
                "required": [],
            },
        ),

        # ── 10. get_sector_heatmap ───────────────────────────────────────────
        types.Tool(
            name="get_sector_heatmap",
            description=(
                "Get a real-time heatmap of all major NSE sectors: IT, Banking, Auto, "
                "Pharma, Energy, FMCG, Metals, Infrastructure. "
                "Shows avg % change, heat status, top gainer/loser per sector."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Tool Handlers
# ══════════════════════════════════════════════════════════════════════════════

@server.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:

    logger.info("Tool called: %s | args: %s", name, arguments)

    try:
        result = await _dispatch(name, arguments)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    except Exception as exc:
        logger.exception("Tool %s raised: %s", name, exc)
        error_payload = {
            "error":   str(exc),
            "tool":    name,
            "details": "Check server logs for full traceback.",
        }
        return [types.TextContent(type="text", text=json.dumps(error_payload, indent=2))]


async def _dispatch(name: str, args: dict) -> dict:
    """Route tool name → implementation function."""

    # ── 1. get_live_price ────────────────────────────────────────────────────
    if name == "get_live_price":
        symbol = args.get("symbol", "").strip()
        if not symbol:
            return {"error": "symbol is required"}
        return get_live_price(symbol)

    # ── 2. get_options_chain ─────────────────────────────────────────────────
    if name == "get_options_chain":
        symbol = args.get("symbol", "").strip()
        expiry = args.get("expiry")
        if not symbol:
            return {"error": "symbol is required"}
        return get_options_chain(symbol, expiry)

    # ── 3. analyze_sentiment ─────────────────────────────────────────────────
    if name in ("analyze_sentiment", "analyse_sentiment"):
        symbol = args.get("symbol", "").strip()
        if not symbol:
            return {"error": "symbol is required"}
        return analyse_sentiment(symbol)

    # ── 4. generate_signal ───────────────────────────────────────────────────
    if name == "generate_signal":
        symbol    = args.get("symbol", "").strip()
        timeframe = args.get("timeframe", "1d")
        if not symbol:
            return {"error": "symbol is required"}
        return generate_signal(symbol, timeframe)

    # ── 5. get_portfolio_pnl ─────────────────────────────────────────────────
    if name == "get_portfolio_pnl":
        return get_portfolio_pnl()

    # ── 6. place_virtual_trade ───────────────────────────────────────────────
    if name == "place_virtual_trade":
        symbol   = args.get("symbol", "").strip()
        quantity = int(args.get("quantity", 0))
        side     = args.get("side", "BUY").upper()
        sl       = args.get("stop_loss")
        tgt      = args.get("target")
        if not symbol or quantity <= 0:
            return {"error": "symbol and positive quantity are required"}
        return place_virtual_trade(
            symbol, quantity, side,
            stop_loss=float(sl) if sl is not None else None,
            target=float(tgt)   if tgt is not None else None,
        )

    # ── 7. calculate_greeks ──────────────────────────────────────────────────
    if name == "calculate_greeks":
        symbol      = args.get("symbol", "").strip()
        strike      = args.get("strike")
        expiry      = args.get("expiry", "").strip()
        option_type = args.get("option_type", "CE").upper()
        custom_iv   = args.get("custom_iv")
        if not symbol or strike is None or not expiry:
            return {"error": "symbol, strike, expiry, and option_type are required"}
        return calculate_greeks(
            symbol, float(strike), expiry, option_type,
            custom_iv=float(custom_iv) if custom_iv is not None else None,
        )

    # ── 8. detect_unusual_activity ───────────────────────────────────────────
    if name == "detect_unusual_activity":
        symbol = args.get("symbol", "").strip()
        if not symbol:
            return {"error": "symbol is required"}
        return detect_unusual_activity(symbol)

    # ── 9. scan_market ───────────────────────────────────────────────────────
    if name == "scan_market":
        universe  = args.get("universe", "nifty50").strip().lower()
        timeframe = args.get("timeframe", "1d")

        # Select symbol universe
        if universe == "nifty50":
            symbols = NIFTY50_SYMBOLS
        elif universe in {k.lower() for k in SECTOR_SYMBOLS}:
            # case-insensitive lookup
            matched = next(k for k in SECTOR_SYMBOLS if k.lower() == universe)
            symbols = SECTOR_SYMBOLS[matched]
        else:
            # Fallback: treat as single symbol list
            symbols = [universe.upper()]

        # Build criteria dict from optional filter args
        criteria: dict = {}
        for key in ("rsi_below", "rsi_above", "min_confidence"):
            if key in args and args[key] is not None:
                criteria[key] = float(args[key])
        if "signal" in args and args["signal"]:
            criteria["signal"] = args["signal"].upper()
        if "pattern" in args and args["pattern"]:
            criteria["pattern"] = args["pattern"]

        matches = scan_market(symbols, criteria, timeframe)
        return {
            "universe":   universe,
            "total_scanned": len(symbols),
            "matches_found": len(matches),
            "criteria":   criteria,
            "results":    matches,
            "timestamp":  __import__("pandas").Timestamp.now(tz="Asia/Kolkata").isoformat(),
        }

    # ── 10. get_sector_heatmap ───────────────────────────────────────────────
    if name == "get_sector_heatmap":
        return get_sector_heatmap()

    return {"error": f"Unknown tool: {name}"}


# ══════════════════════════════════════════════════════════════════════════════
# Server entrypoint
# ══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    """Start the MCP server over stdio."""
    init_db()
    logger.info("Starting IndiaQuant MCP server…")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run() -> None:
    import asyncio
    asyncio.run(main())


if __name__ == "__main__":
    run()
