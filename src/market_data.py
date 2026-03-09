"""Live NSE/BSE prices, OHLCV history and Nifty50 batch fetch via Yahoo Finance."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import yfinance as yf

from src.config import (
    HISTORICAL_CACHE_TTL,
    INDEX_MAP,
    NIFTY50_SYMBOLS,
    PRICE_CACHE_TTL,
)

logger = logging.getLogger(__name__)

@dataclass
class _CacheEntry:
    data: Any
    ts: float = field(default_factory=time.time)

_price_cache: dict[str, _CacheEntry] = {}
_hist_cache:  dict[str, _CacheEntry] = {}


def _cache_get(store: dict, key: str, ttl: int) -> Any | None:
    entry = store.get(key)
    if entry and (time.time() - entry.ts) < ttl:
        return entry.data
    return None


def _cache_set(store: dict, key: str, data: Any) -> None:
    store[key] = _CacheEntry(data=data)


def normalise_symbol(symbol: str) -> str:
    """Convert user-supplied ticker to a yfinance-compatible string.

    Examples
    --------
    >>> normalise_symbol("RELIANCE")   → "RELIANCE.NS"
    >>> normalise_symbol("NIFTY")      → "^NSEI"
    >>> normalise_symbol("RELIANCE.NS") → "RELIANCE.NS"
    >>> normalise_symbol("RELIANCE.BO") → "RELIANCE.BO"
    """
    sym = symbol.strip().upper()

    # Already has an exchange suffix or is an index
    if sym.startswith("^") or sym.endswith(".NS") or sym.endswith(".BO"):
        return sym

    # Known index aliases
    if sym in INDEX_MAP:
        return INDEX_MAP[sym]

    # Default: NSE
    return f"{sym}.NS"



def get_live_price(symbol: str) -> dict:
    """Return a live quote dict for *symbol*.

    Returns
    -------
    {
        "symbol":        str,
        "ticker":        str,           # yfinance ticker used
        "price":         float,
        "open":          float,
        "high":          float,
        "low":           float,
        "prev_close":    float,
        "change":        float,         # absolute change from prev_close
        "change_pct":    float,         # % change
        "volume":        int,
        "market_cap":    float | None,
        "52w_high":      float | None,
        "52w_low":       float | None,
        "currency":      str,
        "exchange":      str,
        "timestamp":     str,           # ISO-8601
    }
    """
    ticker_str = normalise_symbol(symbol)
    cached = _cache_get(_price_cache, ticker_str, PRICE_CACHE_TTL)
    if cached:
        logger.debug("Price cache hit: %s", ticker_str)
        return cached

    logger.info("Fetching live price for %s", ticker_str)
    tkr = yf.Ticker(ticker_str)
    info = tkr.info  # single-call dict from Yahoo Finance

    # Yahoo Finance may return different field names depending on asset type.
    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("ask")
        or info.get("bid")
        or 0.0
    )
    if price == 0.0:
        # Fallback: fetch 1-day bar
        hist = tkr.history(period="1d", interval="1m")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])

    prev_close: float = info.get("previousClose") or info.get("regularMarketPreviousClose") or price
    change     = round(price - prev_close, 4)
    change_pct = round((change / prev_close * 100) if prev_close else 0.0, 4)

    result = {
        "symbol":     symbol.upper(),
        "ticker":     ticker_str,
        "price":      round(float(price), 2),
        "open":       round(float(info.get("open") or info.get("regularMarketOpen") or price), 2),
        "high":       round(float(info.get("dayHigh") or info.get("regularMarketDayHigh") or price), 2),
        "low":        round(float(info.get("dayLow") or info.get("regularMarketDayLow") or price), 2),
        "prev_close": round(float(prev_close), 2),
        "change":     change,
        "change_pct": change_pct,
        "volume":     int(info.get("volume") or info.get("regularMarketVolume") or 0),
        "market_cap": info.get("marketCap"),
        "52w_high":   info.get("fiftyTwoWeekHigh"),
        "52w_low":    info.get("fiftyTwoWeekLow"),
        "currency":   info.get("currency", "INR"),
        "exchange":   info.get("exchange", "NSE"),
        "timestamp":  pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
    }

    _cache_set(_price_cache, ticker_str, result)
    return result



def get_historical_data(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV data for *symbol*.

    Parameters
    ----------
    symbol   : NSE/BSE symbol or index alias.
    period   : yfinance period string  e.g. "1d","5d","1mo","3mo","6mo","1y","2y","5y"
    interval : bar size                e.g. "1m","5m","15m","1h","1d","1wk"

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
    Index: DatetimeIndex (UTC)
    """
    ticker_str = normalise_symbol(symbol)
    cache_key  = f"{ticker_str}_{period}_{interval}"
    cached     = _cache_get(_hist_cache, cache_key, HISTORICAL_CACHE_TTL)
    if cached is not None:
        logger.debug("Historical cache hit: %s", cache_key)
        return cached

    logger.info("Fetching historical data: %s | period=%s interval=%s", ticker_str, period, interval)
    tkr  = yf.Ticker(ticker_str)
    hist = tkr.history(period=period, interval=interval, auto_adjust=True)

    if hist.empty:
        logger.warning("Empty historical data for %s", ticker_str)
        return pd.DataFrame()

    _cache_set(_hist_cache, cache_key, hist)
    return hist



def get_nifty50_prices() -> list[dict]:
    """Return live quotes for all Nifty-50 constituents.

    Uses yfinance batch download for efficiency, then enriches with
    individual quote metadata from _price_cache where available.

    Returns
    -------
    list of dicts, each matching the schema from get_live_price().
    """
    tickers = [f"{sym}.NS" for sym in NIFTY50_SYMBOLS]
    logger.info("Batch-fetching Nifty-50 prices (%d symbols)", len(tickers))

    raw = yf.download(
        tickers,
        period="2d",        # need previous close for % change
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    results: list[dict] = []
    for sym, ticker_str in zip(NIFTY50_SYMBOLS, tickers):
        try:
            df = raw[ticker_str] if len(tickers) > 1 else raw
            if df.empty or len(df) < 1:
                continue
            latest    = df.iloc[-1]
            prev_row  = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
            price     = float(latest["Close"])
            prev_close = float(prev_row["Close"])
            change    = round(price - prev_close, 2)
            change_pct = round((change / prev_close * 100) if prev_close else 0.0, 2)

            results.append({
                "symbol":     sym,
                "ticker":     ticker_str,
                "price":      round(price, 2),
                "change":     change,
                "change_pct": change_pct,
                "volume":     int(latest.get("Volume", 0)),
                "timestamp":  pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
            })
        except Exception as exc:
            logger.warning("Could not get price for %s: %s", sym, exc)

    return results



def get_sector_prices(sector_symbols: list[str]) -> list[dict]:
    """Convenience wrapper: live prices for a given list of symbols."""
    results = []
    for sym in sector_symbols:
        try:
            results.append(get_live_price(sym))
        except Exception as exc:
            logger.warning("Skipping %s: %s", sym, exc)
    return results


def clear_caches() -> None:
    """Wipe all in-memory caches (useful in tests)."""
    _price_cache.clear()
    _hist_cache.clear()
