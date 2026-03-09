"""RSI/MACD/Bollinger + news sentiment → BUY/SELL/HOLD signal with confidence score."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import requests

from src.config import (
    COMPANY_NAMES,
    NEWS_API_KEY,
    NEWS_CACHE_TTL,
    SIGNAL_CACHE_TTL,
)
from src.market_data import get_historical_data

logger = logging.getLogger(__name__)

@dataclass
class _CE:
    data: dict
    ts: float = field(default_factory=time.time)

_signal_cache: dict[str, _CE] = {}
_news_cache:   dict[str, _CE] = {}

def _cget(store: dict, key: str, ttl: int):
    e = store.get(key)
    return e.data if e and (time.time() - e.ts) < ttl else None

def _cset(store: dict, key: str, data: dict):
    store[key] = _CE(data=data)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta   = close.diff()
    gain    = delta.clip(lower=0)
    loss    = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    # Avoid division by zero: if avg_loss is 0, RSI is 100 (all gains)
    rs = avg_gain / avg_loss.replace(0, float('nan'))
    rsi = 100 - (100 / (1 + rs))
    # Where avg_loss was exactly 0 (pure uptrend), RS is nan → set RSI = 100
    rsi = rsi.where(avg_loss != 0, other=100.0)
    return rsi


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram)."""
    ema_fast   = close.ewm(span=fast, adjust=False).mean()
    ema_slow   = close.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (upper_band, middle_band, lower_band)."""
    middle = close.rolling(period).mean()
    std    = close.rolling(period).std()
    return middle + num_std * std, middle, middle - num_std * std


def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _detect_patterns(df: pd.DataFrame) -> list[str]:
    """Detect simple chart patterns from OHLCV DataFrame."""
    patterns: list[str] = []
    if len(df) < 50:
        return patterns

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # ── Double Top ──────────────────────────────────────────────────────────
    recent = close.iloc[-30:]
    rolling_max = recent.rolling(5).max()
    peaks = rolling_max[rolling_max == recent]
    if len(peaks) >= 2:
        p1, p2 = peaks.iloc[-2], peaks.iloc[-1]
        if abs(p1 - p2) / p1 < 0.02:  # within 2 %
            patterns.append("Double Top (Bearish)")

    # ── Double Bottom ───────────────────────────────────────────────────────
    rolling_min = recent.rolling(5).min()
    troughs = rolling_min[rolling_min == recent]
    if len(troughs) >= 2:
        t1, t2 = troughs.iloc[-2], troughs.iloc[-1]
        if abs(t1 - t2) / t1 < 0.02:
            patterns.append("Double Bottom (Bullish)")

    # ── Golden Cross / Death Cross ───────────────────────────────────────────
    sma50  = _sma(close, 50)
    sma200 = _sma(close, 200)
    if not sma50.empty and not sma200.empty:
        if sma50.iloc[-1] > sma200.iloc[-1] and sma50.iloc[-2] <= sma200.iloc[-2]:
            patterns.append("Golden Cross (Bullish)")
        elif sma50.iloc[-1] < sma200.iloc[-1] and sma50.iloc[-2] >= sma200.iloc[-2]:
            patterns.append("Death Cross (Bearish)")

    # ── Bullish / Bearish Engulfing (last 2 candles) ─────────────────────────
    if len(df) >= 2:
        o1, c1 = df["Open"].iloc[-2], df["Close"].iloc[-2]
        o2, c2 = df["Open"].iloc[-1], df["Close"].iloc[-1]
        if c1 < o1 and c2 > o2 and c2 > o1 and o2 < c1:  # bull engulf
            patterns.append("Bullish Engulfing")
        elif c1 > o1 and c2 < o2 and c2 < o1 and o2 > c1:  # bear engulf
            patterns.append("Bearish Engulfing")

    # ── RSI Divergence ───────────────────────────────────────────────────────
    rsi_series = _rsi(close)
    if len(rsi_series.dropna()) >= 5:
        price_trend = close.iloc[-5:].mean() > close.iloc[-10:-5].mean()
        rsi_trend   = rsi_series.iloc[-5:].mean() > rsi_series.iloc[-10:-5].mean()
        if price_trend and not rsi_trend:
            patterns.append("Bearish RSI Divergence")
        elif not price_trend and rsi_trend:
            patterns.append("Bullish RSI Divergence")

    return patterns


# keyword dictionaries for rule-based sentiment scoring
_BULLISH_WORDS = {
    "upgrade", "buy", "outperform", "strong", "growth", "profit", "beat",
    "record", "surge", "rally", "bullish", "positive", "gain", "rise",
    "expand", "win", "award", "milestone", "launch", "partnership",
    "revenue", "earnings", "dividend", "acquisition", "target raised",
    "overweight", "accumulate", "top pick",
}
_BEARISH_WORDS = {
    "downgrade", "sell", "underperform", "weak", "loss", "miss", "decline",
    "fall", "bearish", "negative", "debt", "lawsuit", "fraud", "penalty",
    "cut", "lay off", "layoff", "restructure", "warning", "concern",
    "underweight", "reduce", "avoid", "below expectations",
}


def _score_headline(headline: str) -> float:
    """Return a score in [-1, +1] for a single headline."""
    text = headline.lower()
    bull = sum(1 for w in _BULLISH_WORDS if w in text)
    bear = sum(1 for w in _BEARISH_WORDS if w in text)
    total = bull + bear
    if total == 0:
        return 0.0
    return round((bull - bear) / total, 4)


def _fetch_news(query: str, page_size: int = 10) -> list[dict]:
    """Fetch recent news from NewsAPI.org."""
    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY not set — skipping NewsAPI call")
        return []
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q":        query,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": page_size,
            "apiKey":   NEWS_API_KEY,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("articles", [])
    except Exception as exc:
        logger.error("NewsAPI error: %s", exc)
        return []


def analyse_sentiment(symbol: str) -> dict:
    """Fetch news headlines and compute sentiment for *symbol*.

    Returns
    -------
    {
        "symbol":      str,
        "score":       float,    # –1 (very bearish) … +1 (very bullish)
        "signal":      str,      # "BULLISH" | "BEARISH" | "NEUTRAL"
        "headline_count": int,
        "headlines":   list[{"title": str, "score": float, "url": str}],
        "summary":     str,
        "timestamp":   str,
    }
    """
    cached = _cget(_news_cache, symbol.upper(), NEWS_CACHE_TTL)
    if cached:
        return cached

    company = COMPANY_NAMES.get(symbol.upper(), symbol.upper())
    articles = _fetch_news(f"{company} stock NSE India")

    scored: list[dict] = []
    for art in articles:
        title = art.get("title", "") or ""
        score = _score_headline(title)
        scored.append({
            "title": title,
            "score": score,
            "url":   art.get("url", ""),
        })

    avg_score = round(sum(h["score"] for h in scored) / len(scored), 4) if scored else 0.0
    if avg_score > 0.1:
        signal = "BULLISH"
    elif avg_score < -0.1:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    result = {
        "symbol":         symbol.upper(),
        "score":          avg_score,
        "signal":         signal,
        "headline_count": len(scored),
        "headlines":      scored[:5],   # top 5 for brevity
        "summary":        (
            f"{company}: {len(scored)} headlines analysed. "
            f"Average sentiment score {avg_score:+.2f} → {signal}."
        ),
        "timestamp": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
    }
    _cset(_news_cache, symbol.upper(), result)
    return result


_TIMEFRAME_MAP = {
    "1d":  ("1mo",  "1d"),
    "1w":  ("6mo",  "1d"),
    "1mo": ("1y",   "1wk"),
    "15m": ("5d",   "15m"),
    "1h":  ("1mo",  "1h"),
}


def generate_signal(symbol: str, timeframe: str = "1d") -> dict:
    """Generate a BUY/SELL/HOLD signal with confidence for *symbol*.

    Algorithm
    ---------
    1. Fetch OHLCV data for the requested timeframe.
    2. Compute RSI, MACD, Bollinger Bands, EMA(9/21), SMA(50/200), ATR.
    3. Score each indicator on a –100 … +100 scale.
    4. Detect chart patterns and adjust score.
    5. Run sentiment analysis and add a weighted sentiment score.
    6. Map composite score → BUY / SELL / HOLD + confidence %.

    Returns
    -------
    {
        "symbol":       str,
        "timeframe":    str,
        "signal":       "BUY" | "SELL" | "HOLD",
        "confidence":   float,   # 0–100
        "composite_score": float,
        "indicators": {
            "rsi": float,
            "macd": float,
            "macd_signal": float,
            "macd_histogram": float,
            "bb_upper": float, "bb_mid": float, "bb_lower": float,
            "price": float,
            "ema9": float, "ema21": float,
            "sma50": float, "sma200": float,
            "atr": float,
        },
        "patterns":     list[str],
        "sentiment":    dict,
        "rationale":    list[str],
        "timestamp":    str,
    }
    """
    cache_key = f"{symbol.upper()}_{timeframe}"
    cached = _cget(_signal_cache, cache_key, SIGNAL_CACHE_TTL)
    if cached:
        return cached

    tf = timeframe.lower()
    period, interval = _TIMEFRAME_MAP.get(tf, ("6mo", "1d"))
    df = get_historical_data(symbol, period=period, interval=interval)

    if df.empty or len(df) < 30:
        result = {
            "symbol":    symbol.upper(),
            "timeframe": tf,
            "signal":    "HOLD",
            "confidence": 0.0,
            "error":     "Insufficient historical data",
            "timestamp": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        }
        _cset(_signal_cache, cache_key, result)
        return result

    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)

    # ── Compute indicators ───────────────────────────────────────────────────
    rsi_s        = _rsi(close)
    macd_l, sig_l, hist_s = _macd(close)
    bb_up, bb_mid, bb_low = _bollinger(close)
    ema9  = _ema(close, 9)
    ema21 = _ema(close, 21)
    sma50 = _sma(close, 50)
    sma200= _sma(close, 200)
    atr_s = _atr(high, low, close)

    rsi_val  = float(rsi_s.dropna().iloc[-1])   if not rsi_s.dropna().empty   else 50.0
    macd_val = float(macd_l.dropna().iloc[-1])  if not macd_l.dropna().empty  else 0.0
    sig_val  = float(sig_l.dropna().iloc[-1])   if not sig_l.dropna().empty   else 0.0
    hist_val = float(hist_s.dropna().iloc[-1])  if not hist_s.dropna().empty  else 0.0
    bb_u     = float(bb_up.dropna().iloc[-1])   if not bb_up.dropna().empty   else 0.0
    bb_m     = float(bb_mid.dropna().iloc[-1])  if not bb_mid.dropna().empty  else 0.0
    bb_l     = float(bb_low.dropna().iloc[-1])  if not bb_low.dropna().empty  else 0.0
    ema9_v   = float(ema9.iloc[-1])
    ema21_v  = float(ema21.iloc[-1])
    sma50_v  = float(sma50.dropna().iloc[-1])   if not sma50.dropna().empty   else 0.0
    sma200_v = float(sma200.dropna().iloc[-1])  if not sma200.dropna().empty  else 0.0
    atr_v    = float(atr_s.dropna().iloc[-1])   if not atr_s.dropna().empty   else 0.0
    price    = float(close.iloc[-1])

    # ── Score each indicator (bullish = positive, bearish = negative) ────────
    scores: list[float] = []
    rationale: list[str] = []

    # RSI: below 30 = oversold (bull), above 70 = overbought (bear)
    if rsi_val < 30:
        s = min(40.0, (30 - rsi_val) * 2)
        scores.append(s)
        rationale.append(f"RSI {rsi_val:.1f} – oversold, potential bounce (+{s:.0f})")
    elif rsi_val > 70:
        s = -min(40.0, (rsi_val - 70) * 2)
        scores.append(s)
        rationale.append(f"RSI {rsi_val:.1f} – overbought, possible reversal ({s:.0f})")
    elif 40 <= rsi_val <= 60:
        scores.append(5.0)
        rationale.append(f"RSI {rsi_val:.1f} – neutral momentum (+5)")
    else:
        scores.append(0.0)

    # MACD histogram direction
    if hist_val > 0:
        scores.append(15.0)
        rationale.append(f"MACD histogram positive ({hist_val:+.4f}) – bullish momentum (+15)")
    else:
        scores.append(-15.0)
        rationale.append(f"MACD histogram negative ({hist_val:+.4f}) – bearish momentum (-15)")

    # MACD line vs signal line
    if macd_val > sig_val:
        scores.append(10.0)
        rationale.append("MACD line above signal – bullish cross (+10)")
    else:
        scores.append(-10.0)
        rationale.append("MACD line below signal – bearish cross (-10)")

    # Bollinger Band position
    bb_pct = (price - bb_l) / (bb_u - bb_l + 1e-9)
    if bb_pct < 0.2:
        scores.append(20.0)
        rationale.append(f"Price near lower Bollinger Band ({bb_pct:.0%}) – potential long (+20)")
    elif bb_pct > 0.8:
        scores.append(-20.0)
        rationale.append(f"Price near upper Bollinger Band ({bb_pct:.0%}) – potential short (-20)")
    else:
        scores.append(0.0)

    # EMA9 vs EMA21 crossover
    if ema9_v > ema21_v:
        scores.append(10.0)
        rationale.append("EMA9 > EMA21 – short-term uptrend (+10)")
    else:
        scores.append(-10.0)
        rationale.append("EMA9 < EMA21 – short-term downtrend (-10)")

    # Price vs SMA50 (medium-term trend)
    if sma50_v and price > sma50_v:
        scores.append(10.0)
        rationale.append("Price above SMA50 – medium-term bullish (+10)")
    elif sma50_v:
        scores.append(-10.0)
        rationale.append("Price below SMA50 – medium-term bearish (-10)")

    # Price vs SMA200 (long-term trend)
    if sma200_v and price > sma200_v:
        scores.append(10.0)
        rationale.append("Price above SMA200 – long-term bull market (+10)")
    elif sma200_v:
        scores.append(-10.0)
        rationale.append("Price below SMA200 – long-term bear market (-10)")

    # ── Chart patterns (±15 each) ────────────────────────────────────────────
    patterns = _detect_patterns(df)
    for pat in patterns:
        if "Bullish" in pat or "Bottom" in pat or "Golden" in pat:
            scores.append(15.0)
            rationale.append(f"Pattern: {pat} (+15)")
        elif "Bearish" in pat or "Top" in pat or "Death" in pat:
            scores.append(-15.0)
            rationale.append(f"Pattern: {pat} (-15)")

    # ── Sentiment (weighted 20 % of total) ───────────────────────────────────
    sentiment = analyse_sentiment(symbol)
    sent_score = sentiment["score"] * 20  # maps [–1,+1] → [–20,+20]
    scores.append(sent_score)
    rationale.append(
        f"Sentiment score {sentiment['score']:+.2f} ({sentiment['signal']}) "
        f"→ weighted {sent_score:+.1f}"
    )

    # ── Composite score ───────────────────────────────────────────────────────
    composite = sum(scores)
    # Clamp to [–100, +100]
    composite = max(-100.0, min(100.0, composite))

    # Map composite → signal + confidence
    if composite >= 20:
        signal = "BUY"
        confidence = min(100.0, 50.0 + composite)
    elif composite <= -20:
        signal = "SELL"
        confidence = min(100.0, 50.0 + abs(composite))
    else:
        signal = "HOLD"
        confidence = max(0.0, 50.0 - abs(composite))

    confidence = round(confidence, 1)

    result = {
        "symbol":    symbol.upper(),
        "timeframe": tf,
        "signal":    signal,
        "confidence": confidence,
        "composite_score": round(composite, 2),
        "indicators": {
            "rsi":            round(rsi_val, 2),
            "macd":           round(macd_val, 4),
            "macd_signal":    round(sig_val, 4),
            "macd_histogram": round(hist_val, 4),
            "bb_upper":       round(bb_u, 2),
            "bb_mid":         round(bb_m, 2),
            "bb_lower":       round(bb_l, 2),
            "price":          round(price, 2),
            "ema9":           round(ema9_v, 2),
            "ema21":          round(ema21_v, 2),
            "sma50":          round(sma50_v, 2),
            "sma200":         round(sma200_v, 2),
            "atr":            round(atr_v, 2),
        },
        "patterns":  patterns,
        "sentiment": sentiment,
        "rationale": rationale,
        "timestamp": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
    }
    _cset(_signal_cache, cache_key, result)
    return result


def scan_market(
    symbols: list[str],
    criteria: dict,
    timeframe: str = "1d",
) -> list[dict]:
    """Screen a list of symbols against filter *criteria*.

    Supported criteria keys
    -----------------------
    rsi_below : float       – RSI must be < value
    rsi_above : float       – RSI must be > value
    signal    : str         – "BUY" | "SELL" | "HOLD"
    min_confidence : float  – confidence must be >= value
    pattern   : str         – pattern substring match (case-insensitive)

    Returns
    -------
    list of dicts, each a full generate_signal() result that matched.
    """
    matches = []
    for sym in symbols:
        try:
            sig = generate_signal(sym, timeframe)
        except Exception as exc:
            logger.warning("Scan skipping %s: %s", sym, exc)
            continue

        indicators = sig.get("indicators", {})
        passed = True

        if "rsi_below" in criteria:
            if indicators.get("rsi", 100) >= criteria["rsi_below"]:
                passed = False
        if "rsi_above" in criteria:
            if indicators.get("rsi", 0) <= criteria["rsi_above"]:
                passed = False
        if "signal" in criteria:
            if sig.get("signal") != criteria["signal"].upper():
                passed = False
        if "min_confidence" in criteria:
            if sig.get("confidence", 0) < criteria["min_confidence"]:
                passed = False
        if "pattern" in criteria:
            pat_filter = criteria["pattern"].lower()
            if not any(pat_filter in p.lower() for p in sig.get("patterns", [])):
                passed = False

        if passed:
            matches.append(sig)

    return matches
