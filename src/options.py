"""Live options chain, Black-Scholes Greeks from scratch, max pain and unusual activity."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import OPTIONS_CACHE_TTL, RISK_FREE_RATE
from src.market_data import normalise_symbol

logger = logging.getLogger(__name__)


@dataclass
class _CE:
    data: dict
    ts: float = field(default_factory=time.time)

_opts_cache: dict[str, _CE] = {}

def _cget(key: str) -> Optional[dict]:
    e = _opts_cache.get(key)
    return e.data if e and (time.time() - e.ts) < OPTIONS_CACHE_TTL else None

def _cset(key: str, data: dict) -> None:
    _opts_cache[key] = _CE(data=data)


def _norm_pdf(x: float) -> float:
    """Standard normal PDF — pure math, no scipy."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via Abramowitz & Stegun rational approximation.

    Maximum error < 7.5e-8.  Pure Python — no scipy.
    """
    # Handle negative values via symmetry
    negative = x < 0
    x = abs(x)

    # A&S 26.2.17 rational polynomial approximation
    t = 1.0 / (1.0 + 0.2316419 * x)
    poly = t * (
        0.319381530
        + t * (-0.356563782
        + t * (1.781477937
        + t * (-1.821255978
        + t * 1.330274429)))
    )
    cdf = 1.0 - _norm_pdf(x) * poly
    return 1.0 - cdf if negative else cdf


def black_scholes_greeks(
    option_type: str,    # "CE" or "PE"  (Call / Put)
    S: float,            # Current underlying price
    K: float,            # Strike price
    T: float,            # Time to expiry in years  (e.g. 30 days → 30/365)
    r: float,            # Risk-free rate (annual, decimal)
    sigma: float,        # Implied volatility (annual, decimal)
) -> dict:
    """Calculate full Black-Scholes option Greeks from scratch.

    Parameters
    ----------
    option_type : "CE" (call) or "PE" (put)
    S           : Underlying spot price
    K           : Strike price
    T           : Time to expiry in years  (must be > 0)
    r           : Risk-free annual rate (e.g. 0.0725 for 7.25 %)
    sigma       : Annualised implied volatility (e.g. 0.20 for 20 %)

    Returns
    -------
    {
        "option_type": str,
        "S": float, "K": float, "T": float, "r": float, "sigma": float,
        "price":  float,   # theoretical Black-Scholes price
        "delta":  float,   # rate of price change w.r.t. S
        "gamma":  float,   # rate of delta change w.r.t. S
        "theta":  float,   # time decay per day
        "vega":   float,   # sensitivity to 1 % change in vol
        "rho":    float,   # sensitivity to 1 % change in rate
        "d1": float, "d2": float,
    }
    """
    if T <= 0:
        # Expired option
        intrinsic = max(S - K, 0) if option_type.upper() == "CE" else max(K - S, 0)
        return {
            "option_type": option_type.upper(),
            "S": S, "K": K, "T": 0, "r": r, "sigma": sigma,
            "price": intrinsic,
            "delta": (1.0 if option_type.upper() == "CE" else -1.0) if S != K else 0.5,
            "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0,
            "d1": 0.0, "d2": 0.0,
        }

    sqrt_T = math.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T

    n_d1  = _norm_pdf(d1)
    disc  = math.exp(-r * T)

    if option_type.upper() == "CE":
        Nd1   = _norm_cdf(d1)
        Nd2   = _norm_cdf(d2)
        Nd1n  = _norm_cdf(-d1)

        price = S * Nd1 - K * disc * Nd2
        delta = Nd1
        rho   = K * T * disc * Nd2 / 100          # per 1 % change in r
    else:  # PE
        Nd1n  = _norm_cdf(-d1)
        Nd2n  = _norm_cdf(-d2)
        Nd1   = _norm_cdf(d1)

        price = K * disc * Nd2n - S * Nd1n
        delta = Nd1 - 1.0                          # negative for puts
        rho   = -K * T * disc * Nd2n / 100

    # Gamma is identical for calls and puts
    gamma = n_d1 / (S * sigma_sqrt_T)

    # Theta: annualised → per calendar day
    if option_type.upper() == "CE":
        theta = (
            -(S * n_d1 * sigma) / (2 * sqrt_T)
            - r * K * disc * _norm_cdf(d2)
        ) / 365
    else:
        theta = (
            -(S * n_d1 * sigma) / (2 * sqrt_T)
            + r * K * disc * _norm_cdf(-d2)
        ) / 365

    # Vega: per 1 % change in implied vol
    vega = S * n_d1 * sqrt_T / 100

    return {
        "option_type": option_type.upper(),
        "S":     round(S, 2),
        "K":     round(K, 2),
        "T":     round(T, 6),
        "r":     r,
        "sigma": round(sigma, 4),
        "price": round(price, 2),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega":  round(vega, 4),
        "rho":   round(rho, 4),
        "d1":    round(d1, 4),
        "d2":    round(d2, 4),
    }


def _estimate_iv(
    market_price: float,
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float = RISK_FREE_RATE,
) -> float:
    """Estimate implied volatility using bisection method.

    Returns IV as a decimal (e.g. 0.25 for 25 %).
    Falls back to 0.25 if market price is zero / invalid.
    """
    if market_price <= 0 or T <= 0:
        return 0.25

    lo, hi = 1e-6, 10.0  # 0 % – 1000 % vol range
    for _ in range(100):  # up to 100 iterations for convergence
        mid   = (lo + hi) / 2
        price = black_scholes_greeks(option_type, S, K, T, r, mid)["price"]
        if abs(price - market_price) < 0.01:
            return mid
        if price < market_price:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2


def get_options_chain(symbol: str, expiry: Optional[str] = None) -> dict:
    """Fetch live options chain data for *symbol* via yfinance.

    Parameters
    ----------
    symbol : NSE/BSE symbol or index alias
    expiry : YYYY-MM-DD string, or None for nearest expiry

    Returns
    -------
    {
        "symbol":       str,
        "expiry":       str,           # actual expiry date used
        "all_expiries": list[str],
        "underlying_price": float,
        "calls":        list[dict],    # each has strike, CE OI, volume, LTP, greeks
        "puts":         list[dict],    # same schema for puts
        "max_pain":     float,
        "pcr":          float,         # put-call ratio (by OI)
        "timestamp":    str,
    }
    """
    ticker_str = normalise_symbol(symbol)
    cache_key  = f"{ticker_str}_{expiry or 'nearest'}"
    cached     = _cget(cache_key)
    if cached:
        return cached

    tkr = yf.Ticker(ticker_str)

    # ── Get available expiry dates ───────────────────────────────────────────
    try:
        all_expiries: list[str] = list(tkr.options)
    except Exception:
        all_expiries = []

    if not all_expiries:
        return {
            "symbol":    symbol.upper(),
            "expiry":    None,
            "all_expiries": [],
            "error":     "No options data available for this symbol",
            "timestamp": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        }

    # Select expiry
    if expiry and expiry in all_expiries:
        chosen_expiry = expiry
    else:
        chosen_expiry = all_expiries[0]  # nearest expiry

    # ── Underlying price ─────────────────────────────────────────────────────
    info  = tkr.info
    S = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or 0.0
    )
    if S == 0.0:
        hist = tkr.history(period="1d", interval="1m")
        S = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0

    # ── Options chain ────────────────────────────────────────────────────────
    chain_obj = tkr.option_chain(chosen_expiry)
    calls_df  = chain_obj.calls.copy()
    puts_df   = chain_obj.puts.copy()

    # Time to expiry
    exp_dt = pd.Timestamp(chosen_expiry)
    now_dt = pd.Timestamp.now(tz="UTC").tz_localize(None)
    T = max((exp_dt - now_dt).days / 365, 0.0)

    def _enrich(row: pd.Series, opt_type: str) -> dict:
        K      = float(row.get("strike", 0))
        ltp    = float(row.get("lastPrice", 0))
        iv_raw = row.get("impliedVolatility", None)
        iv     = float(iv_raw) if iv_raw and not math.isnan(float(iv_raw)) else _estimate_iv(ltp, opt_type, S, K, T)

        greeks = black_scholes_greeks(opt_type, S, K, T, RISK_FREE_RATE, iv)

        return {
            "strike":          K,
            "option_type":     opt_type,
            "ltp":             ltp,
            "open_interest":   int(row.get("openInterest", 0) or 0),
            "volume":          int(row.get("volume", 0) or 0),
            "iv":              round(iv * 100, 2),  # store as %
            "bid":             float(row.get("bid", 0) or 0),
            "ask":             float(row.get("ask", 0) or 0),
            "delta":           greeks["delta"],
            "gamma":           greeks["gamma"],
            "theta":           greeks["theta"],
            "vega":            greeks["vega"],
            "bs_price":        greeks["price"],
        }

    calls: list[dict] = []
    puts:  list[dict] = []
    for _, row in calls_df.iterrows():
        try:
            calls.append(_enrich(row, "CE"))
        except Exception as exc:
            logger.debug("Skipping call row: %s", exc)
    for _, row in puts_df.iterrows():
        try:
            puts.append(_enrich(row, "PE"))
        except Exception as exc:
            logger.debug("Skipping put row: %s", exc)

    # ── Max Pain ─────────────────────────────────────────────────────────────
    max_pain = _calculate_max_pain(calls, puts)

    # ── PCR ──────────────────────────────────────────────────────────────────
    total_call_oi = sum(c["open_interest"] for c in calls)
    total_put_oi  = sum(p["open_interest"] for p in puts)
    pcr = round(total_put_oi / total_call_oi, 4) if total_call_oi > 0 else 0.0

    result = {
        "symbol":            symbol.upper(),
        "expiry":            chosen_expiry,
        "all_expiries":      all_expiries,
        "underlying_price":  round(float(S), 2),
        "calls":             calls,
        "puts":              puts,
        "max_pain":          max_pain,
        "pcr":               pcr,
        "total_call_oi":     total_call_oi,
        "total_put_oi":      total_put_oi,
        "timestamp":         pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
    }
    _cset(cache_key, result)
    return result


def _calculate_max_pain(calls: list[dict], puts: list[dict]) -> float:
    """Max-pain strike: the price at which combined options expire with
    minimum total payout to all option buyers.

    For each candidate strike K*:
      pain = Σ_calls  OI_c × max(K* − K_c, 0)   (in-the-money calls)
           + Σ_puts   OI_p × max(K_p − K*, 0)   (in-the-money puts)

    We iterate over all strikes present in the chain.
    """
    strikes = sorted(set(c["strike"] for c in calls) | set(p["strike"] for p in puts))
    if not strikes:
        return 0.0

    min_pain = float("inf")
    max_pain_strike = strikes[0]

    for candidate in strikes:
        # Call pain: calls with K < candidate are ITM
        call_pain = sum(
            c["open_interest"] * max(candidate - c["strike"], 0)
            for c in calls
        )
        # Put pain: puts with K > candidate are ITM
        put_pain = sum(
            p["open_interest"] * max(p["strike"] - candidate, 0)
            for p in puts
        )
        total_pain = call_pain + put_pain
        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = candidate

    return float(max_pain_strike)


def detect_unusual_activity(symbol: str) -> dict:
    """Detect unusual options volume or OI spikes.

    Logic
    -----
    For each options contract in the nearest 3 expiries:
      1. Flag contracts where volume > 3× average volume.
      2. Flag contracts where OI change proxy is large (volume / OI > 0.5).
      3. Flag contracts where volume > 1000 and OI > 5000 (absolute threshold).
    Optionally flag "whale" prints (single contracts with huge notional).

    Returns
    -------
    {
        "symbol":    str,
        "alerts":    list[dict],
        "anomalies": list[str],   # human-readable summaries
        "timestamp": str,
    }
    """
    ticker_str = normalise_symbol(symbol)
    tkr = yf.Ticker(ticker_str)
    try:
        expiries = list(tkr.options)[:3]  # analyse nearest 3 expiries
    except Exception:
        return {
            "symbol":    symbol.upper(),
            "alerts":    [],
            "anomalies": ["No options data available"],
            "timestamp": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        }

    info = tkr.info
    S = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0

    all_contracts: list[dict] = []
    for exp in expiries:
        try:
            chain = tkr.option_chain(exp)
            for _, row in pd.concat([chain.calls, chain.puts]).iterrows():
                all_contracts.append({
                    "expiry":       exp,
                    "strike":       float(row.get("strike", 0)),
                    "type":         "CE" if _ in chain.calls.index else "PE",
                    "volume":       int(row.get("volume", 0) or 0),
                    "open_interest": int(row.get("openInterest", 0) or 0),
                    "ltp":          float(row.get("lastPrice", 0) or 0),
                    "iv":           float(row.get("impliedVolatility", 0.25) or 0.25) * 100,
                })
        except Exception as exc:
            logger.debug("Expiry %s failed: %s", exp, exc)

    if not all_contracts:
        return {
            "symbol":    symbol.upper(),
            "alerts":    [],
            "anomalies": ["No contracts data retrieved"],
            "timestamp": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        }

    # ── Compute rolling z-score of volume across contracts ───────────────────
    volumes = [c["volume"] for c in all_contracts]
    avg_vol = sum(volumes) / len(volumes) if volumes else 1
    std_vol = (sum((v - avg_vol) ** 2 for v in volumes) / len(volumes)) ** 0.5 or 1

    alerts: list[dict] = []
    anomalies: list[str] = []

    for c in all_contracts:
        vol   = c["volume"]
        oi    = c["open_interest"]
        z     = (vol - avg_vol) / std_vol
        flags = []

        if vol > 3 * avg_vol and vol > 500:
            flags.append(f"volume spike (z={z:.1f})")
        if oi > 0 and vol / oi > 0.5 and vol > 500:
            flags.append(f"high vol/OI ratio ({vol/oi:.2f})")
        if vol > 2000 and oi > 10_000:
            flags.append("large absolute volume+OI")

        if flags:
            desc = (
                f"{symbol.upper()} {c['expiry']} {c['strike']} {c['type']}: "
                + ", ".join(flags)
            )
            alerts.append({**c, "flags": flags, "z_score": round(z, 2)})
            anomalies.append(desc)

    return {
        "symbol":    symbol.upper(),
        "alerts":    sorted(alerts, key=lambda x: x["volume"], reverse=True)[:20],
        "anomalies": anomalies[:20],
        "timestamp": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
    }


def calculate_greeks(
    symbol: str,
    strike: float,
    expiry: str,
    option_type: str = "CE",
    custom_iv: Optional[float] = None,
) -> dict:
    """High-level wrapper to calculate Greeks for a specific contract.

    Fetches live spot price, estimates T, derives IV from chain or custom,
    and runs Black-Scholes.

    Parameters
    ----------
    symbol      : underlying symbol (e.g. "NIFTY")
    strike      : strike price
    expiry      : "YYYY-MM-DD"
    option_type : "CE" or "PE"
    custom_iv   : override IV (decimal, e.g. 0.18 for 18 %)

    Returns
    -------
    Full greek dict from black_scholes_greeks() + metadata.
    """
    ticker_str = normalise_symbol(symbol)
    tkr = yf.Ticker(ticker_str)
    info = tkr.info
    S = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
    if S == 0.0:
        hist = tkr.history(period="1d", interval="1m")
        S = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0

    exp_dt = pd.Timestamp(expiry)
    T = max((exp_dt - pd.Timestamp.now(tz="UTC").tz_localize(None)).days / 365, 0.0)

    # Try to get market IV from chain
    iv = custom_iv
    if iv is None:
        try:
            chain = tkr.option_chain(expiry)
            df = chain.calls if option_type.upper() == "CE" else chain.puts
            row = df[df["strike"].round(0) == round(strike, 0)]
            if not row.empty:
                iv_raw = row.iloc[0].get("impliedVolatility")
                if iv_raw and not math.isnan(float(iv_raw)):
                    iv = float(iv_raw)
                else:
                    ltp = float(row.iloc[0].get("lastPrice", 0))
                    iv = _estimate_iv(ltp, option_type, S, strike, T)
        except Exception:
            pass
    if iv is None:
        iv = 0.25  # fallback

    greeks = black_scholes_greeks(option_type, S, strike, T, RISK_FREE_RATE, iv)
    greeks["symbol"]          = symbol.upper()
    greeks["expiry"]          = expiry
    greeks["days_to_expiry"]  = max(0, (exp_dt - pd.Timestamp.now(tz="UTC").tz_localize(None)).days)
    greeks["iv_pct"]          = round(iv * 100, 2)
    greeks["underlying_price"] = round(float(S), 2)
    greeks["timestamp"]       = pd.Timestamp.now(tz="Asia/Kolkata").isoformat()
    return greeks
