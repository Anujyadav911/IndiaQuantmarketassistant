"""Per-sector average % change heatmap using live prices."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from src.config import SECTOR_CACHE_TTL, SECTOR_SYMBOLS
from src.market_data import get_live_price

logger = logging.getLogger(__name__)


@dataclass
class _CE:
    data: dict
    ts: float = field(default_factory=time.time)


_heatmap_cache: dict[str, _CE] = {}


def get_sector_heatmap() -> dict:
    """Return % change, top movers and heat for each sector.

    Returns
    -------
    {
        "sectors": [
            {
                "name":       str,
                "change_pct": float,         # avg % change of constituent stocks
                "heat":       "BULLISH" | "BEARISH" | "NEUTRAL",
                "top_gainer": {"symbol": str, "change_pct": float},
                "top_loser":  {"symbol": str, "change_pct": float},
                "stocks":     list[{"symbol": str, "price": float, "change_pct": float}],
            },
            ...
        ],
        "timestamp": str,
    }
    """
    cached_e = _heatmap_cache.get("heatmap")
    if cached_e and (time.time() - cached_e.ts) < SECTOR_CACHE_TTL:
        return cached_e.data

    sectors_out = []
    for sector_name, symbols in SECTOR_SYMBOLS.items():
        stocks = []
        for sym in symbols:
            try:
                q = get_live_price(sym)
                stocks.append({
                    "symbol":     sym,
                    "price":      q["price"],
                    "change_pct": q["change_pct"],
                })
            except Exception as exc:
                logger.debug("Skipping %s in %s: %s", sym, sector_name, exc)

        if not stocks:
            continue

        avg_chg = round(sum(s["change_pct"] for s in stocks) / len(stocks), 2)
        top_gainer = max(stocks, key=lambda x: x["change_pct"])
        top_loser  = min(stocks, key=lambda x: x["change_pct"])
        heat = "BULLISH" if avg_chg > 0.3 else ("BEARISH" if avg_chg < -0.3 else "NEUTRAL")

        sectors_out.append({
            "name":       sector_name,
            "change_pct": avg_chg,
            "heat":       heat,
            "top_gainer": {"symbol": top_gainer["symbol"], "change_pct": top_gainer["change_pct"]},
            "top_loser":  {"symbol": top_loser["symbol"],  "change_pct": top_loser["change_pct"]},
            "stocks":     stocks,
        })

    # Sort sectors by change_pct descending
    sectors_out.sort(key=lambda x: x["change_pct"], reverse=True)

    result = {
        "sectors":   sectors_out,
        "timestamp": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
    }
    _heatmap_cache["heatmap"] = _CE(data=result)
    return result
