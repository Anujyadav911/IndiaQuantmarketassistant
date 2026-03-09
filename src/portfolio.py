"""Virtual portfolio backed by SQLite — live P&L, stop-loss alerts, risk scoring."""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Generator

import pandas as pd

from src.config import DATABASE_PATH, DEFAULT_CASH
from src.market_data import get_historical_data, get_live_price

logger = logging.getLogger(__name__)

_CREATE_PORTFOLIO_TABLE = """
CREATE TABLE IF NOT EXISTS portfolio (
    id          TEXT PRIMARY KEY,
    symbol      TEXT NOT NULL,
    quantity    INTEGER NOT NULL,
    avg_price   REAL NOT NULL,
    side        TEXT NOT NULL,          -- 'BUY' or 'SELL'
    stop_loss   REAL,
    target      REAL,
    entered_at  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'OPEN'   -- 'OPEN' | 'CLOSED'
);
"""

_CREATE_CASH_TABLE = """
CREATE TABLE IF NOT EXISTS cash (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    balance     REAL NOT NULL
);
"""

_CREATE_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id          TEXT PRIMARY KEY,
    symbol      TEXT NOT NULL,
    quantity    INTEGER NOT NULL,
    price       REAL NOT NULL,
    side        TEXT NOT NULL,          -- 'BUY' | 'SELL'
    order_type  TEXT NOT NULL DEFAULT 'MARKET',
    executed_at TEXT NOT NULL,
    pnl         REAL                    -- realised P&L on close
);
"""


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables and seed cash balance if first run."""
    with _db() as conn:
        conn.execute(_CREATE_PORTFOLIO_TABLE)
        conn.execute(_CREATE_CASH_TABLE)
        conn.execute(_CREATE_TRADES_TABLE)
        # Seed cash only if table is empty
        row = conn.execute("SELECT balance FROM cash WHERE id = 1").fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO cash (id, balance) VALUES (1, ?)", (DEFAULT_CASH,)
            )
    logger.info("Portfolio DB initialised at %s", DATABASE_PATH)


def _get_cash() -> float:
    with _db() as conn:
        row = conn.execute("SELECT balance FROM cash WHERE id = 1").fetchone()
        return float(row["balance"]) if row else DEFAULT_CASH


def _update_cash(new_balance: float) -> None:
    with _db() as conn:
        conn.execute(
            "INSERT INTO cash (id, balance) VALUES (1, ?) "
            "ON CONFLICT(id) DO UPDATE SET balance = excluded.balance",
            (new_balance,),
        )


def place_virtual_trade(
    symbol: str,
    quantity: int,
    side: str,
    stop_loss: float | None = None,
    target: float | None = None,
) -> dict:
    """Place a simulated market order.

    Rules
    -----
    BUY  → debit cash; open long position.
    SELL → credit cash; open short position (short-selling allowed virtually).
    If a matching opposite open position exists, it is closed and P&L realised.

    Returns
    -------
    {
        "order_id":  str,
        "symbol":    str,
        "side":      str,
        "quantity":  int,
        "price":     float,        # execution price (live)
        "status":    str,          # "FILLED"
        "cash_before": float,
        "cash_after":  float,
        "realised_pnl": float,     # if closing a position
        "message":   str,
        "timestamp": str,
    }
    """
    init_db()
    side = side.upper()
    if side not in ("BUY", "SELL"):
        return {"error": "side must be BUY or SELL"}
    if quantity <= 0:
        return {"error": "quantity must be a positive integer"}

    # Live price
    quote  = get_live_price(symbol)
    price  = quote["price"]
    if price == 0.0:
        return {"error": f"Could not get live price for {symbol}"}

    order_id   = str(uuid.uuid4())[:12].upper()
    cash       = _get_cash()
    cash_before = cash
    order_value = price * quantity
    realised_pnl = 0.0
    timestamp    = pd.Timestamp.now(tz="Asia/Kolkata").isoformat()

    with _db() as conn:
        # Check for opposite open position to close
        opposite = "SELL" if side == "BUY" else "BUY"
        existing = conn.execute(
            "SELECT * FROM portfolio WHERE symbol = ? AND side = ? AND status = 'OPEN'",
            (symbol.upper(), opposite),
        ).fetchone()

        if existing:
            # Close (partial or full)
            ex_qty   = existing["quantity"]
            ex_price = existing["avg_price"]
            close_qty = min(ex_qty, quantity)

            if opposite == "BUY":   # closing a long
                realised_pnl = (price - ex_price) * close_qty
                cash += order_value  # receive proceeds
            else:                   # closing a short
                realised_pnl = (ex_price - price) * close_qty
                cash -= order_value  # buy back

            if close_qty >= ex_qty:
                conn.execute(
                    "UPDATE portfolio SET status = 'CLOSED' WHERE id = ?",
                    (existing["id"],),
                )
            else:
                conn.execute(
                    "UPDATE portfolio SET quantity = quantity - ? WHERE id = ?",
                    (close_qty, existing["id"]),
                )
            remaining = quantity - close_qty
        else:
            remaining = quantity

        # Open a new position for any remaining quantity
        if remaining > 0:
            if side == "BUY":
                if cash < price * remaining:
                    return {
                        "error":   "Insufficient cash",
                        "cash":    cash,
                        "required": price * remaining,
                    }
                cash -= price * remaining
            else:
                cash += price * remaining  # short sale proceeds

            conn.execute(
                """INSERT INTO portfolio
                       (id, symbol, quantity, avg_price, side, stop_loss, target, entered_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')""",
                (order_id, symbol.upper(), remaining, price, side, stop_loss, target, timestamp),
            )

        # Record in trade history
        conn.execute(
            """INSERT INTO trades (id, symbol, quantity, price, side, executed_at, pnl)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (order_id, symbol.upper(), quantity, price, side, timestamp, realised_pnl),
        )

    _update_cash(cash)

    return {
        "order_id":     order_id,
        "symbol":       symbol.upper(),
        "side":         side,
        "quantity":     quantity,
        "price":        round(price, 2),
        "order_value":  round(order_value, 2),
        "status":       "FILLED",
        "cash_before":  round(cash_before, 2),
        "cash_after":   round(cash, 2),
        "realised_pnl": round(realised_pnl, 2),
        "message":      (
            f"{side} {quantity} {symbol.upper()} @ ₹{price:.2f} | "
            f"P&L: ₹{realised_pnl:+.2f}"
        ),
        "timestamp": timestamp,
    }


def _volatility_score(symbol: str) -> float:
    """Return annualised volatility (%) as a risk score.

    Uses 30-day daily returns.  Falls back to 20 % if data unavailable.
    """
    try:
        df = get_historical_data(symbol, period="3mo", interval="1d")
        if df.empty or len(df) < 5:
            return 20.0
        daily_returns = df["Close"].pct_change().dropna()
        ann_vol = float(daily_returns.std() * (252 ** 0.5) * 100)
        return round(ann_vol, 2)
    except Exception:
        return 20.0


def get_portfolio_pnl() -> dict:
    """Compute real-time P&L for all open positions.

    Returns
    -------
    {
        "cash":         float,
        "positions":    list[PositionPNL],
        "total_invested": float,
        "total_current_value": float,
        "total_unrealised_pnl": float,
        "total_unrealised_pnl_pct": float,
        "portfolio_value": float,    # cash + current equity value
        "risk_summary": dict,
        "timestamp":    str,
    }
    """
    init_db()
    cash = _get_cash()

    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM portfolio WHERE status = 'OPEN'"
        ).fetchall()

    positions: list[dict] = []
    total_invested     = 0.0
    total_current_val  = 0.0

    for row in rows:
        sym       = row["symbol"]
        qty       = row["quantity"]
        avg_price = row["avg_price"]
        side      = row["side"]
        sl        = row["stop_loss"]
        tgt       = row["target"]

        try:
            quote  = get_live_price(sym)
            ltp    = quote["price"]
        except Exception:
            ltp = avg_price  # fallback if market is closed

        if side == "BUY":
            unrealised = (ltp - avg_price) * qty
            invested   = avg_price * qty
            cur_val    = ltp * qty
        else:  # short
            unrealised = (avg_price - ltp) * qty
            invested   = avg_price * qty
            cur_val    = avg_price * qty + unrealised

        pnl_pct   = round((unrealised / invested * 100) if invested else 0.0, 2)
        vol_score = _volatility_score(sym)

        # Stop-loss / target hit check
        sl_hit  = sl  is not None and (ltp <= sl if side == "BUY" else ltp >= sl)
        tgt_hit = tgt is not None and (ltp >= tgt if side == "BUY" else ltp <= tgt)

        positions.append({
            "id":               row["id"],
            "symbol":           sym,
            "side":             side,
            "quantity":         qty,
            "avg_price":        round(avg_price, 2),
            "ltp":              round(ltp, 2),
            "invested_value":   round(invested, 2),
            "current_value":    round(cur_val, 2),
            "unrealised_pnl":   round(unrealised, 2),
            "unrealised_pnl_pct": pnl_pct,
            "stop_loss":        sl,
            "target":           tgt,
            "stop_loss_hit":    sl_hit,
            "target_hit":       tgt_hit,
            "volatility_score": vol_score,
            "risk_level":       "HIGH" if vol_score > 35 else ("MEDIUM" if vol_score > 20 else "LOW"),
            "change_pct":       quote.get("change_pct", 0.0) if "quote" in dir() else 0.0,
            "entered_at":       row["entered_at"],
        })

        total_invested    += invested
        total_current_val += cur_val

    unrealised_total     = total_current_val - total_invested
    unrealised_total_pct = round(
        (unrealised_total / total_invested * 100) if total_invested else 0.0, 2
    )
    portfolio_value = cash + total_current_val

    # Risk summary
    high_risk = [p for p in positions if p["risk_level"] == "HIGH"]
    med_risk  = [p for p in positions if p["risk_level"] == "MEDIUM"]
    low_risk  = [p for p in positions if p["risk_level"] == "LOW"]

    return {
        "cash":                     round(cash, 2),
        "positions":                positions,
        "total_invested":           round(total_invested, 2),
        "total_current_value":      round(total_current_val, 2),
        "total_unrealised_pnl":     round(unrealised_total, 2),
        "total_unrealised_pnl_pct": unrealised_total_pct,
        "portfolio_value":          round(portfolio_value, 2),
        "risk_summary": {
            "high_risk_positions":   len(high_risk),
            "medium_risk_positions": len(med_risk),
            "low_risk_positions":    len(low_risk),
            "sl_alerts":  [p["symbol"] for p in positions if p["stop_loss_hit"]],
            "tgt_alerts": [p["symbol"] for p in positions if p["target_hit"]],
        },
        "timestamp": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
    }


def get_open_positions() -> list[dict]:
    """Return raw list of open positions from DB."""
    init_db()
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM portfolio WHERE status = 'OPEN' ORDER BY entered_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_trade_history(limit: int = 50) -> list[dict]:
    """Return most-recent trades from the trades table."""
    init_db()
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY executed_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
