"""
tests/test_portfolio.py – Unit tests for the Portfolio Risk Manager.

Uses an in-memory SQLite database (via tmp_path fixture) to avoid
polluting the real portfolio.
"""
import os
import pytest
from unittest.mock import patch

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Redirect SQLite to a temp file for each test."""
    db_file = str(tmp_path / "test_portfolio.db")
    monkeypatch.setenv("DATABASE_PATH", db_file)

    # Patch the module-level constant after env is set
    import src.portfolio as pm
    pm.DATABASE_PATH = db_file
    # Also patch config
    import src.config as cfg
    cfg.DATABASE_PATH = db_file

    pm.init_db()
    yield
    # Cleanup is automatic via tmp_path


def _mock_price(symbol, price=1000.0):
    return {
        "symbol": symbol, "ticker": symbol + ".NS", "price": price,
        "open": price, "high": price + 10, "low": price - 10,
        "prev_close": price - 5, "change": 5.0, "change_pct": 0.5,
        "volume": 500000, "market_cap": None, "52w_high": None,
        "52w_low": None, "currency": "INR", "exchange": "NSE",
        "timestamp": "2026-03-09T10:00:00+05:30",
    }


# ── Trade Placement ───────────────────────────────────────────────────────────

class TestPlaceVirtualTrade:
    def test_buy_reduces_cash(self):
        from src.portfolio import place_virtual_trade, _get_cash
        initial_cash = _get_cash()
        with patch("src.portfolio.get_live_price", return_value=_mock_price("INFY", 1500.0)):
            result = place_virtual_trade("INFY", 10, "BUY")
        assert result["status"] == "FILLED"
        assert result["cash_after"] == pytest.approx(initial_cash - 15000.0, abs=1.0)

    def test_sell_increases_cash(self):
        from src.portfolio import place_virtual_trade, _get_cash
        initial_cash = _get_cash()
        with patch("src.portfolio.get_live_price", return_value=_mock_price("INFY", 1500.0)):
            result = place_virtual_trade("INFY", 10, "SELL")
        assert result["status"] == "FILLED"
        assert result["cash_after"] == pytest.approx(initial_cash + 15000.0, abs=1.0)

    def test_order_id_generated(self):
        from src.portfolio import place_virtual_trade
        with patch("src.portfolio.get_live_price", return_value=_mock_price("TCS", 3000.0)):
            result = place_virtual_trade("TCS", 5, "BUY")
        assert "order_id" in result
        assert len(result["order_id"]) > 0

    def test_zero_quantity_rejected(self):
        from src.portfolio import place_virtual_trade
        result = place_virtual_trade("RELIANCE", 0, "BUY")
        assert "error" in result

    def test_close_long_realises_pnl(self):
        """Buy 10 @ 1000, sell 10 @ 1100 → P&L = +1000."""
        from src.portfolio import place_virtual_trade
        with patch("src.portfolio.get_live_price", return_value=_mock_price("SBIN", 1000.0)):
            place_virtual_trade("SBIN", 10, "BUY")
        with patch("src.portfolio.get_live_price", return_value=_mock_price("SBIN", 1100.0)):
            result = place_virtual_trade("SBIN", 10, "SELL")
        assert result["realised_pnl"] == pytest.approx(1000.0, abs=1.0)

    def test_insufficient_cash_rejected(self):
        from src.portfolio import place_virtual_trade, _update_cash
        _update_cash(100.0)  # very little cash
        with patch("src.portfolio.get_live_price", return_value=_mock_price("RELIANCE", 2900.0)):
            result = place_virtual_trade("RELIANCE", 1000, "BUY")
        assert "error" in result


# ── Portfolio P&L ─────────────────────────────────────────────────────────────

class TestPortfolioPNL:
    def test_empty_portfolio(self):
        from src.portfolio import get_portfolio_pnl
        pnl = get_portfolio_pnl()
        assert pnl["positions"] == []
        assert pnl["total_unrealised_pnl"] == 0.0

    def test_positive_pnl(self):
        from src.portfolio import place_virtual_trade, get_portfolio_pnl
        with patch("src.portfolio.get_live_price", return_value=_mock_price("HDFCBANK", 1600.0)):
            place_virtual_trade("HDFCBANK", 10, "BUY")

        with patch("src.portfolio.get_live_price", return_value=_mock_price("HDFCBANK", 1700.0)):
            with patch("src.portfolio.get_historical_data") as mock_hist:
                import pandas as pd, numpy as np
                prices = pd.Series([1600.0 + i for i in range(60)], dtype=float)
                df = pd.DataFrame({"Close": prices})
                mock_hist.return_value = df
                pnl = get_portfolio_pnl()

        assert pnl["total_unrealised_pnl"] > 0

    def test_portfolio_value_includes_cash(self):
        from src.portfolio import get_portfolio_pnl, _get_cash
        pnl = get_portfolio_pnl()
        assert abs(pnl["portfolio_value"] - _get_cash()) < 0.01  # no positions → value = cash

    def test_stop_loss_alert(self):
        from src.portfolio import place_virtual_trade, get_portfolio_pnl
        with patch("src.portfolio.get_live_price", return_value=_mock_price("WIPRO", 400.0)):
            place_virtual_trade("WIPRO", 5, "BUY", stop_loss=450.0)

        with patch("src.portfolio.get_live_price", return_value=_mock_price("WIPRO", 380.0)):
            with patch("src.portfolio.get_historical_data") as mock_h:
                import pandas as pd
                mock_h.return_value = pd.DataFrame({"Close": [400.0] * 60})
                pnl = get_portfolio_pnl()

        sl_alerts = pnl["risk_summary"]["sl_alerts"]
        assert "WIPRO" in sl_alerts


# ── Trade History ─────────────────────────────────────────────────────────────

class TestTradeHistory:
    def test_history_records_trades(self):
        from src.portfolio import place_virtual_trade, get_trade_history
        with patch("src.portfolio.get_live_price", return_value=_mock_price("ITC", 430.0)):
            place_virtual_trade("ITC", 20, "BUY")
        history = get_trade_history()
        assert len(history) >= 1
        assert history[0]["symbol"] == "ITC"
