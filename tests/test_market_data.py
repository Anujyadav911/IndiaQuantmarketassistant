"""
tests/test_market_data.py – Unit tests for Market Data Engine.

Network calls are mocked, so these tests run offline.
"""
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.market_data import normalise_symbol, clear_caches


# ─── Symbol Normalisation ─────────────────────────────────────────────────────

class TestNormaliseSymbol:
    def test_plain_symbol(self):
        assert normalise_symbol("RELIANCE") == "RELIANCE.NS"

    def test_already_ns(self):
        assert normalise_symbol("RELIANCE.NS") == "RELIANCE.NS"

    def test_already_bo(self):
        assert normalise_symbol("RELIANCE.BO") == "RELIANCE.BO"

    def test_nifty_alias(self):
        assert normalise_symbol("NIFTY") == "^NSEI"

    def test_banknifty_alias(self):
        assert normalise_symbol("BANKNIFTY") == "^NSEBANK"

    def test_sensex_alias(self):
        assert normalise_symbol("SENSEX") == "^BSESN"

    def test_nifty50_alias(self):
        assert normalise_symbol("NIFTY50") == "^NSEI"

    def test_index_caret(self):
        assert normalise_symbol("^NSEI") == "^NSEI"

    def test_lowercase_input(self):
        assert normalise_symbol("infy") == "INFY.NS"


# ─── Live Price ───────────────────────────────────────────────────────────────

class TestGetLivePrice:
    @pytest.fixture(autouse=True)
    def _clear(self):
        clear_caches()
        yield
        clear_caches()

    def _make_mock_ticker(self, info_dict, hist_df=None):
        tkr = MagicMock()
        tkr.info = info_dict
        if hist_df is not None:
            tkr.history.return_value = hist_df
        else:
            tkr.history.return_value = pd.DataFrame()
        return tkr

    def test_returns_expected_fields(self):
        info = {
            "currentPrice": 2500.0,
            "previousClose": 2480.0,
            "open": 2490.0,
            "dayHigh": 2510.0,
            "dayLow": 2470.0,
            "volume": 1000000,
            "currency": "INR",
            "exchange": "NSE",
        }
        with patch("src.market_data.yf.Ticker", return_value=self._make_mock_ticker(info)):
            from src.market_data import get_live_price
            result = get_live_price("RELIANCE")

        assert result["price"] == 2500.0
        assert result["symbol"] == "RELIANCE"
        assert "change_pct" in result
        assert result["change_pct"] == pytest.approx((2500 - 2480) / 2480 * 100, rel=0.01)

    def test_change_pct_calculation(self):
        info = {"currentPrice": 1100.0, "previousClose": 1000.0}
        with patch("src.market_data.yf.Ticker", return_value=self._make_mock_ticker(info)):
            from src.market_data import get_live_price
            result = get_live_price("TEST")
        assert result["change_pct"] == pytest.approx(10.0, rel=0.01)

    def test_cache_hit(self):
        info = {"currentPrice": 500.0, "previousClose": 490.0}
        mock_ticker = self._make_mock_ticker(info)
        with patch("src.market_data.yf.Ticker", return_value=mock_ticker) as mock_yf:
            from src.market_data import get_live_price
            get_live_price("WIPRO")
            get_live_price("WIPRO")
            # Ticker should only be instantiated once due to cache
            assert mock_yf.call_count == 1


# ─── Historical Data ──────────────────────────────────────────────────────────

class TestGetHistoricalData:
    @pytest.fixture(autouse=True)
    def _clear(self):
        clear_caches()
        yield
        clear_caches()

    def test_returns_dataframe(self):
        df = pd.DataFrame({
            "Open": [100, 101], "High": [105, 106],
            "Low": [98, 99], "Close": [102, 103], "Volume": [1000, 2000]
        })
        mock_tkr = MagicMock()
        mock_tkr.history.return_value = df
        with patch("src.market_data.yf.Ticker", return_value=mock_tkr):
            from src.market_data import get_historical_data
            result = get_historical_data("INFY", period="1mo", interval="1d")
        assert not result.empty
        assert "Close" in result.columns

    def test_empty_dataframe_on_no_data(self):
        mock_tkr = MagicMock()
        mock_tkr.history.return_value = pd.DataFrame()
        with patch("src.market_data.yf.Ticker", return_value=mock_tkr):
            from src.market_data import get_historical_data
            result = get_historical_data("INVALID_SYM")
        assert result.empty
