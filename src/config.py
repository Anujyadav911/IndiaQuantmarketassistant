"""Central config — all constants, API keys and symbol lists live here."""

import os
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
ALPHA_VANTAGE_KEY: str = os.getenv("ALPHA_VANTAGE_KEY", "")

DATABASE_PATH: str = os.getenv("DATABASE_PATH", "portfolio.db")

DEFAULT_CASH: float = float(os.getenv("DEFAULT_CASH", "1000000"))  # ₹10 Lakh

PRICE_CACHE_TTL: int = 30        # Live price — refresh every 30 s
HISTORICAL_CACHE_TTL: int = 300  # OHLCV history — refresh every 5 min
OPTIONS_CACHE_TTL: int = 60      # Options chain — refresh every 1 min
NEWS_CACHE_TTL: int = 600        # News headlines — refresh every 10 min
SIGNAL_CACHE_TTL: int = 120      # Trade signals — refresh every 2 min
SECTOR_CACHE_TTL: int = 120      # Sector heatmap — refresh every 2 min

# Approximate Indian 10-year G-Sec yield (risk-free rate for Black-Scholes)
RISK_FREE_RATE: float = 0.0725   # 7.25 % p.a.

INDEX_MAP: dict[str, str] = {
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "BANK NIFTY": "^NSEBANK",
    "NIFTYBANK": "^NSEBANK",
    "SENSEX": "^BSESN",
    "FINNIFTY": "^CNXFIN",
    "MIDCAP": "^NSEMDCP50",
    "NIFTYMIDCAP": "^NSEMDCP50",
}

NIFTY50_SYMBOLS: list[str] = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BHARTIARTL", "BPCL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK",
    "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN",
    "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]

SECTOR_SYMBOLS: dict[str, list[str]] = {
    "IT":             ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"],
    "Banking":        ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK"],
    "Auto":           ["MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "M&M"],
    "Pharma":         ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP"],
    "Energy":         ["RELIANCE", "ONGC", "BPCL", "NTPC", "POWERGRID"],
    "FMCG":           ["HINDUNILVR", "ITC", "BRITANNIA", "NESTLEIND", "TATACONSUM"],
    "Metals":         ["TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA"],
    "Infrastructure": ["LT", "ADANIPORTS", "ADANIENT", "GRASIM", "ULTRACEMCO"],
}

COMPANY_NAMES: dict[str, str] = {
    "RELIANCE":   "Reliance Industries",
    "HDFCBANK":   "HDFC Bank",
    "ICICIBANK":  "ICICI Bank",
    "INFY":       "Infosys",
    "TCS":        "TCS Tata Consultancy",
    "SBIN":       "State Bank India SBI",
    "WIPRO":      "Wipro",
    "AXISBANK":   "Axis Bank",
    "KOTAKBANK":  "Kotak Mahindra Bank",
    "BAJFINANCE": "Bajaj Finance",
    "TATAMOTORS": "Tata Motors",
    "MARUTI":     "Maruti Suzuki",
    "HCLTECH":    "HCL Technologies",
    "ADANIENT":   "Adani Enterprises",
    "SUNPHARMA":  "Sun Pharmaceutical",
    "DRREDDY":    "Dr Reddy Laboratories",
    "CIPLA":      "Cipla",
    "BHARTIARTL": "Bharti Airtel",
    "LT":         "Larsen Toubro",
    "NTPC":       "NTPC Power",
}
