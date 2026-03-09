"""
tests/test_options_analysis.py – Unit tests for options analysis helpers.

Tests max-pain calculation and unusual-activity detection logic
without making real network calls.
"""
import pytest
from src.options import _calculate_max_pain, black_scholes_greeks


class TestMaxPain:
    def test_simple_example(self):
        """
        Synthetic scenario:
          Calls: strike=100 OI=1000, strike=105 OI=500
          Puts:  strike=100 OI=800, strike=95 OI=1500

        At candidate=100:
          call_pain = 1000*(100-100) + 500*(100-105 clip 0) = 0 + 0 = 0
          put_pain  = 800*(100-100) + 1500*(95-100 clip 0) = 0 + 0 = 0
          total = 0  ← minimum → max pain = 100

        At candidate=95:
          call_pain = 1000*(95-100 clip 0) + 500*(95-105 clip 0) = 0
          put_pain  = 800*(100-95) + 1500*(95-95) = 4000
          total = 4000
        """
        calls = [
            {"strike": 100, "open_interest": 1000},
            {"strike": 105, "open_interest": 500},
        ]
        puts = [
            {"strike": 100, "open_interest": 800},
            {"strike": 95,  "open_interest": 1500},
        ]
        mp = _calculate_max_pain(calls, puts)
        assert mp == 100.0

    def test_empty_chain_returns_zero(self):
        assert _calculate_max_pain([], []) == 0.0

    def test_single_strike(self):
        calls = [{"strike": 200, "open_interest": 100}]
        puts  = [{"strike": 200, "open_interest": 200}]
        mp    = _calculate_max_pain(calls, puts)
        assert mp == 200.0

    def test_max_pain_is_a_valid_strike(self):
        """Max pain must be one of the strikes in the chain."""
        strikes = [18000, 18100, 18200, 18300, 18400]
        calls = [{"strike": s, "open_interest": 5000 - abs(s - 18200) * 10} for s in strikes]
        puts  = [{"strike": s, "open_interest": 5000 - abs(s - 18200) * 10} for s in strikes]
        mp = _calculate_max_pain(calls, puts)
        assert mp in strikes


class TestGreeksIntegration:
    """Spot-check Greeks at real-world-like values (Nifty ATM call)."""

    def test_nifty_atm_call(self):
        # Nifty ≈ 22000, ATM call, 7 days to expiry, IV ≈ 15%
        g = black_scholes_greeks("CE", S=22000, K=22000, T=7/365, r=0.0725, sigma=0.15)
        assert 0.4 < g["delta"] < 0.65      # ATM ≈ 0.5
        assert g["gamma"] > 0
        assert g["theta"] < 0               # time decay
        assert g["vega"] > 0
        assert g["price"] > 0

    def test_nifty_deep_otm_put(self):
        # 3% OTM put, 7 days
        g = black_scholes_greeks("PE", S=22000, K=21340, T=7/365, r=0.0725, sigma=0.15)
        assert -0.15 < g["delta"] < 0.0     # low delta OTM put
        assert g["price"] < 100             # cheap OTM put

    def test_greeks_values_are_finite(self):
        g = black_scholes_greeks("CE", S=500, K=550, T=30/365, r=0.07, sigma=0.25)
        for field in ("price", "delta", "gamma", "theta", "vega", "rho"):
            import math
            assert math.isfinite(g[field]), f"{field} is not finite"
