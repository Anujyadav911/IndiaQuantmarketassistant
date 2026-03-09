"""
tests/test_black_scholes.py – Unit tests for the Black-Scholes implementation.

All tests are pure math — no external network calls.
Run with:  pytest tests/test_black_scholes.py -v
"""
import math
import pytest
from src.options import black_scholes_greeks, _norm_cdf, _norm_pdf, _estimate_iv


# ─── Normal Distribution ──────────────────────────────────────────────────────

class TestNormCDF:
    def test_cdf_at_zero(self):
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-6

    def test_cdf_symmetry(self):
        """CDF(x) + CDF(-x) should equal 1 for all x."""
        for x in [0.1, 0.5, 1.0, 1.96, 2.5, 3.0]:
            assert abs(_norm_cdf(x) + _norm_cdf(-x) - 1.0) < 1e-6

    def test_cdf_known_values(self):
        """Spot-check against well-known standard normal table values."""
        cases = [
            (1.0,  0.8413),
            (1.96, 0.9750),
            (2.0,  0.9772),
            (-1.0, 0.1587),
        ]
        for x, expected in cases:
            assert abs(_norm_cdf(x) - expected) < 1e-3, f"CDF({x}) ≠ {expected}"

    def test_pdf_at_zero(self):
        expected = 1.0 / math.sqrt(2 * math.pi)
        assert abs(_norm_pdf(0.0) - expected) < 1e-9


# ─── Black-Scholes Call ───────────────────────────────────────────────────────

class TestCallGreeks:
    """ATM call: S=100, K=100, T=1yr, r=5%, σ=20%"""

    @pytest.fixture(autouse=True)
    def _compute(self):
        self.g = black_scholes_greeks("CE", S=100, K=100, T=1.0, r=0.05, sigma=0.20)

    def test_price_is_positive(self):
        assert self.g["price"] > 0

    def test_atm_call_price_approximately(self):
        # Standard BS price for ATM (S=K=100, T=1, r=5%, σ=20%) ≈ 10.45
        assert 9.0 < self.g["price"] < 12.0

    def test_call_delta_range(self):
        """Delta for a call must be in (0, 1)."""
        assert 0 < self.g["delta"] < 1

    def test_atm_call_delta_near_half(self):
        """ATM call delta ≈ 0.5–0.6 (due to r and σ)."""
        assert 0.45 < self.g["delta"] < 0.65

    def test_gamma_positive(self):
        assert self.g["gamma"] > 0

    def test_theta_negative(self):
        """Theta (time decay) must be negative for long options."""
        assert self.g["theta"] < 0

    def test_vega_positive(self):
        assert self.g["vega"] > 0

    def test_put_call_parity(self):
        """C - P = S - K * e^(-rT)  (put-call parity)."""
        call = black_scholes_greeks("CE", S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        put  = black_scholes_greeks("PE", S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        parity_lhs = call["price"] - put["price"]
        parity_rhs = 100 - 100 * math.exp(-0.05 * 1.0)
        assert abs(parity_lhs - parity_rhs) < 0.01


# ─── Black-Scholes Put ────────────────────────────────────────────────────────

class TestPutGreeks:
    @pytest.fixture(autouse=True)
    def _compute(self):
        self.g = black_scholes_greeks("PE", S=100, K=100, T=1.0, r=0.05, sigma=0.20)

    def test_price_positive(self):
        assert self.g["price"] > 0

    def test_put_delta_range(self):
        """Delta for a put must be in (-1, 0)."""
        assert -1 < self.g["delta"] < 0

    def test_put_theta_negative(self):
        assert self.g["theta"] < 0

    def test_put_vega_positive(self):
        assert self.g["vega"] > 0

    def test_put_gamma_positive(self):
        assert self.g["gamma"] > 0


# ─── Deep ITM / OTM ──────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_deep_itm_call_delta_near_one(self):
        g = black_scholes_greeks("CE", S=200, K=100, T=0.1, r=0.05, sigma=0.2)
        assert g["delta"] > 0.95

    def test_deep_otm_call_delta_near_zero(self):
        g = black_scholes_greeks("CE", S=100, K=200, T=0.1, r=0.05, sigma=0.2)
        assert g["delta"] < 0.05

    def test_expired_option_call(self):
        g = black_scholes_greeks("CE", S=110, K=100, T=0.0, r=0.05, sigma=0.2)
        assert g["price"] == 10  # intrinsic value
        assert g["gamma"] == 0.0

    def test_expired_option_put_otm(self):
        g = black_scholes_greeks("PE", S=110, K=100, T=0.0, r=0.05, sigma=0.2)
        assert g["price"] == 0  # OTM put → worthless

    def test_high_volatility_call(self):
        g = black_scholes_greeks("CE", S=100, K=100, T=1.0, r=0.05, sigma=2.0)
        assert g["price"] > 0
        assert g["delta"] < 1.0

    def test_different_strikes_delta_ordering(self):
        """Higher strike → lower call delta (monotonicity)."""
        d1 = black_scholes_greeks("CE", S=100, K=90,  T=1.0, r=0.05, sigma=0.2)["delta"]
        d2 = black_scholes_greeks("CE", S=100, K=100, T=1.0, r=0.05, sigma=0.2)["delta"]
        d3 = black_scholes_greeks("CE", S=100, K=110, T=1.0, r=0.05, sigma=0.2)["delta"]
        assert d1 > d2 > d3


# ─── Implied Volatility ───────────────────────────────────────────────────────

class TestIV:
    def test_round_trip(self):
        """BS price → IV estimation should recover the original sigma."""
        sigma_true = 0.25
        price = black_scholes_greeks("CE", S=100, K=100, T=0.5, r=0.05, sigma=sigma_true)["price"]
        sigma_est = _estimate_iv(price, "CE", S=100, K=100, T=0.5, r=0.05)
        assert abs(sigma_est - sigma_true) < 0.005  # within 0.5 %

    def test_zero_market_price_returns_default(self):
        iv = _estimate_iv(0.0, "CE", S=100, K=100, T=0.5, r=0.05)
        assert iv == 0.25

    def test_expired_returns_default(self):
        iv = _estimate_iv(10.0, "CE", S=100, K=100, T=0.0, r=0.05)
        assert iv == 0.25
