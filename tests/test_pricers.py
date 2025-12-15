"""
Unit tests for pricers module.
"""

from datetime import date, timedelta
import numpy as np
import pytest

from rateslib.curves import Curve, create_flat_curve
from rateslib.pricers import BondPricer, SwapPricer, FuturesPricer, FuturesContract


@pytest.fixture
def sample_curve():
    """Create sample discount curve for pricing."""
    anchor_date = date(2024, 1, 15)
    # Create a flat 5% curve
    return create_flat_curve(anchor_date, rate=0.05, max_tenor_years=30.0)


class TestBondPricer:
    """Tests for bond pricing."""
    
    def test_par_bond_price(self, sample_curve):
        """Test that par bond prices near 100."""
        pricer = BondPricer(sample_curve)
        
        # Bond with coupon = yield should price near par
        dirty, clean, accrued = pricer.price(
            settlement=date(2024, 1, 15),
            maturity=date(2029, 1, 15),
            coupon_rate=0.05,  # 5% coupon = 5% discount rate
            frequency=2,
            face_value=100.0
        )
        
        # Should be close to 100
        assert 98 < clean < 102
    
    def test_premium_bond(self, sample_curve):
        """Test premium bond prices above par."""
        pricer = BondPricer(sample_curve)
        
        # High coupon bond should price above par
        dirty, clean, accrued = pricer.price(
            settlement=date(2024, 1, 15),
            maturity=date(2029, 1, 15),
            coupon_rate=0.07,  # 7% coupon > 5% yield
            frequency=2,
            face_value=100.0
        )
        
        assert clean > 100
    
    def test_discount_bond(self, sample_curve):
        """Test discount bond prices below par."""
        pricer = BondPricer(sample_curve)
        
        # Low coupon bond should price below par
        dirty, clean, accrued = pricer.price(
            settlement=date(2024, 1, 15),
            maturity=date(2029, 1, 15),
            coupon_rate=0.03,  # 3% coupon < 5% yield
            frequency=2,
            face_value=100.0
        )
        
        assert clean < 100
    
    def test_zero_coupon_bond(self, sample_curve):
        """Test zero coupon bond."""
        pricer = BondPricer(sample_curve)
        
        dirty, clean, accrued = pricer.price(
            settlement=date(2024, 1, 15),
            maturity=date(2025, 1, 15),  # 1 year
            coupon_rate=0.0,
            frequency=2,  # Still need frequency for schedule
            face_value=100.0
        )
        
        # Should be around exp(-0.05 * 1) * 100 ≈ 95.12
        assert 94 < clean < 96
    
    def test_dv01_positive_for_long(self, sample_curve):
        """Test DV01 is positive for long bond position."""
        pricer = BondPricer(sample_curve)
        
        dv01 = pricer.compute_dv01(
            settlement=date(2024, 1, 15),
            maturity=date(2034, 1, 15),  # 10 year
            coupon_rate=0.04,
            frequency=2,
            notional=1_000_000
        )
        
        # DV01 should be positive (price goes up when rates go down)
        assert dv01 > 0
    
    def test_longer_bond_higher_dv01(self, sample_curve):
        """Test longer bonds have higher DV01."""
        pricer = BondPricer(sample_curve)
        
        dv01_5y = pricer.compute_dv01(
            settlement=date(2024, 1, 15),
            maturity=date(2029, 1, 15),
            coupon_rate=0.04,
            frequency=2,
            notional=1_000_000
        )
        
        dv01_10y = pricer.compute_dv01(
            settlement=date(2024, 1, 15),
            maturity=date(2034, 1, 15),
            coupon_rate=0.04,
            frequency=2,
            notional=1_000_000
        )
        
        assert dv01_10y > dv01_5y


class TestSwapPricer:
    """Tests for swap pricing."""
    
    def test_at_market_swap_zero_pv(self, sample_curve):
        """Test at-market swap has near zero PV."""
        pricer = SwapPricer(sample_curve)
        
        # First find par rate
        par_rate = pricer.par_rate(
            effective=date(2024, 1, 17),
            maturity=date(2029, 1, 17)
        )
        
        # Price swap at par rate
        pv = pricer.present_value(
            effective=date(2024, 1, 17),
            maturity=date(2029, 1, 17),
            notional=10_000_000,
            fixed_rate=par_rate,
            pay_receive="PAY"
        )
        
        # PV should be close to zero
        assert abs(pv) < 10_000  # Within $10k for $10M notional
    
    def test_pay_fixed_vs_receive_fixed(self, sample_curve):
        """Test pay fixed and receive fixed have opposite signs."""
        pricer = SwapPricer(sample_curve)
        
        pv_pay = pricer.present_value(
            effective=date(2024, 1, 17),
            maturity=date(2029, 1, 17),
            notional=10_000_000,
            fixed_rate=0.04,
            pay_receive="PAY"
        )
        
        pv_rec = pricer.present_value(
            effective=date(2024, 1, 17),
            maturity=date(2029, 1, 17),
            notional=10_000_000,
            fixed_rate=0.04,
            pay_receive="RECEIVE"
        )
        
        # Should have opposite signs
        assert pv_pay * pv_rec < 0 or (abs(pv_pay) < 1 and abs(pv_rec) < 1)
    
    def test_swap_dv01(self, sample_curve):
        """Test swap DV01 calculation."""
        pricer = SwapPricer(sample_curve)
        
        dv01 = pricer.dv01(
            effective=date(2024, 1, 17),
            maturity=date(2029, 1, 17),
            notional=10_000_000,
            fixed_rate=0.05,
            pay_receive="PAY"
        )
        
        # DV01 should be non-zero
        assert dv01 != 0


class TestFuturesPricer:
    """Tests for futures pricing."""
    
    def test_futures_price_formula(self, sample_curve):
        """Test futures price = 100 - rate."""
        pricer = FuturesPricer(sample_curve)
        
        price = pricer.price_from_rate(0.05)  # 5% rate
        assert abs(price - 95.0) < 1e-10
        
        price = pricer.price_from_rate(0.0525)  # 5.25% rate
        assert abs(price - 94.75) < 1e-10
    
    def test_implied_rate(self, sample_curve):
        """Test implied rate calculation."""
        pricer = FuturesPricer(sample_curve)
        
        rate = pricer.rate_from_price(95.0)  # Price = 95
        assert abs(rate - 0.05) < 1e-10
    
    def test_futures_dv01(self, sample_curve):
        """Test futures DV01."""
        pricer = FuturesPricer(sample_curve)
        
        contract = FuturesContract(
            contract_code="SFRH4",
            expiry=date(2024, 3, 20),
            contract_size=1_000_000,
            tick_size=0.0025,
            underlying_tenor="3M"
        )
        
        dv01 = abs(pricer.dv01(contract, num_contracts=1))
        
        # SOFR futures DV01 should be ~$25 per bp per contract
        assert 20 < dv01 < 30
