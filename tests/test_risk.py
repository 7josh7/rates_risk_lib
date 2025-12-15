"""
Unit tests for risk module.
"""

from datetime import date, timedelta
import numpy as np
import pytest

from rateslib.curves import Curve, create_flat_curve
from rateslib.risk import BumpEngine, RiskCalculator, KeyRateEngine


@pytest.fixture
def sample_curve():
    """Create sample discount curve."""
    anchor_date = date(2024, 1, 15)
    return create_flat_curve(anchor_date, rate=0.05, max_tenor_years=30.0)


class TestBumpEngine:
    """Tests for bump engine."""
    
    def test_parallel_bump_up(self, sample_curve):
        """Test parallel upward bump."""
        engine = BumpEngine(sample_curve)
        bumped = engine.parallel_bump(10)  # +10bp
        
        # All zero rates should increase
        anchor = sample_curve.anchor_date
        for days in [90, 365, 730]:
            target = anchor + timedelta(days=days)
            orig = sample_curve.zero_rate(target)
            new = bumped.zero_rate(target)
            assert new > orig
    
    def test_parallel_bump_down(self, sample_curve):
        """Test parallel downward bump."""
        engine = BumpEngine(sample_curve)
        bumped = engine.parallel_bump(-10)  # -10bp
        
        anchor = sample_curve.anchor_date
        target = anchor + timedelta(days=365)
        
        orig = sample_curve.zero_rate(target)
        new = bumped.zero_rate(target)
        
        assert new < orig
    
    def test_node_bump(self, sample_curve):
        """Test single node bump."""
        engine = BumpEngine(sample_curve)
        
        # Get valid node index (skip index 0 which is t=0)
        nodes = sample_curve.get_nodes()
        node_idx = 1  # First non-zero node
        
        bumped = engine.node_bump(node_idx, 10)
        
        # The bumped node should be different
        orig_nodes = sample_curve.get_nodes()
        bump_nodes = bumped.get_nodes()
        
        assert bump_nodes[node_idx][2] != orig_nodes[node_idx][2]
    
    def test_compute_dv01(self, sample_curve):
        """Test DV01 computation."""
        engine = BumpEngine(sample_curve)
        
        # Simple pricer that returns sum of discount factors
        def simple_pricer(curve):
            total = 0
            anchor = curve.anchor_date
            for i in range(1, 13):  # 1 year of monthly flows
                d = anchor + timedelta(days=30*i)
                total += curve.discount_factor(d)
            return total * 1_000_000  # Scale to $1M
        
        dv01 = engine.compute_dv01(simple_pricer)
        
        # DV01 should be positive (higher rates = lower DFs = lower value)
        assert dv01 > 0
    
    def test_compute_convexity(self, sample_curve):
        """Test convexity computation."""
        engine = BumpEngine(sample_curve)
        
        def simple_pricer(curve):
            anchor = curve.anchor_date
            return curve.discount_factor(anchor + timedelta(days=3650)) * 1_000_000
        
        convexity = engine.compute_convexity(simple_pricer)
        
        # Convexity should be non-zero
        assert convexity != 0


class TestKeyRateEngine:
    """Tests for key rate calculations."""
    
    def test_key_rate_dv01(self, sample_curve):
        """Test key rate DV01 calculation."""
        engine = KeyRateEngine(sample_curve)
        
        def bond_pricer(curve):
            # Simple 5-year bond-like pricer
            anchor = curve.anchor_date
            pv = 0
            for i in range(1, 11):  # 5 years semi-annual
                d = anchor + timedelta(days=182*i)
                pv += curve.discount_factor(d) * 25  # $25 coupon
            pv += curve.discount_factor(anchor + timedelta(days=1825)) * 1000  # Principal
            return pv
        
        kr_result = engine.compute_key_rate_dv01(bond_pricer)
        
        # Should have results for standard tenors - access dv01s dict
        assert '2Y' in kr_result.dv01s or '5Y' in kr_result.dv01s or '10Y' in kr_result.dv01s
        
        # Check that we have non-zero DV01s
        total_kr = sum(kr_result.dv01s.values())
        assert total_kr != 0, "Total key-rate DV01 should not be zero"


class TestRiskCalculator:
    """Tests for risk calculator."""
    
    def test_compute_bond_risk(self, sample_curve):
        """Test bond risk calculation."""
        calculator = RiskCalculator(sample_curve)
        
        risk = calculator.compute_bond_risk(
            instrument_id="TEST_BOND",
            settlement=date(2024, 1, 15),
            maturity=date(2029, 1, 15),
            coupon_rate=0.04,
            notional=1_000_000,
            frequency=2
        )
        
        assert risk.dv01 > 0
        assert risk.pv != 0
    
    def test_compute_swap_risk(self, sample_curve):
        """Test swap risk calculation."""
        calculator = RiskCalculator(sample_curve)
        
        risk = calculator.compute_swap_risk(
            instrument_id="TEST_SWAP",
            effective=date(2024, 1, 17),
            maturity=date(2029, 1, 17),
            fixed_rate=0.05,
            notional=10_000_000,
            pay_receive="PAY"
        )
        
        assert risk.dv01 != 0
