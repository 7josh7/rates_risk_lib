"""
Tests for options pricing module.
"""

import pytest
import numpy as np
from datetime import date

from rateslib.options.base_models import (
    bachelier_call,
    bachelier_put,
    black76_call,
    black76_put,
    shifted_black_call,
    shifted_black_put,
    bachelier_greeks,
    black76_greeks,
    implied_vol_bachelier,
    implied_vol_black,
)


class TestBachelierModel:
    """Tests for Bachelier (normal) model."""
    
    def test_bachelier_call_atm(self):
        """Test ATM call price."""
        F = 0.04
        K = 0.04
        T = 1.0
        vol = 0.005  # 50 bps normal vol
        df = 0.96
        
        price = bachelier_call(F, K, T, vol, df)
        
        # ATM call has specific formula: vol * sqrt(T) * N'(0) * df
        # N'(0) = 1/sqrt(2*pi) ≈ 0.3989
        expected = vol * np.sqrt(T) * 0.3989 * df
        np.testing.assert_allclose(price, expected, rtol=0.01)
    
    def test_bachelier_put_atm(self):
        """Test ATM put price equals ATM call."""
        F = 0.04
        K = 0.04
        T = 1.0
        vol = 0.005
        df = 0.96
        
        call = bachelier_call(F, K, T, vol, df)
        put = bachelier_put(F, K, T, vol, df)
        
        # ATM call = ATM put (by symmetry)
        np.testing.assert_allclose(call, put, rtol=0.01)
    
    def test_bachelier_put_call_parity(self):
        """Test put-call parity: C - P = df * (F - K)."""
        F = 0.04
        K = 0.035
        T = 1.0
        vol = 0.005
        df = 0.96
        
        call = bachelier_call(F, K, T, vol, df)
        put = bachelier_put(F, K, T, vol, df)
        
        # Put-call parity
        np.testing.assert_allclose(call - put, df * (F - K), rtol=0.001)
    
    def test_bachelier_greeks(self):
        """Test Greek calculations."""
        F = 0.04
        K = 0.04
        T = 1.0
        vol = 0.005
        df = 0.96
        
        greeks = bachelier_greeks(F, K, T, vol, df, is_call=True)
        
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'vega' in greeks
        assert 'theta' in greeks
        
        # ATM delta should be ~0.5 * df
        np.testing.assert_allclose(greeks['delta'], 0.5 * df, rtol=0.05)
        
        # Gamma and vega should be positive
        assert greeks['gamma'] > 0
        assert greeks['vega'] > 0
    
    def test_bachelier_call_intrinsic(self):
        """Test deep ITM call approaches intrinsic."""
        F = 0.05
        K = 0.03  # Deep ITM
        T = 0.01  # Very short expiry
        vol = 0.005
        df = 0.99
        
        call = bachelier_call(F, K, T, vol, df)
        intrinsic = df * max(F - K, 0)
        
        np.testing.assert_allclose(call, intrinsic, rtol=0.1)


class TestBlack76Model:
    """Tests for Black'76 (lognormal) model."""
    
    def test_black76_call_atm(self):
        """Test ATM call price."""
        F = 0.04
        K = 0.04
        T = 1.0
        vol = 0.20  # 20% Black vol
        df = 0.96
        
        price = black76_call(F, K, T, vol, df)
        
        # Should be positive
        assert price > 0
        # ATM approximation: F * vol * sqrt(T) * 0.4 * df
        expected_order = F * vol * np.sqrt(T) * 0.4 * df
        assert 0.5 * expected_order < price < 1.5 * expected_order
    
    def test_black76_put_call_parity(self):
        """Test put-call parity: C - P = df * (F - K)."""
        F = 0.04
        K = 0.035
        T = 1.0
        vol = 0.20
        df = 0.96
        
        call = black76_call(F, K, T, vol, df)
        put = black76_put(F, K, T, vol, df)
        
        np.testing.assert_allclose(call - put, df * (F - K), rtol=0.001)
    
    def test_black76_greeks(self):
        """Test Greek calculations."""
        F = 0.04
        K = 0.04
        T = 1.0
        vol = 0.20
        df = 0.96
        
        greeks = black76_greeks(F, K, T, vol, df, is_call=True)
        
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'vega' in greeks
        assert 'theta' in greeks
        
        # ATM delta should be ~0.5 * df
        np.testing.assert_allclose(greeks['delta'], 0.5 * df, rtol=0.1)
    
    def test_black76_negative_rates_fail(self):
        """Test that Black76 raises with negative rates."""
        F = -0.01  # Negative forward
        K = -0.01
        T = 1.0
        vol = 0.20
        df = 0.96
        
        # Should raise ValueError for negative F/K (log undefined)
        with pytest.raises(ValueError):
            black76_call(F, K, T, vol, df)


class TestShiftedBlack:
    """Tests for shifted Black model."""
    
    def test_shifted_black_negative_rates(self):
        """Test shifted Black handles negative rates."""
        F = -0.01  # Negative forward
        K = -0.01
        T = 1.0
        vol = 0.20
        shift = 0.03  # 3% shift
        df = 0.96
        
        price = shifted_black_call(F, K, T, vol, shift, df)
        
        # Should produce positive price
        assert price > 0
    
    def test_shifted_black_reduces_to_black(self):
        """Test shifted Black with shift=0 equals Black."""
        F = 0.04
        K = 0.04
        T = 1.0
        vol = 0.20
        df = 0.96
        
        black_price = black76_call(F, K, T, vol, df)
        shifted_price = shifted_black_call(F, K, T, vol, 0.0, df)
        
        np.testing.assert_allclose(black_price, shifted_price, rtol=0.01)


class TestImpliedVol:
    """Tests for implied vol calculations."""
    
    def test_implied_vol_bachelier_round_trip(self):
        """Test implied vol inverts price correctly."""
        F = 0.04
        K = 0.04
        T = 1.0
        vol = 0.005
        df = 0.96
        
        price = bachelier_call(F, K, T, vol, df)
        implied = implied_vol_bachelier(price, F, K, T, df, is_call=True)
        
        np.testing.assert_allclose(implied, vol, rtol=0.01)
    
    def test_implied_vol_black_round_trip(self):
        """Test Black implied vol inverts correctly."""
        F = 0.04
        K = 0.04
        T = 1.0
        vol = 0.20
        df = 0.96
        
        price = black76_call(F, K, T, vol, df)
        implied = implied_vol_black(price, F, K, T, df, is_call=True)
        
        np.testing.assert_allclose(implied, vol, rtol=0.01)


class TestCapletPricer:
    """Tests for caplet pricer."""
    
    @pytest.fixture
    def discount_curve(self):
        """Create a simple discount curve."""
        from rateslib.curves import Curve
        from datetime import date
        
        curve = Curve(anchor_date=date(2024, 1, 15))
        # Add nodes at various tenors with discount factors
        # DF(t) = exp(-r*t) with r around 4.5%
        curve.add_node(0.25, np.exp(-0.045 * 0.25))   # 3M
        curve.add_node(0.5, np.exp(-0.046 * 0.5))     # 6M
        curve.add_node(1.0, np.exp(-0.048 * 1.0))     # 1Y
        curve.add_node(2.0, np.exp(-0.045 * 2.0))     # 2Y
        curve.add_node(5.0, np.exp(-0.042 * 5.0))     # 5Y
        curve.build()
        return curve
    
    def test_caplet_price(self, discount_curve):
        """Test basic caplet pricing."""
        from rateslib.options.caplet import CapletPricer
        
        pricer = CapletPricer(
            discount_curve=discount_curve,
            projection_curve=discount_curve
        )
        
        # Get forward rate for the period
        F = pricer.forward_rate(1.0, 1.25)
        K = 0.04
        T = 1.0
        df = discount_curve.discount_factor(1.25)
        vol = 0.005  # 50bp normal vol
        
        price = pricer.price(
            F=F,
            K=K,
            T=T,
            df=df,
            vol=vol,
            vol_type="NORMAL",
            notional=1_000_000,
            delta_t=0.25,
            is_cap=True
        )
        
        assert price > 0
    
    def test_caplet_greeks(self, discount_curve):
        """Test caplet Greek calculations."""
        from rateslib.options.caplet import CapletPricer
        
        pricer = CapletPricer(
            discount_curve=discount_curve,
            projection_curve=discount_curve
        )
        
        F = pricer.forward_rate(1.0, 1.25)
        K = 0.04
        T = 1.0
        df = discount_curve.discount_factor(1.25)
        vol = 0.005
        
        greeks = pricer.greeks(
            F=F,
            K=K,
            T=T,
            df=df,
            vol=vol,
            vol_type="NORMAL",
            is_cap=True,
            notional=1_000_000,
            delta_t=0.25
        )
        
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'vega' in greeks


class TestSwaptionPricer:
    """Tests for swaption pricer."""
    
    @pytest.fixture
    def discount_curve(self):
        """Create a simple discount curve."""
        from rateslib.curves import Curve
        from datetime import date
        
        curve = Curve(anchor_date=date(2024, 1, 15))
        # Add nodes at various tenors
        curve.add_node(0.25, np.exp(-0.045 * 0.25))   # 3M
        curve.add_node(0.5, np.exp(-0.046 * 0.5))     # 6M
        curve.add_node(1.0, np.exp(-0.048 * 1.0))     # 1Y
        curve.add_node(2.0, np.exp(-0.045 * 2.0))     # 2Y
        curve.add_node(5.0, np.exp(-0.042 * 5.0))     # 5Y
        curve.add_node(10.0, np.exp(-0.040 * 10.0))   # 10Y
        curve.build()
        return curve
    
    def test_forward_swap_rate(self, discount_curve):
        """Test forward swap rate calculation."""
        from rateslib.options.swaption import SwaptionPricer
        
        pricer = SwaptionPricer(
            discount_curve=discount_curve,
            projection_curve=discount_curve
        )
        
        S, annuity = pricer.forward_swap_rate(expiry=1.0, tenor=5.0)
        
        # Forward swap rate should be positive and reasonable
        assert S > 0
        assert S < 0.1  # Less than 10%
        
        # Annuity should be positive
        assert annuity > 0
    
    def test_swaption_price(self, discount_curve):
        """Test basic swaption pricing."""
        from rateslib.options.swaption import SwaptionPricer
        
        pricer = SwaptionPricer(
            discount_curve=discount_curve,
            projection_curve=discount_curve
        )
        
        S, annuity = pricer.forward_swap_rate(expiry=1.0, tenor=5.0)
        
        price = pricer.price(
            S=S,
            K=S,  # ATM
            T=1.0,
            annuity=annuity,
            vol=0.005,
            vol_type="NORMAL",
            payer_receiver="PAYER",
            notional=10_000_000
        )
        
        assert price > 0


class TestSabrOptionRisk:
    """Tests for SABR risk analytics."""
    
    @pytest.fixture
    def sabr_params(self):
        from rateslib.vol.sabr import SabrParams
        return SabrParams(
            sigma_atm=0.20,  # 20% Black vol
            beta=0.5,
            rho=-0.2,
            nu=0.4
        )
    
    def test_risk_report(self, sabr_params):
        """Test risk report generation."""
        from rateslib.options.sabr_risk import SabrOptionRisk
        
        engine = SabrOptionRisk(vol_type="NORMAL")
        
        report = engine.risk_report(
            F=0.04,
            K=0.04,
            T=1.0,
            sabr_params=sabr_params,
            annuity=4.5,
            is_call=True,
            notional=10_000_000
        )
        
        # Check all risk metrics are computed
        assert hasattr(report, 'delta_base')
        assert hasattr(report, 'delta_sabr')
        assert hasattr(report, 'gamma_base')
        assert hasattr(report, 'vega_atm')
        assert hasattr(report, 'vanna')
        assert hasattr(report, 'volga')
    
    def test_delta_decomposition(self, sabr_params):
        """Test delta decomposition into sideways and backbone."""
        from rateslib.options.sabr_risk import SabrOptionRisk
        
        engine = SabrOptionRisk(vol_type="NORMAL")
        
        decomp = engine.delta_decomposition(
            F=0.04,
            K=0.04,
            T=1.0,
            sabr_params=sabr_params,
            annuity=4.5,
            is_call=True
        )
        
        assert 'sideways' in decomp
        assert 'backbone' in decomp
        assert 'total' in decomp
        
        # Total should equal sideways + backbone
        np.testing.assert_allclose(
            decomp['total'],
            decomp['sideways'] + decomp['backbone'],
            rtol=0.001
        )
    
    def test_model_consistent_delta_differs(self, sabr_params):
        """Test that model-consistent delta differs from base delta."""
        from rateslib.options.sabr_risk import SabrOptionRisk
        
        engine = SabrOptionRisk(vol_type="NORMAL")
        
        # For OTM option, backbone component should be noticeable
        report = engine.risk_report(
            F=0.04,
            K=0.045,  # OTM
            T=1.0,
            sabr_params=sabr_params,
            annuity=4.5,
            is_call=True,
            notional=10_000_000
        )
        
        # With non-zero rho and nu, backbone should be non-zero
        assert abs(report.delta_backbone) > 0
