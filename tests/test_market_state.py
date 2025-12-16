"""
Tests for SABR Greeks API fixes and Market State.
"""

import pytest
import numpy as np
from datetime import date

from rateslib.vol.sabr import SabrParams, SabrModel
from rateslib.market_state import CurveState, SabrSurface, MarketState
from rateslib.curves import Curve
from rateslib.options import SwaptionPricer, SabrOptionRisk


class TestSabrGreeksAPI:
    """Test SABR Greeks API accepts vol_type parameter."""
    
    def test_dsigma_drho_accepts_vol_type(self):
        """Test dsigma_drho accepts BLACK and NORMAL vol_type."""
        model = SabrModel()
        params = SabrParams(sigma_atm=0.005, beta=0.5, rho=-0.2, nu=0.3)
        
        F = 0.04
        K = 0.045
        T = 1.0
        
        # Should not raise error with BLACK
        result_black = model.dsigma_drho(F, K, T, params, vol_type="BLACK")
        assert isinstance(result_black, float)
        
        # Should not raise error with NORMAL
        result_normal = model.dsigma_drho(F, K, T, params, vol_type="NORMAL")
        assert isinstance(result_normal, float)
    
    def test_dsigma_dnu_accepts_vol_type(self):
        """Test dsigma_dnu accepts BLACK and NORMAL vol_type."""
        model = SabrModel()
        params = SabrParams(sigma_atm=0.005, beta=0.5, rho=-0.2, nu=0.3)
        
        F = 0.04
        K = 0.045
        T = 1.0
        
        # Should not raise error with BLACK
        result_black = model.dsigma_dnu(F, K, T, params, vol_type="BLACK")
        assert isinstance(result_black, float)
        
        # Should not raise error with NORMAL
        result_normal = model.dsigma_dnu(F, K, T, params, vol_type="NORMAL")
        assert isinstance(result_normal, float)
    
    def test_dsigma_dF_accepts_vol_type(self):
        """Test dsigma_dF accepts BLACK and NORMAL vol_type."""
        model = SabrModel()
        params = SabrParams(sigma_atm=0.005, beta=0.5, rho=-0.2, nu=0.3)
        
        F = 0.04
        K = 0.045
        T = 1.0
        
        # Should not raise error with BLACK
        result_black = model.dsigma_dF(F, K, T, params, vol_type="BLACK")
        assert isinstance(result_black, float)
        
        # Should not raise error with NORMAL
        result_normal = model.dsigma_dF(F, K, T, params, vol_type="NORMAL")
        assert isinstance(result_normal, float)
    
    def test_sabr_option_risk_uses_vol_type(self):
        """Test SabrOptionRisk properly uses vol_type in Greeks calculation."""
        params = SabrParams(sigma_atm=0.005, beta=0.5, rho=-0.2, nu=0.3)
        
        # Test with NORMAL vol type
        risk_engine_normal = SabrOptionRisk(vol_type="NORMAL")
        report_normal = risk_engine_normal.risk_report(
            F=0.04,
            K=0.045,
            T=1.0,
            sabr_params=params,
            annuity=0.96,
            is_call=True,
            notional=1_000_000
        )
        
        assert report_normal.vol_type == "NORMAL"
        assert isinstance(report_normal.delta_sabr, float)
        assert isinstance(report_normal.vega_atm, float)
        
        # Test with LOGNORMAL vol type
        risk_engine_log = SabrOptionRisk(vol_type="LOGNORMAL")
        report_log = risk_engine_log.risk_report(
            F=0.04,
            K=0.045,
            T=1.0,
            sabr_params=params,
            annuity=0.96,
            is_call=True,
            notional=1_000_000
        )
        
        assert report_log.vol_type == "LOGNORMAL"
        assert isinstance(report_log.delta_sabr, float)
        assert isinstance(report_log.vega_atm, float)


class TestMarketState:
    """Test MarketState architecture."""
    
    def setup_method(self):
        """Set up test data."""
        self.valuation_date = date(2024, 1, 15)
        
        # Create simple discount curve
        times = [0.25, 0.5, 1, 2, 5, 10]
        dfs = [0.99, 0.98, 0.96, 0.93, 0.85, 0.75]
        self.discount_curve = Curve(self.valuation_date)
        for t, df in zip(times, dfs):
            self.discount_curve.add_node(t, df)
        self.discount_curve.build()
    
    def test_curve_state_creation(self):
        """Test CurveState creation."""
        curve_state = CurveState(
            valuation_date=self.valuation_date,
            discount_curve=self.discount_curve
        )
        
        assert curve_state.valuation_date == self.valuation_date
        assert curve_state.discount_curve == self.discount_curve
        assert curve_state.projection_curve == self.discount_curve  # Default to discount
    
    def test_curve_state_methods(self):
        """Test CurveState convenience methods."""
        curve_state = CurveState(
            valuation_date=self.valuation_date,
            discount_curve=self.discount_curve
        )
        
        # Test discount factor
        df = curve_state.get_discount_factor(1.0)
        assert 0 < df < 1
        
        # Test zero rate
        zr = curve_state.get_zero_rate(1.0)
        assert zr > 0
        
        # Test forward rate
        fr = curve_state.get_forward_rate(1.0, 2.0)
        assert isinstance(fr, float)
    
    def test_sabr_surface_creation(self):
        """Test SabrSurface creation."""
        surface = SabrSurface(beta=0.5, vol_type="NORMAL")
        
        assert surface.beta == 0.5
        assert surface.vol_type == "NORMAL"
        assert len(surface.list_buckets()) == 0
    
    def test_sabr_surface_set_get_params(self):
        """Test setting and getting SABR parameters."""
        surface = SabrSurface(beta=0.5, vol_type="NORMAL")
        
        params = SabrParams(sigma_atm=0.005, beta=0.5, rho=-0.2, nu=0.3)
        surface.set_params("1Y", "5Y", params)
        
        # Retrieve params
        retrieved = surface.get_params("1Y", "5Y")
        assert retrieved is not None
        assert retrieved.sigma_atm == 0.005
        assert retrieved.rho == -0.2
        
        # Non-existent bucket returns None
        assert surface.get_params("2Y", "10Y") is None
    
    def test_market_state_from_curves(self):
        """Test MarketState creation from curves only."""
        market_state = MarketState.from_curves(
            valuation_date=self.valuation_date,
            discount_curve=self.discount_curve
        )
        
        assert market_state.valuation_date == self.valuation_date
        assert market_state.discount_curve == self.discount_curve
        assert not market_state.has_sabr_surface()
    
    def test_market_state_with_sabr(self):
        """Test MarketState with SABR surface."""
        surface = SabrSurface(beta=0.5, vol_type="NORMAL")
        params = SabrParams(sigma_atm=0.005, beta=0.5, rho=-0.2, nu=0.3)
        surface.set_params("1Y", "5Y", params)
        
        market_state = MarketState.from_curves_and_sabr(
            valuation_date=self.valuation_date,
            discount_curve=self.discount_curve,
            sabr_surface=surface
        )
        
        assert market_state.has_sabr_surface()
        assert market_state.get_sabr_params("1Y", "5Y") is not None
        assert market_state.get_sabr_params("1Y", "5Y").sigma_atm == 0.005
    
    def test_market_state_serialization(self):
        """Test MarketState to_dict."""
        market_state = MarketState.from_curves(
            valuation_date=self.valuation_date,
            discount_curve=self.discount_curve
        )
        
        state_dict = market_state.to_dict()
        
        assert 'curve_state' in state_dict
        assert 'metadata' in state_dict
        assert state_dict['curve_state']['valuation_date'] == self.valuation_date.isoformat()


class TestOptionPricing:
    """Test option pricing properties."""
    
    def test_option_price_increases_with_vol(self):
        """Test that option price increases with volatility."""
        from rateslib.options import bachelier_call
        
        F = 0.04
        K = 0.04
        T = 1.0
        df = 0.96
        
        vol_low = 0.003
        vol_high = 0.006
        
        price_low = bachelier_call(F, K, T, vol_low, df)
        price_high = bachelier_call(F, K, T, vol_high, df)
        
        assert price_high > price_low, "Option price should increase with volatility"
    
    def test_atm_payer_receiver_symmetry(self):
        """Test ATM payer and receiver have same price (symmetry)."""
        from rateslib.options import bachelier_call, bachelier_put
        
        F = 0.04
        K = 0.04  # ATM
        T = 1.0
        vol = 0.005
        df = 0.96
        
        call_price = bachelier_call(F, K, T, vol, df)
        put_price = bachelier_put(F, K, T, vol, df)
        
        # ATM call and put should be equal (by symmetry)
        np.testing.assert_allclose(call_price, put_price, rtol=0.01)
    
    def test_sabr_greeks_sign_checks(self):
        """Test SABR Greeks have correct signs."""
        params = SabrParams(sigma_atm=0.005, beta=0.5, rho=-0.2, nu=0.3)
        risk_engine = SabrOptionRisk(vol_type="NORMAL")
        
        # Call option
        report = risk_engine.risk_report(
            F=0.04,
            K=0.04,
            T=1.0,
            sabr_params=params,
            annuity=0.96,
            is_call=True,
            notional=1_000_000
        )
        
        # Gamma should be positive
        assert report.gamma_base > 0, "Gamma should be positive for vanilla options"
        
        # Vega should be positive
        assert report.vega_atm > 0, "Vega should be positive for vanilla options"


class TestRiskLimits:
    """Test risk limits framework."""
    
    def test_limit_value_check(self):
        """Test LimitValue check method."""
        from rateslib.risk import LimitValue, LimitLevel
        
        limit = LimitValue(warning_threshold=100, breach_threshold=150)
        
        assert limit.check(50) == LimitLevel.OK
        assert limit.check(120) == LimitLevel.WARNING
        assert limit.check(200) == LimitLevel.BREACH
    
    def test_limit_utilization(self):
        """Test limit utilization calculation."""
        from rateslib.risk import LimitValue
        
        limit = LimitValue(warning_threshold=100, breach_threshold=200)
        
        assert limit.utilization(100) == 50.0
        assert limit.utilization(200) == 100.0
        assert limit.utilization(50) == 25.0
    
    def test_risk_limit_checker(self):
        """Test RiskLimitChecker."""
        from rateslib.risk import RiskLimits, RiskLimitChecker
        
        limits = RiskLimits.default_limits()
        checker = RiskLimitChecker(limits)
        
        metrics = {
            'dv01': 500_000,  # OK
            'var_95': 3_000_000,  # BREACH
        }
        
        results = checker.check_all(metrics)
        assert len(results) > 0
        
        # Check for breaches
        assert checker.has_breaches(metrics)
        breaches = checker.get_breaches(metrics)
        assert len(breaches) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
