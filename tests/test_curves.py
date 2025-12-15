"""
Unit tests for curves module.
"""

from datetime import date, timedelta
import numpy as np
import pytest

from rateslib.curves import (
    Curve,
    LinearInterpolator,
    CubicSplineInterpolator,
    LogLinearInterpolator,
    OISBootstrapper,
    NelsonSiegelSvensson,
    create_flat_curve,
    bootstrap_from_quotes,
)


class TestInterpolators:
    """Tests for interpolation methods."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample interpolation data."""
        x = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
        y = np.array([0.050, 0.051, 0.052, 0.053, 0.050, 0.048, 0.045])
        return x, y
    
    def test_linear_interpolator(self, sample_data):
        """Test linear interpolation."""
        x, y = sample_data
        interp = LinearInterpolator()
        interp.fit(x, y)
        
        # Test exact points
        assert abs(interp(0.0) - 0.050) < 1e-10
        assert abs(interp(1.0) - 0.053) < 1e-10
        
        # Test interpolated point
        result = interp(0.125)  # Between 0 and 0.25
        assert 0.050 < result < 0.051
    
    def test_cubic_spline_interpolator(self, sample_data):
        """Test cubic spline interpolation."""
        x, y = sample_data
        interp = CubicSplineInterpolator()
        interp.fit(x, y)
        
        # Test exact points
        assert abs(interp(0.0) - 0.050) < 1e-10
        assert abs(interp(1.0) - 0.053) < 1e-10
        
        # Test smoothness (should be differentiable)
        result1 = interp(1.0)
        result2 = interp(1.001)
        assert result1 != result2  # Should differ slightly
    
    def test_log_linear_interpolator(self, sample_data):
        """Test log-linear interpolation on discount factors."""
        x, y = sample_data
        # Log-linear interpolator expects discount factors, not rates
        # Create discount factors from rates: DF = exp(-r * t)
        dfs = np.exp(-y * x)  # Element-wise: exp(-rate * time)
        dfs[0] = 1.0  # DF at t=0 is always 1
        
        interp = LogLinearInterpolator()
        interp.fit(x, dfs)
        
        # Test that interpolated log_df at exact point matches
        result = interp(1.0)  # Returns log(DF)
        expected_log_df = np.log(np.exp(-0.053 * 1.0))  # log(exp(-r*t)) = -r*t
        assert abs(result - expected_log_df) < 0.01


class TestCurve:
    """Tests for Curve class."""
    
    @pytest.fixture
    def sample_curve(self):
        """Create sample discount curve using flat curve helper."""
        anchor_date = date(2024, 1, 15)
        return create_flat_curve(anchor_date, rate=0.05, max_tenor_years=30.0)
    
    def test_discount_factor_base_date(self, sample_curve):
        """Test discount factor at anchor date is 1."""
        df = sample_curve.discount_factor(sample_curve.anchor_date)
        assert abs(df - 1.0) < 1e-10
    
    def test_discount_factor_future(self, sample_curve):
        """Test discount factor decreases for future dates."""
        anchor = sample_curve.anchor_date
        df1 = sample_curve.discount_factor(anchor + timedelta(days=365))
        df2 = sample_curve.discount_factor(anchor + timedelta(days=730))
        
        assert df1 > df2  # DF should decrease
        assert df1 < 1.0
        assert df2 < df1
    
    def test_zero_rate(self, sample_curve):
        """Test zero rate calculation."""
        anchor = sample_curve.anchor_date
        rate = sample_curve.zero_rate(anchor + timedelta(days=365))
        
        # Should be around 5% for flat 5% curve
        assert 0.04 < rate < 0.06
    
    def test_forward_rate(self, sample_curve):
        """Test forward rate calculation."""
        anchor = sample_curve.anchor_date
        start = anchor + timedelta(days=365)
        end = anchor + timedelta(days=730)
        
        fwd = sample_curve.forward_rate(start, end)
        
        # Forward rate should be around 5% for flat curve
        assert 0.04 < fwd < 0.06
    
    def test_bump_parallel(self, sample_curve):
        """Test parallel bump."""
        bumped = sample_curve.bump_parallel(10)  # +10bp
        
        # Zero rates should increase
        anchor = sample_curve.anchor_date
        target = anchor + timedelta(days=365)
        
        orig_rate = sample_curve.zero_rate(target)
        bumped_rate = bumped.zero_rate(target)
        
        assert bumped_rate > orig_rate
        # Should be approximately 10bp higher
        assert abs((bumped_rate - orig_rate) * 10000 - 10) < 1
    
    def test_bump_node(self, sample_curve):
        """Test node bump."""
        # Bump a node that exists (not index 0 which is t=0)
        nodes = sample_curve.get_nodes()
        # Find a non-zero node index
        node_idx = 1  # Should be the first non-zero node
        
        bumped = sample_curve.bump_node(node_idx, 10)  # Bump by 10bp
        
        # Original node should be different from bumped
        orig_nodes = sample_curve.get_nodes()
        bump_nodes = bumped.get_nodes()
        
        # The bumped node zero rate should be ~10bp higher
        orig_zr = orig_nodes[node_idx][2]
        bump_zr = bump_nodes[node_idx][2]
        assert abs((bump_zr - orig_zr) * 10000 - 10) < 1


class TestOISBootstrap:
    """Tests for OIS bootstrapping."""
    
    def test_bootstrap_simple(self):
        """Test simple bootstrap case."""
        anchor_date = date(2024, 1, 15)
        
        # Build quotes list - need instrument_type and quote fields
        quotes = [
            {"instrument_type": "DEPOSIT", "tenor": "1W", "quote": 0.0530},
            {"instrument_type": "DEPOSIT", "tenor": "1M", "quote": 0.0525},
            {"instrument_type": "OIS", "tenor": "3M", "quote": 0.0520},
            {"instrument_type": "OIS", "tenor": "6M", "quote": 0.0510},
            {"instrument_type": "OIS", "tenor": "1Y", "quote": 0.0500},
        ]
        
        result = bootstrap_from_quotes(anchor_date, quotes)
        
        assert result is not None
        assert len(result.get_nodes()) > 0
        
        # DF should be 1 at anchor date
        assert abs(result.discount_factor(anchor_date) - 1.0) < 1e-10
    
    def test_bootstrap_monotonic_df(self):
        """Test that bootstrapped DFs are monotonically decreasing."""
        anchor_date = date(2024, 1, 15)
        
        quotes = [
            {"instrument_type": "DEPOSIT", "tenor": "1M", "quote": 0.0530},
            {"instrument_type": "OIS", "tenor": "3M", "quote": 0.0528},
            {"instrument_type": "OIS", "tenor": "6M", "quote": 0.0525},
            {"instrument_type": "OIS", "tenor": "1Y", "quote": 0.0520},
            {"instrument_type": "OIS", "tenor": "2Y", "quote": 0.0510},
        ]
        
        result = bootstrap_from_quotes(anchor_date, quotes)
        
        dfs = result.get_node_dfs()
        for i in range(1, len(dfs)):
            assert dfs[i] < dfs[i-1], "DFs should be monotonically decreasing"


class TestNSS:
    """Tests for Nelson-Siegel-Svensson fitting."""
    
    def test_nss_fit(self):
        """Test NSS fitting."""
        anchor_date = date(2024, 1, 15)
        nss = NelsonSiegelSvensson(anchor_date)
        
        tenors = np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 30])
        yields = np.array([0.052, 0.050, 0.048, 0.046, 0.044, 0.043, 0.042, 0.043, 0.044])
        
        nss.fit(tenors, yields)
        
        # Check parameters are reasonable
        assert nss.params is not None
        assert nss.params.lambda1 > 0
        assert nss.params.lambda2 > 0
    
    def test_nss_yield_at(self):
        """Test NSS yield calculation."""
        anchor_date = date(2024, 1, 15)
        nss = NelsonSiegelSvensson(anchor_date)
        
        tenors = np.array([1, 2, 5, 10, 30])
        yields = np.array([0.05, 0.048, 0.044, 0.042, 0.044])
        
        nss.fit(tenors, yields)
        
        # Test yield at fitted point should be close
        y = nss.yield_at(5)
        assert abs(y - 0.044) < 0.01  # Within 100bp
    
    def test_nss_to_curve(self):
        """Test NSS to curve conversion."""
        anchor_date = date(2024, 1, 15)
        nss = NelsonSiegelSvensson(anchor_date)
        
        tenors = np.array([1, 2, 5, 10, 30])
        yields = np.array([0.05, 0.048, 0.044, 0.042, 0.044])
        
        nss.fit(tenors, yields)
        
        curve = nss.to_curve(tenors=[0.5, 1, 2, 5, 10, 20, 30])
        
        assert curve is not None
        assert len(curve.get_nodes()) > 0
        assert curve.anchor_date == anchor_date
