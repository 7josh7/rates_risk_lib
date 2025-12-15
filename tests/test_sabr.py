"""
Tests for SABR volatility model.
"""

import pytest
import numpy as np
from rateslib.vol.sabr import (
    SabrParams,
    SabrModel,
    hagan_black_vol,
    hagan_normal_vol,
    _hagan_atm_vol,
)


class TestSabrParams:
    """Tests for SabrParams dataclass."""
    
    def test_default_params(self):
        """Test parameter initialization with required fields."""
        params = SabrParams(sigma_atm=0.005, beta=0.5, rho=0.0, nu=0.3)
        assert params.sigma_atm == 0.005
        assert params.beta == 0.5
        assert params.rho == 0.0
        assert params.nu == 0.3
        assert params.shift == 0.0
    
    def test_full_params(self):
        """Test full parameter initialization."""
        params = SabrParams(
            sigma_atm=0.006,
            beta=0.7,
            rho=-0.2,
            nu=0.4,
            shift=0.02
        )
        assert params.sigma_atm == 0.006
        assert params.beta == 0.7
        assert params.rho == -0.2
        assert params.nu == 0.4
        assert params.shift == 0.02


class TestHaganFormulas:
    """Tests for Hagan SABR approximation."""
    
    def test_atm_vol(self):
        """Test ATM vol approximation."""
        F = 0.04
        T = 1.0
        alpha = 0.03
        beta = 0.5
        rho = -0.2
        nu = 0.4
        
        vol = _hagan_atm_vol(F, T, alpha, beta, rho, nu)
        assert vol > 0
        # ATM vol should be roughly proportional to alpha / F^(1-beta)
        expected_order = alpha / (F ** (1 - beta))
        assert 0.5 * expected_order < vol < 2.0 * expected_order
    
    def test_hagan_black_vol_atm(self):
        """Test Black vol at ATM."""
        F = 0.04
        K = F  # ATM
        T = 1.0
        alpha = 0.03
        beta = 0.5
        rho = -0.2
        nu = 0.4
        
        vol = hagan_black_vol(F, K, T, alpha, beta, rho, nu)
        assert vol > 0
        assert vol < 1.0  # Reasonable bound for Black vol
    
    def test_hagan_black_vol_away_from_atm(self):
        """Test Black vol away from ATM."""
        F = 0.04
        T = 1.0
        alpha = 0.03
        beta = 0.5
        rho = -0.2
        nu = 0.4
        
        K_low = F - 0.01
        K_high = F + 0.01
        
        vol_low = hagan_black_vol(F, K_low, T, alpha, beta, rho, nu)
        vol_atm = hagan_black_vol(F, F, T, alpha, beta, rho, nu)
        vol_high = hagan_black_vol(F, K_high, T, alpha, beta, rho, nu)
        
        # All should be positive
        assert vol_low > 0
        assert vol_atm > 0
        assert vol_high > 0
        
        # With negative rho, smile typically has skew (higher vol for low strikes)
        assert vol_low > vol_atm  # Typical for rates with negative rho
    
    def test_hagan_normal_vol(self):
        """Test normal vol approximation."""
        F = 0.04
        K = 0.04
        T = 1.0
        alpha = 0.03
        beta = 0.5
        rho = -0.2
        nu = 0.4
        
        vol = hagan_normal_vol(F, K, T, alpha, beta, rho, nu)
        assert vol > 0
        # Normal vol should be much smaller than Black vol (in rate units)
        assert vol < 0.1  # Normal vol in rate units


class TestSabrModel:
    """Tests for SabrModel class."""
    
    @pytest.fixture
    def model(self):
        return SabrModel()
    
    @pytest.fixture
    def params(self):
        # sigma_atm here is Black ATM vol (not normal)
        return SabrParams(
            sigma_atm=0.20,  # 20% Black vol
            beta=0.5,
            rho=-0.2,
            nu=0.4,
            shift=0.0
        )
    
    def test_alpha_from_sigma_atm(self, model, params):
        """Test alpha inversion from ATM vol."""
        F = 0.04
        T = 1.0
        
        alpha = model.alpha_from_sigma_atm(F, T, params)
        assert alpha > 0
        
        # Verify by recomputing ATM Black vol with this alpha
        computed_vol = _hagan_atm_vol(F, T, alpha, params.beta, params.rho, params.nu)
        np.testing.assert_allclose(computed_vol, params.sigma_atm, rtol=0.01)
    
    def test_implied_vol_normal(self, model, params):
        """Test normal implied vol computation."""
        F = 0.04
        K = 0.04
        T = 1.0
        
        vol = model.implied_vol_normal(F, K, T, params)
        # Normal vol should be roughly Black vol * F at ATM
        expected_order = params.sigma_atm * F
        assert vol > 0
        assert 0.5 * expected_order < vol < 2.0 * expected_order
    
    def test_implied_vol_black(self, model, params):
        """Test Black implied vol computation."""
        F = 0.04
        K = 0.04
        T = 1.0
        
        vol = model.implied_vol_black(F, K, T, params)
        assert vol > 0
        assert vol < 1.0
    
    def test_shifted_sabr(self, model):
        """Test shifted SABR for negative rates."""
        params = SabrParams(
            sigma_atm=0.20,  # Black vol
            beta=0.5,
            rho=-0.2,
            nu=0.4,
            shift=0.02  # 2% shift
        )
        
        F = -0.005  # Negative forward
        K = -0.005
        T = 1.0
        
        # Without shift, this would fail
        # With shift, should work
        vol = model.implied_vol_normal(F, K, T, params)
        assert vol > 0
    
    def test_smile_at_strikes(self, model, params):
        """Test smile computation across strikes."""
        F = 0.04
        T = 1.0
        strikes = np.array([0.03, 0.035, 0.04, 0.045, 0.05])
        
        vols = model.smile_at_strikes(F, strikes, T, params)
        
        assert len(vols) == len(strikes)
        assert all(v > 0 for v in vols.values())
    
    def test_dsigma_dF(self, model, params):
        """Test vol sensitivity to forward."""
        F = 0.04
        K = 0.04
        T = 1.0
        
        dsigma = model.dsigma_dF(F, K, T, params)
        
        # Verify numerically (keeping alpha fixed for sideways)
        alpha = model.alpha_from_sigma_atm(F, T, params)
        eps = 1e-6
        vol_up = hagan_black_vol(F + eps, K, T, alpha, params.beta, params.rho, params.nu, params.shift)
        vol_dn = hagan_black_vol(F - eps, K, T, alpha, params.beta, params.rho, params.nu, params.shift)
        numerical = (vol_up - vol_dn) / (2 * eps)
        
        np.testing.assert_allclose(dsigma, numerical, rtol=0.1)
    
    def test_dsigma_drho(self, model, params):
        """Test vol sensitivity to rho."""
        F = 0.04
        K = 0.035  # OTM for more sensitivity
        T = 1.0
        
        dsigma = model.dsigma_drho(F, K, T, params)
        
        # Should be non-zero for OTM
        assert abs(dsigma) > 0


class TestSabrCalibration:
    """Tests for SABR calibration."""
    
    def test_basic_calibration(self):
        """Test calibration to synthetic market data."""
        from rateslib.vol.calibration import SabrCalibrator
        import pandas as pd
        
        # Generate synthetic market data from known SABR params
        # Using Black vol parameterization
        true_params = SabrParams(sigma_atm=0.20, beta=0.5, rho=-0.15, nu=0.35)
        model = SabrModel()
        
        F = 0.04
        T = 1.0
        strikes = np.array([0.03, 0.035, 0.04, 0.045, 0.05])
        market_vols = np.array([model.implied_vol_black(F, K, T, true_params) for K in strikes])
        
        # Create DataFrame for calibrator
        quotes_df = pd.DataFrame({'strike': strikes, 'vol': market_vols})
        
        # Calibrate
        calibrator = SabrCalibrator(beta=0.5)
        result = calibrator.fit(quotes_df, F, T, vol_type="LOGNORMAL")
        
        # Check calibration quality
        assert result.fit_error < 0.01  # Should fit reasonably well
        # Parameters may not match exactly due to numerical optimization
        assert result.params.sigma_atm > 0
    
    def test_calibration_with_noise(self):
        """Test calibration robustness to noisy data."""
        from rateslib.vol.calibration import SabrCalibrator
        import pandas as pd
        
        true_params = SabrParams(sigma_atm=0.20, beta=0.5, rho=-0.15, nu=0.35)
        model = SabrModel()
        
        F = 0.04
        T = 1.0
        strikes = np.array([0.03, 0.035, 0.04, 0.045, 0.05])
        market_vols = np.array([model.implied_vol_black(F, K, T, true_params) for K in strikes])
        
        # Add noise (1% relative)
        np.random.seed(42)
        market_vols = market_vols + np.random.normal(0, 0.002, len(strikes))
        
        quotes_df = pd.DataFrame({'strike': strikes, 'vol': market_vols})
        
        calibrator = SabrCalibrator(beta=0.5)
        result = calibrator.fit(quotes_df, F, T, vol_type="LOGNORMAL")
        
        # Should still calibrate reasonably well
        assert result.fit_error < 0.1


class TestVolQuotes:
    """Tests for vol quote handling."""
    
    def test_vol_quote_creation(self):
        """Test VolQuote dataclass."""
        from rateslib.vol.quotes import VolQuote
        from datetime import date
        
        quote = VolQuote(
            quote_date=date(2024, 1, 15),
            expiry="1Y",
            underlying_tenor="5Y",
            strike="ATM",
            vol=0.0045,
            vol_type="NORMAL"
        )
        
        assert quote.expiry == "1Y"
        assert quote.underlying_tenor == "5Y"
        assert quote.vol == 0.0045
    
    def test_strike_value(self):
        """Test strike value calculation."""
        from rateslib.vol.quotes import VolQuote
        from datetime import date
        
        F = 0.04
        
        # ATM
        atm = VolQuote(date(2024, 1, 15), "1Y", "5Y", "ATM", 0.005, "NORMAL")
        assert atm.strike_value(F) == F
        
        # BPS offset (+50bp)
        otm = VolQuote(date(2024, 1, 15), "1Y", "5Y", "+50bp", 0.005, "NORMAL")
        np.testing.assert_allclose(otm.strike_value(F), F + 0.005)
        
        # Fixed strike (numeric)
        fixed = VolQuote(date(2024, 1, 15), "1Y", "5Y", 0.045, 0.005, "NORMAL")
        assert fixed.strike_value(F) == 0.045
