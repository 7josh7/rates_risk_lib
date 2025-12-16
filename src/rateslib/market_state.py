"""
Market state abstraction layer.

Provides a clean separation of concerns between:
- Curve state (OIS, Treasury, projection curves)
- Volatility surface state (SABR parameters by bucket)
- Combined market state

This abstraction ensures:
- No pricing logic bypasses market state
- No circular dependencies
- Clean separation of curve and vol responsibilities
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from datetime import date
import numpy as np
import pandas as pd

from .curves.curve import Curve
from .curves.nss import NelsonSiegelSvensson
from .vol.sabr import SabrParams, SabrModel
from .vol.calibration import SabrCalibrator, CalibrationResult


@dataclass
class CurveState:
    """
    Encapsulates curve market data.
    
    Cleanly separates curve building from pricing/risk logic.
    
    Attributes:
        valuation_date: Market valuation date
        discount_curve: OIS curve for discounting
        projection_curve: Curve for forward rates (often same as discount)
        treasury_curve: Treasury curve (if available)
        nss_params: Nelson-Siegel-Svensson parameters (if fitted)
        curve_metadata: Additional curve metadata (nodes, quotes, etc.)
    """
    valuation_date: date
    discount_curve: Curve
    projection_curve: Optional[Curve] = None
    treasury_curve: Optional[Curve] = None
    nss_params: Optional[dict] = None
    curve_metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Set defaults."""
        if self.projection_curve is None:
            self.projection_curve = self.discount_curve
    
    def get_discount_factor(self, t: float) -> float:
        """Get discount factor at time t."""
        return self.discount_curve.discount_factor(t)
    
    def get_forward_rate(self, t1: float, t2: float) -> float:
        """Get forward rate between t1 and t2."""
        df1 = self.projection_curve.discount_factor(t1)
        df2 = self.projection_curve.discount_factor(t2)
        return (df1 / df2 - 1) / (t2 - t1)
    
    def get_zero_rate(self, t: float) -> float:
        """Get zero rate at time t."""
        return self.discount_curve.zero_rate(t)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        result = {
            'valuation_date': self.valuation_date.isoformat(),
            'curve_metadata': self.curve_metadata
        }
        
        if self.nss_params:
            result['nss_params'] = self.nss_params
        
        return result


@dataclass
class SabrSurface:
    """
    SABR volatility surface state.
    
    Independent of curve state. Stores calibrated SABR parameters
    per bucket (expiry × tenor).
    
    Attributes:
        params_by_bucket: SABR parameters keyed by (expiry, tenor)
        calibration_results: Full calibration diagnostics per bucket
        beta: Fixed beta parameter used globally
        vol_type: "NORMAL" or "LOGNORMAL"
        fallback_buckets: List of buckets using fallback parameters
    """
    params_by_bucket: Dict[Tuple[str, str], SabrParams] = field(default_factory=dict)
    calibration_results: Dict[Tuple[str, str], CalibrationResult] = field(default_factory=dict)
    beta: float = 0.5
    vol_type: str = "NORMAL"
    fallback_buckets: List[Tuple[str, str]] = field(default_factory=list)
    
    def get_params(self, expiry: str, tenor: str) -> Optional[SabrParams]:
        """
        Get SABR parameters for a bucket.
        
        Args:
            expiry: Expiry tenor (e.g., "1Y", "2Y")
            tenor: Underlying tenor (e.g., "5Y", "10Y")
            
        Returns:
            SabrParams or None if not calibrated
        """
        return self.params_by_bucket.get((expiry, tenor))
    
    def set_params(
        self,
        expiry: str,
        tenor: str,
        params: SabrParams,
        calibration_result: Optional[CalibrationResult] = None,
        is_fallback: bool = False
    ):
        """
        Set SABR parameters for a bucket.
        
        Args:
            expiry: Expiry tenor
            tenor: Underlying tenor
            params: SABR parameters
            calibration_result: Calibration diagnostics
            is_fallback: True if using fallback parameters
        """
        bucket = (expiry, tenor)
        self.params_by_bucket[bucket] = params
        
        if calibration_result:
            self.calibration_results[bucket] = calibration_result
        
        if is_fallback and bucket not in self.fallback_buckets:
            self.fallback_buckets.append(bucket)
    
    def get_diagnostics(self, expiry: str, tenor: str) -> Optional[CalibrationResult]:
        """Get calibration diagnostics for a bucket."""
        return self.calibration_results.get((expiry, tenor))
    
    def list_buckets(self) -> List[Tuple[str, str]]:
        """List all calibrated buckets."""
        return list(self.params_by_bucket.keys())
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        result = {
            'beta': self.beta,
            'vol_type': self.vol_type,
            'buckets': []
        }
        
        for (expiry, tenor), params in self.params_by_bucket.items():
            bucket_data = {
                'expiry': expiry,
                'tenor': tenor,
                'params': params.to_dict(),
                'is_fallback': (expiry, tenor) in self.fallback_buckets
            }
            
            # Add diagnostics if available
            diag = self.calibration_results.get((expiry, tenor))
            if diag:
                bucket_data['fit_error'] = diag.fit_error
                bucket_data['success'] = diag.success
            
            result['buckets'].append(bucket_data)
        
        return result


@dataclass
class MarketState:
    """
    Combined market state: curves + volatility.
    
    This is the single source of truth for market data.
    All pricing and risk calculations must go through MarketState.
    
    Design principles:
    - No circular dependencies
    - Clean separation of curve and vol responsibilities
    - No pricing logic in MarketState itself
    - Immutable after construction (for thread safety)
    
    Attributes:
        curve_state: Curve market data
        sabr_surface: SABR volatility surface (optional)
        metadata: Additional market metadata
    """
    curve_state: CurveState
    sabr_surface: Optional[SabrSurface] = None
    metadata: Dict = field(default_factory=dict)
    
    @property
    def valuation_date(self) -> date:
        """Get valuation date from curve state."""
        return self.curve_state.valuation_date
    
    @property
    def discount_curve(self) -> Curve:
        """Get discount curve."""
        return self.curve_state.discount_curve
    
    @property
    def projection_curve(self) -> Curve:
        """Get projection curve."""
        return self.curve_state.projection_curve
    
    def has_sabr_surface(self) -> bool:
        """Check if SABR surface is available."""
        return self.sabr_surface is not None
    
    def get_sabr_params(self, expiry: str, tenor: str) -> Optional[SabrParams]:
        """
        Get SABR parameters for a bucket.
        
        Args:
            expiry: Expiry tenor
            tenor: Underlying tenor
            
        Returns:
            SabrParams or None if not available
        """
        if not self.has_sabr_surface():
            return None
        return self.sabr_surface.get_params(expiry, tenor)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        result = {
            'curve_state': self.curve_state.to_dict(),
            'metadata': self.metadata
        }
        
        if self.sabr_surface:
            result['sabr_surface'] = self.sabr_surface.to_dict()
        
        return result
    
    @classmethod
    def from_curves(
        cls,
        valuation_date: date,
        discount_curve: Curve,
        projection_curve: Optional[Curve] = None,
        treasury_curve: Optional[Curve] = None,
        nss_params: Optional[dict] = None
    ) -> "MarketState":
        """
        Create MarketState from curves only (no vol surface).
        
        Args:
            valuation_date: Valuation date
            discount_curve: OIS discount curve
            projection_curve: Projection curve (defaults to discount)
            treasury_curve: Treasury curve
            nss_params: NSS parameters
            
        Returns:
            MarketState with curves only
        """
        curve_state = CurveState(
            valuation_date=valuation_date,
            discount_curve=discount_curve,
            projection_curve=projection_curve,
            treasury_curve=treasury_curve,
            nss_params=nss_params
        )
        
        return cls(curve_state=curve_state)
    
    @classmethod
    def from_curves_and_sabr(
        cls,
        valuation_date: date,
        discount_curve: Curve,
        sabr_surface: SabrSurface,
        projection_curve: Optional[Curve] = None,
        treasury_curve: Optional[Curve] = None,
        nss_params: Optional[dict] = None
    ) -> "MarketState":
        """
        Create MarketState with both curves and SABR surface.
        
        Args:
            valuation_date: Valuation date
            discount_curve: OIS discount curve
            sabr_surface: Calibrated SABR surface
            projection_curve: Projection curve
            treasury_curve: Treasury curve
            nss_params: NSS parameters
            
        Returns:
            MarketState with curves and vol surface
        """
        curve_state = CurveState(
            valuation_date=valuation_date,
            discount_curve=discount_curve,
            projection_curve=projection_curve,
            treasury_curve=treasury_curve,
            nss_params=nss_params
        )
        
        return cls(curve_state=curve_state, sabr_surface=sabr_surface)


def build_sabr_surface_from_quotes(
    vol_quotes_df: pd.DataFrame,
    forward_rates: Dict[Tuple[str, str], float],
    beta: float = 0.5,
    vol_type: str = "NORMAL"
) -> SabrSurface:
    """
    Build SABR surface by calibrating to vol quotes.
    
    Calibrates SABR parameters per bucket (expiry × tenor).
    
    Args:
        vol_quotes_df: Vol quotes with columns:
            - expiry: str (e.g., "1Y")
            - underlying_tenor: str (e.g., "5Y")
            - strike_type: str ("ATM", "BPS")
            - strike: float (0 for ATM, bp offset for BPS)
            - vol: float
            - vol_type: str
        forward_rates: Forward swap rates by (expiry, tenor)
        beta: Fixed SABR beta parameter
        vol_type: Default vol type if not in quotes
        
    Returns:
        Calibrated SabrSurface
    """
    surface = SabrSurface(beta=beta, vol_type=vol_type)
    calibrator = SabrCalibrator(beta=beta)
    
    # Group quotes by bucket
    grouped = vol_quotes_df.groupby(['expiry', 'underlying_tenor'])
    
    for (expiry, tenor), group in grouped:
        # Get forward rate for this bucket
        forward = forward_rates.get((expiry, tenor))
        if forward is None:
            # Skip this bucket if no forward rate
            surface.set_params(
                expiry, tenor,
                SabrParams(sigma_atm=0.005, beta=beta, rho=0.0, nu=0.3),
                is_fallback=True
            )
            continue
        
        # Convert quotes to calibration format
        calibration_df = _prepare_calibration_data(group, forward, vol_type)
        
        if len(calibration_df) < 2:
            # Not enough data for calibration, use fallback
            atm_vol = calibration_df['vol'].iloc[0] if len(calibration_df) > 0 else 0.005
            surface.set_params(
                expiry, tenor,
                SabrParams(sigma_atm=atm_vol, beta=beta, rho=0.0, nu=0.3),
                is_fallback=True
            )
            continue
        
        # Calibrate SABR
        try:
            # Parse expiry to years
            from .dates import DateUtils
            T = DateUtils.tenor_to_years(expiry)
            
            result = calibrator.fit(
                quotes_df=calibration_df,
                F=forward,
                T=T,
                shift=0.0,
                vol_type=vol_type
            )
            
            surface.set_params(expiry, tenor, result.params, result, is_fallback=not result.success)
        
        except Exception as e:
            # Calibration failed, use fallback
            atm_vol = calibration_df['vol'].iloc[0]
            surface.set_params(
                expiry, tenor,
                SabrParams(sigma_atm=atm_vol, beta=beta, rho=0.0, nu=0.3),
                is_fallback=True
            )
    
    return surface


def _prepare_calibration_data(
    group: pd.DataFrame,
    forward: float,
    vol_type: str
) -> pd.DataFrame:
    """
    Prepare quote data for SABR calibration.
    
    Converts strike_type and strike to absolute strike levels.
    
    Args:
        group: Quotes for a single bucket
        forward: Forward rate for this bucket
        vol_type: Vol quoting convention
        
    Returns:
        DataFrame with columns [strike, vol, weight]
    """
    calibration_data = []
    
    for _, row in group.iterrows():
        strike_type = row['strike_type']
        strike_value = row['strike']
        vol = row['vol']
        
        # Convert to absolute strike
        if strike_type == 'ATM':
            strike = forward
        elif strike_type == 'BPS':
            strike = forward + strike_value / 10000.0
        else:
            # Absolute strike
            strike = strike_value
        
        calibration_data.append({
            'strike': strike,
            'vol': vol,
            'weight': 1.0  # Equal weight by default
        })
    
    return pd.DataFrame(calibration_data)
