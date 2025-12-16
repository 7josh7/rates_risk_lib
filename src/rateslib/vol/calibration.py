"""
SABR model calibration.

Calibrates SABR parameters from market vol quotes:
- Supports normal and lognormal vol quoting
- Uses sigma_atm parameterization (more stable)
- Fits rho and nu to match smile
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Any
import numpy as np
from scipy.optimize import minimize, differential_evolution
import pandas as pd

from .sabr import SabrParams, SabrModel, hagan_black_vol, hagan_normal_vol
from .sabr_surface import SabrSurfaceState, SabrBucketParams

if TYPE_CHECKING:
    from ..market_state import CurveState


@dataclass
class CalibrationResult:
    """Result of SABR calibration."""
    params: SabrParams
    fit_error: float
    vol_errors: Dict[float, float]  # {strike: error}
    success: bool
    message: str


class SabrCalibrator:
    """
    Calibrator for SABR parameters from market vol quotes.
    
    Uses the sigma_atm parameterization:
    1. Fix beta (from market convention or historical analysis)
    2. Extract sigma_atm from ATM quote
    3. Calibrate rho and nu to match the smile
    """
    
    def __init__(
        self,
        beta: float = 0.5,
        use_sigma_atm_param: bool = True
    ):
        """
        Initialize calibrator.
        
        Args:
            beta: Fixed CEV exponent (typically 0, 0.5, or 1)
            use_sigma_atm_param: If True, parameterize by sigma_atm
        """
        if not 0 <= beta <= 1:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        
        self.beta = beta
        self.use_sigma_atm_param = use_sigma_atm_param
        self.model = SabrModel()
    
    def fit(
        self,
        quotes_df: pd.DataFrame,
        F: float,
        T: float,
        shift: float = 0.0,
        vol_type: str = "NORMAL"
    ) -> CalibrationResult:
        """
        Fit SABR parameters to market vol quotes.
        
        Args:
            quotes_df: DataFrame with columns [strike, vol] or [strike, vol, weight]
            F: Forward rate
            T: Time to expiry in years
            shift: Shift for negative rates
            vol_type: "NORMAL" or "LOGNORMAL"
            
        Returns:
            CalibrationResult with fitted parameters
        """
        # Validate inputs
        if F + shift <= 0:
            raise ValueError(f"Shifted forward must be positive: F={F}, shift={shift}")
        if T <= 0:
            raise ValueError(f"Time to expiry must be positive: T={T}")
        
        # Extract data
        strikes = quotes_df['strike'].values
        vols = quotes_df['vol'].values
        
        # Handle weights - check if column exists in DataFrame
        if 'weight' in quotes_df.columns:
            weights = quotes_df['weight'].values
        else:
            weights = np.ones(len(strikes))
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Find ATM vol (strike closest to forward)
        atm_idx = np.argmin(np.abs(strikes - F))
        sigma_atm = vols[atm_idx]
        
        # Convert to Black vol if needed for internal calculations
        if vol_type.upper() == "NORMAL":
            # Approximate conversion: sigma_B ≈ sigma_N / F
            sigma_atm_black = sigma_atm / (F + shift)
        else:
            sigma_atm_black = sigma_atm
        
        # Define objective function
        def objective(x):
            rho, nu = x
            
            # Bounds check
            if not -0.99 < rho < 0.99:
                return 1e10
            if nu < 0.001:
                return 1e10
            
            try:
                params = SabrParams(
                    sigma_atm=sigma_atm_black,
                    beta=self.beta,
                    rho=rho,
                    nu=nu,
                    shift=shift
                )
                
                error = 0.0
                alpha = self.model.alpha_from_sigma_atm(F, T, params)
                
                for i, (K, v_mkt) in enumerate(zip(strikes, vols)):
                    if vol_type.upper() == "NORMAL":
                        v_model = hagan_normal_vol(F, K, T, alpha, self.beta, rho, nu, shift)
                    else:
                        v_model = hagan_black_vol(F, K, T, alpha, self.beta, rho, nu, shift)
                    
                    error += weights[i] * (v_model - v_mkt) ** 2
                
                return error
                
            except Exception:
                return 1e10
        
        # Initial guess
        x0 = [0.0, 0.3]  # rho=0, nu=0.3
        
        # Bounds for optimization
        bounds = [(-0.95, 0.95), (0.01, 2.0)]
        
        # Try local optimization first
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500}
        )
        
        # If local fails or error is large, try global optimization
        if not result.success or result.fun > 1e-8:
            try:
                result_de = differential_evolution(
                    objective,
                    bounds,
                    maxiter=200,
                    tol=1e-10,
                    seed=42
                )
                if result_de.fun < result.fun:
                    result = result_de
            except Exception:
                pass
        
        # Extract fitted parameters
        rho_fit, nu_fit = result.x
        
        params = SabrParams(
            sigma_atm=sigma_atm_black,
            beta=self.beta,
            rho=rho_fit,
            nu=nu_fit,
            shift=shift
        )
        
        # Compute individual errors
        vol_errors = {}
        alpha = self.model.alpha_from_sigma_atm(F, T, params)
        
        for K, v_mkt in zip(strikes, vols):
            if vol_type.upper() == "NORMAL":
                v_model = hagan_normal_vol(F, K, T, alpha, self.beta, rho_fit, nu_fit, shift)
            else:
                v_model = hagan_black_vol(F, K, T, alpha, self.beta, rho_fit, nu_fit, shift)
            vol_errors[K] = v_model - v_mkt
        
        return CalibrationResult(
            params=params,
            fit_error=result.fun,
            vol_errors=vol_errors,
            success=result.fun < 1e-6,
            message="Calibration successful" if result.fun < 1e-6 else f"High fit error: {result.fun:.2e}"
        )
    
    def fit_from_vol_quotes(
        self,
        quotes: List[Dict],
        F: float,
        T: float,
        shift: float = 0.0
    ) -> CalibrationResult:
        """
        Fit from list of vol quote dictionaries.
        
        Args:
            quotes: List of dicts with keys 'strike', 'vol', 'vol_type'
            F: Forward rate
            T: Time to expiry
            shift: Shift for negative rates
            
        Returns:
            CalibrationResult
        """
        # Convert to DataFrame
        df = pd.DataFrame(quotes)
        
        # Determine vol_type (assume all same)
        vol_type = quotes[0].get('vol_type', 'NORMAL')
        
        return self.fit(df, F, T, shift, vol_type)
    
    def fit_error(
        self,
        params: SabrParams,
        quotes_df: pd.DataFrame,
        F: float,
        T: float,
        vol_type: str = "NORMAL"
    ) -> float:
        """
        Compute calibration error for given parameters.
        
        Args:
            params: SABR parameters to evaluate
            quotes_df: Market quotes
            F: Forward rate
            T: Time to expiry
            vol_type: "NORMAL" or "LOGNORMAL"
            
        Returns:
            Sum of squared vol differences
        """
        strikes = quotes_df['strike'].values
        vols = quotes_df['vol'].values
        
        alpha = self.model.alpha_from_sigma_atm(F, T, params)
        
        error = 0.0
        for K, v_mkt in zip(strikes, vols):
            if vol_type.upper() == "NORMAL":
                v_model = hagan_normal_vol(F, K, T, alpha, params.beta, params.rho, params.nu, params.shift)
            else:
                v_model = hagan_black_vol(F, K, T, alpha, params.beta, params.rho, params.nu, params.shift)
            error += (v_model - v_mkt) ** 2
        
        return error


def calibrate_sabr_surface(
    quotes_df: pd.DataFrame,
    forwards: Dict[str, float],
    beta: float = 0.5
) -> Dict[str, SabrParams]:
    """
    Calibrate SABR for multiple expiries.
    
    Args:
        quotes_df: DataFrame with columns [expiry, underlying_tenor, strike, vol, vol_type, shift]
        forwards: Dict of {expiry_tenor: forward_rate}
        beta: Fixed beta for all calibrations
        
    Returns:
        Dict of {expiry_tenor: SabrParams}
    """
    from ..dates import DateUtils
    
    results = {}
    calibrator = SabrCalibrator(beta=beta)
    
    # Group by expiry
    for expiry, group in quotes_df.groupby('expiry'):
        F = forwards.get(expiry)
        if F is None:
            continue
        
        T = DateUtils.tenor_to_years(expiry)
        shift = group['shift'].iloc[0] if 'shift' in group.columns else 0.0
        vol_type = group['vol_type'].iloc[0] if 'vol_type' in group.columns else 'NORMAL'
        
        # Prepare quote data
        quote_data = group[['strike', 'vol']].copy()
        
        result = calibrator.fit(quote_data, F, T, shift, vol_type)
        if result.success:
            results[expiry] = result.params
        
    return results


def calibrate_sabr_bucket(
    quotes_bucket: pd.DataFrame,
    beta: float,
    vol_type: str,
    shift: float = 0.0
) -> Tuple[SabrBucketParams, CalibrationResult]:
    """
    Calibrate SABR parameters for a single (expiry, tenor) bucket.

    Args:
        quotes_bucket: Normalized quotes with columns K, sigma_mkt, F0, T
        beta: Fixed beta to use
        vol_type: "NORMAL" or "LOGNORMAL"
        shift: Shift policy (per-bucket)

    Returns:
        (SabrBucketParams, CalibrationResult)
    """
    required_cols = {"K", "sigma_mkt", "F0", "T"}
    missing = required_cols - set(quotes_bucket.columns)
    if missing:
        raise ValueError(f"Missing required columns for calibration: {missing}")

    F0 = float(quotes_bucket["F0"].iloc[0])
    T = float(quotes_bucket["T"].iloc[0])

    calibrator = SabrCalibrator(beta=beta)
    fit_df = quotes_bucket.rename(columns={"K": "strike", "sigma_mkt": "vol"})
    # Preserve optional weights
    columns = [c for c in ["strike", "vol", "weight"] if c in fit_df.columns]
    fit_df = fit_df[columns]

    result = calibrator.fit(fit_df, F=F0, T=T, shift=shift, vol_type=vol_type)

    # Compute diagnostics: RMSE and max abs error in the quoting convention
    errors = np.array(list(result.vol_errors.values()))
    rmse = float(np.sqrt(np.mean(errors ** 2))) if len(errors) else 0.0
    max_abs_error = float(np.max(np.abs(errors))) if len(errors) else 0.0

    diagnostics = {
        "rmse": rmse,
        "max_abs_error": max_abs_error,
        "num_quotes": len(fit_df),
        "fit_error": float(result.fit_error),
        "success": bool(result.success),
        "vol_type": vol_type,
    }

    bucket_params = SabrBucketParams(
        sigma_atm=result.params.sigma_atm,
        nu=result.params.nu,
        rho=result.params.rho,
        beta=result.params.beta,
        shift=result.params.shift,
        diagnostics=diagnostics,
    )

    return bucket_params, result


def build_sabr_surface(
    normalized_quotes: pd.DataFrame,
    curve_state: "CurveState",
    beta_policy: float = 0.5,
    min_quotes_per_bucket: int = 3
) -> SabrSurfaceState:
    """
    Calibrate a bucketed SABR surface from normalized quotes.

    Args:
        normalized_quotes: Output from normalize_vol_quotes
        curve_state: CurveState providing anchor/asof metadata
        beta_policy: Fixed beta to use for all buckets
        min_quotes_per_bucket: Skip buckets with fewer strikes than this

    Returns:
        SabrSurfaceState with diagnostics attached
    """
    if normalized_quotes.empty:
        return SabrSurfaceState(
            params_by_bucket={},
            convention={"beta": beta_policy},
            asof=str(getattr(curve_state.discount_curve, "anchor_date", "")),
        )

    params_by_bucket: Dict[Tuple[str, str], SabrBucketParams] = {}

    for bucket, group in normalized_quotes.groupby("bucket_key"):
        if len(group) < min_quotes_per_bucket:
            continue

        vol_type = str(group.get("vol_type", "NORMAL").iloc[0]).upper()
        shift = float(group.get("shift", 0.0).iloc[0]) if "shift" in group.columns else 0.0

        bucket_params, result = calibrate_sabr_bucket(
            quotes_bucket=group,
            beta=beta_policy,
            vol_type=vol_type,
            shift=shift,
        )
        params_by_bucket[bucket] = bucket_params

    convention = {
        "beta": beta_policy,
        "vol_type": normalized_quotes["vol_type"].iloc[0] if "vol_type" in normalized_quotes.columns else "NORMAL",
        "quote_source": normalized_quotes["source"].iloc[0] if "source" in normalized_quotes.columns else None,
    }

    return SabrSurfaceState(
        params_by_bucket=params_by_bucket,
        convention=convention,
        asof=str(curve_state.discount_curve.anchor_date),
    )
