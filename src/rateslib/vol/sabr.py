"""
SABR stochastic volatility model.

Implements the SABR model for rates volatility:
- Hagan et al. implied volatility approximation
- Shifted SABR for negative rates
- Alpha inversion from ATM vol
- Implicit differentiation for risk

References:
- Hagan, P.S. et al. (2002). "Managing Smile Risk." Wilmott Magazine.
- Bartlett, B. (2006). "Hedging Under SABR Model."
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.optimize import brentq


@dataclass
class SabrParams:
    """
    SABR model parameters.
    
    Uses the alternative parameterization with sigma_atm instead of alpha,
    which is more stable for production use and aligns with desk conventions.
    
    Attributes:
        sigma_atm: ATM implied volatility (Black or normal, depending on context)
        beta: CEV exponent (0 = normal, 1 = lognormal, typically fixed)
        rho: Correlation between forward and vol (-1 < rho < 1)
        nu: Volatility of volatility (vol-of-vol)
        shift: Shift parameter for negative rates (default 0)
    """
    sigma_atm: float
    beta: float
    rho: float
    nu: float
    shift: float = 0.0
    
    def __post_init__(self):
        """Validate parameters."""
        if not -1 < self.rho < 1:
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        if self.nu < 0:
            raise ValueError(f"nu must be non-negative, got {self.nu}")
        if not 0 <= self.beta <= 1:
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")
        if self.sigma_atm <= 0:
            raise ValueError(f"sigma_atm must be positive, got {self.sigma_atm}")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "sigma_atm": self.sigma_atm,
            "beta": self.beta,
            "rho": self.rho,
            "nu": self.nu,
            "shift": self.shift
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "SabrParams":
        """Create from dictionary."""
        return cls(
            sigma_atm=d["sigma_atm"],
            beta=d["beta"],
            rho=d["rho"],
            nu=d["nu"],
            shift=d.get("shift", 0.0)
        )


def hagan_black_vol(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0
) -> float:
    """
    Hagan et al. approximation for SABR Black implied volatility.
    
    Args:
        F: Forward rate
        K: Strike
        T: Time to expiry (years)
        alpha: SABR alpha (instantaneous vol)
        beta: CEV exponent
        rho: Correlation
        nu: Vol of vol
        shift: Shift for negative rates
        
    Returns:
        Black implied volatility
    """
    # Apply shift for negative rates
    F_s = F + shift
    K_s = K + shift
    
    if F_s <= 0 or K_s <= 0:
        raise ValueError(f"Shifted forward ({F_s}) and strike ({K_s}) must be positive")
    
    # Handle ATM case
    if abs(F_s - K_s) < 1e-10:
        return _hagan_atm_vol(F_s, T, alpha, beta, rho, nu)
    
    # General case
    log_fk = np.log(F_s / K_s)
    fk_mid = (F_s * K_s) ** ((1 - beta) / 2)
    
    # Denominator terms
    one_minus_beta = 1 - beta
    denom1 = fk_mid * (1 + one_minus_beta**2 / 24 * log_fk**2 
                       + one_minus_beta**4 / 1920 * log_fk**4)
    
    # z and x(z) terms
    z = nu / alpha * fk_mid * log_fk
    
    if abs(z) < 1e-10:
        x_z = 1.0
    else:
        sqrt_term = np.sqrt(1 - 2 * rho * z + z**2)
        x_z = z / np.log((sqrt_term + z - rho) / (1 - rho))
    
    # Time correction terms
    fk_beta = (F_s * K_s) ** ((1 - beta) / 2)
    term1 = one_minus_beta**2 * alpha**2 / (24 * fk_beta**2)
    term2 = rho * beta * nu * alpha / (4 * fk_beta)
    term3 = (2 - 3 * rho**2) * nu**2 / 24
    
    time_adj = 1 + (term1 + term2 + term3) * T
    
    sigma_b = alpha / denom1 * x_z * time_adj
    
    return sigma_b


def _hagan_atm_vol(
    F: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float
) -> float:
    """ATM Black vol from Hagan formula."""
    F_beta = F ** (1 - beta)
    
    term1 = (1 - beta)**2 * alpha**2 / (24 * F**(2 - 2*beta))
    term2 = rho * beta * nu * alpha / (4 * F**(1 - beta))
    term3 = (2 - 3 * rho**2) * nu**2 / 24
    
    sigma_atm = alpha / F_beta * (1 + (term1 + term2 + term3) * T)
    
    return sigma_atm


def hagan_normal_vol(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0
) -> float:
    """
    SABR normal (Bachelier) implied volatility.
    
    Converts Black vol to normal vol using the approximation:
    sigma_N ≈ sigma_B * F for ATM, with adjustments for OTM.
    
    Args:
        F: Forward rate
        K: Strike
        T: Time to expiry
        alpha, beta, rho, nu: SABR parameters
        shift: Shift for negative rates
        
    Returns:
        Normal (Bachelier) implied volatility
    """
    F_s = F + shift
    K_s = K + shift
    
    # Get Black vol
    sigma_b = hagan_black_vol(F, K, T, alpha, beta, rho, nu, shift)
    
    # Convert to normal vol
    # At ATM: sigma_N ≈ sigma_B * F
    # General: sigma_N ≈ sigma_B * (F*K)^0.5 * some adjustment
    if abs(F_s - K_s) < 1e-10:
        sigma_n = sigma_b * F_s
    else:
        # Use the relationship between Black and Normal vols
        log_fk = np.log(F_s / K_s)
        fk_sqrt = np.sqrt(F_s * K_s)
        sigma_n = sigma_b * fk_sqrt * (1 - log_fk**2 / 24)
    
    return sigma_n


class SabrModel:
    """
    SABR stochastic volatility model.
    
    Implements:
    - Implied volatility calculation (Black and normal)
    - Alpha inversion from ATM vol
    - Derivatives for risk computation
    - Shifted SABR for negative rates
    """
    
    def __init__(self):
        """Initialize SABR model."""
        self._alpha_cache = {}
    
    def implied_vol_black(
        self,
        F: float,
        K: float,
        T: float,
        params: SabrParams
    ) -> float:
        """
        Compute Black'76 implied vol using Hagan approximation.
        
        Args:
            F: Forward rate
            K: Strike
            T: Time to expiry
            params: SABR parameters (with sigma_atm)
            
        Returns:
            Black implied volatility
        """
        alpha = self.alpha_from_sigma_atm(F, T, params)
        return hagan_black_vol(F, K, T, alpha, params.beta, params.rho, params.nu, params.shift)
    
    def implied_vol_normal(
        self,
        F: float,
        K: float,
        T: float,
        params: SabrParams
    ) -> float:
        """
        Compute Bachelier (normal) implied vol.
        
        Args:
            F: Forward rate
            K: Strike
            T: Time to expiry
            params: SABR parameters
            
        Returns:
            Normal implied volatility
        """
        alpha = self.alpha_from_sigma_atm(F, T, params)
        return hagan_normal_vol(F, K, T, alpha, params.beta, params.rho, params.nu, params.shift)
    
    def alpha_from_sigma_atm(
        self,
        F: float,
        T: float,
        params: SabrParams
    ) -> float:
        """
        Invert ATM formula to get alpha from sigma_atm.
        
        Solves: G(alpha) = sigma_atm where G is the ATM vol formula.
        
        Args:
            F: Forward rate
            T: Time to expiry
            params: SABR parameters with sigma_atm
            
        Returns:
            Alpha parameter
        """
        F_s = F + params.shift
        sigma_atm = params.sigma_atm
        beta = params.beta
        rho = params.rho
        nu = params.nu
        
        # Define the equation to solve: ATM_vol(alpha) - sigma_atm = 0
        def objective(alpha):
            if alpha <= 0:
                return float('inf')
            return _hagan_atm_vol(F_s, T, alpha, beta, rho, nu) - sigma_atm
        
        # Initial guess: alpha ≈ sigma_atm * F^(1-beta)
        alpha_init = sigma_atm * F_s ** (1 - beta)
        
        # Bracket search
        alpha_low = alpha_init * 0.01
        alpha_high = alpha_init * 10.0
        
        try:
            # Try to find brackets
            f_low = objective(alpha_low)
            f_high = objective(alpha_high)
            
            if f_low * f_high > 0:
                # Adjust brackets
                if f_low > 0:
                    alpha_low = alpha_init * 0.001
                else:
                    alpha_high = alpha_init * 100.0
            
            alpha = brentq(objective, alpha_low, alpha_high, xtol=1e-12)
        except (ValueError, RuntimeError):
            # Fallback to simple approximation
            alpha = alpha_init
        
        return alpha
    
    def dalpha_dsigma_atm(
        self,
        F: float,
        T: float,
        params: SabrParams
    ) -> float:
        """
        Derivative of alpha w.r.t. sigma_atm.
        
        Using implicit differentiation:
        d_alpha/d_sigma_atm = 1 / (dG/d_alpha)
        
        Args:
            F: Forward rate
            T: Time to expiry
            params: SABR parameters
            
        Returns:
            d_alpha/d_sigma_atm
        """
        alpha = self.alpha_from_sigma_atm(F, T, params)
        F_s = F + params.shift
        
        # Compute dG/d_alpha numerically
        eps = alpha * 1e-6
        vol_up = _hagan_atm_vol(F_s, T, alpha + eps, params.beta, params.rho, params.nu)
        vol_down = _hagan_atm_vol(F_s, T, alpha - eps, params.beta, params.rho, params.nu)
        
        dG_dalpha = (vol_up - vol_down) / (2 * eps)
        
        if abs(dG_dalpha) < 1e-15:
            return 0.0
        
        return 1.0 / dG_dalpha
    
    def dalpha_dtheta(
        self,
        F: float,
        T: float,
        params: SabrParams,
        theta_name: str
    ) -> float:
        """
        Derivative of alpha w.r.t. other parameters.
        
        Using implicit differentiation:
        d_alpha/d_theta = -(dG/d_theta) / (dG/d_alpha)
        
        Args:
            F: Forward rate
            T: Time to expiry
            params: SABR parameters
            theta_name: 'beta', 'rho', 'nu', or 'F'
            
        Returns:
            d_alpha/d_theta
        """
        alpha = self.alpha_from_sigma_atm(F, T, params)
        F_s = F + params.shift
        eps = 1e-6
        
        # Compute dG/d_alpha
        vol_up = _hagan_atm_vol(F_s, T, alpha + eps * alpha, params.beta, params.rho, params.nu)
        vol_down = _hagan_atm_vol(F_s, T, alpha - eps * alpha, params.beta, params.rho, params.nu)
        dG_dalpha = (vol_up - vol_down) / (2 * eps * alpha)
        
        if abs(dG_dalpha) < 1e-15:
            return 0.0
        
        # Compute dG/d_theta
        if theta_name == 'beta':
            eps_t = 0.01
            vol_up = _hagan_atm_vol(F_s, T, alpha, min(params.beta + eps_t, 1.0), params.rho, params.nu)
            vol_down = _hagan_atm_vol(F_s, T, alpha, max(params.beta - eps_t, 0.0), params.rho, params.nu)
            dG_dtheta = (vol_up - vol_down) / (2 * eps_t)
        elif theta_name == 'rho':
            eps_t = 0.01
            vol_up = _hagan_atm_vol(F_s, T, alpha, params.beta, min(params.rho + eps_t, 0.99), params.nu)
            vol_down = _hagan_atm_vol(F_s, T, alpha, params.beta, max(params.rho - eps_t, -0.99), params.nu)
            dG_dtheta = (vol_up - vol_down) / (2 * eps_t)
        elif theta_name == 'nu':
            eps_t = params.nu * 0.01 if params.nu > 0 else 0.01
            vol_up = _hagan_atm_vol(F_s, T, alpha, params.beta, params.rho, params.nu + eps_t)
            vol_down = _hagan_atm_vol(F_s, T, alpha, params.beta, params.rho, max(params.nu - eps_t, 0))
            dG_dtheta = (vol_up - vol_down) / (2 * eps_t)
        elif theta_name == 'F':
            eps_t = F_s * 1e-4
            vol_up = _hagan_atm_vol(F_s + eps_t, T, alpha, params.beta, params.rho, params.nu)
            vol_down = _hagan_atm_vol(F_s - eps_t, T, alpha, params.beta, params.rho, params.nu)
            dG_dtheta = (vol_up - vol_down) / (2 * eps_t)
        else:
            raise ValueError(f"Unknown parameter: {theta_name}")
        
        return -dG_dtheta / dG_dalpha
    
    def dsigma_dF(
        self,
        F: float,
        K: float,
        T: float,
        params: SabrParams,
        hold_atm_fixed: bool = True
    ) -> float:
        """
        Derivative of implied vol w.r.t. forward.
        
        Decomposes into sideways and backbone components:
        d_sigma/dF = d_sigma/dF|_sigma_atm + d_sigma/d_sigma_atm * d_sigma_atm/dF
        
        If hold_atm_fixed=True, only returns sideways (first term).
        
        Args:
            F: Forward rate
            K: Strike
            T: Time to expiry
            params: SABR parameters
            hold_atm_fixed: If True, backbone term is zero
            
        Returns:
            d_sigma/dF
        """
        alpha = self.alpha_from_sigma_atm(F, T, params)
        F_s = F + params.shift
        eps = F_s * 1e-5
        
        if hold_atm_fixed:
            # Sideways only: bump F, keep alpha fixed (sigma_atm fixed)
            vol_up = hagan_black_vol(F + eps, K, T, alpha, params.beta, params.rho, params.nu, params.shift)
            vol_down = hagan_black_vol(F - eps, K, T, alpha, params.beta, params.rho, params.nu, params.shift)
            return (vol_up - vol_down) / (2 * eps)
        else:
            # Full derivative including backbone
            # Need to recalibrate alpha when F changes
            vol_base = self.implied_vol_black(F, K, T, params)
            vol_up = self.implied_vol_black(F + eps, K, T, params)
            vol_down = self.implied_vol_black(F - eps, K, T, params)
            return (vol_up - vol_down) / (2 * eps)
    
    def dsigma_drho(
        self,
        F: float,
        K: float,
        T: float,
        params: SabrParams,
        vol_type: str = "BLACK"
    ) -> float:
        """
        Derivative of implied vol w.r.t. rho (for vanna).
        
        Args:
            F: Forward rate
            K: Strike
            T: Time to expiry
            params: SABR parameters
            vol_type: "BLACK" or "NORMAL"
            
        Returns:
            d_sigma/d_rho
        """
        alpha = self.alpha_from_sigma_atm(F, T, params)
        eps = 0.01
        
        rho_up = min(params.rho + eps, 0.99)
        rho_down = max(params.rho - eps, -0.99)
        
        vol_type_upper = vol_type.upper()

        if vol_type_upper == "NORMAL":
            vol_fn = hagan_normal_vol
        else:
            vol_fn = hagan_black_vol

        vol_up = vol_fn(F, K, T, alpha, params.beta, rho_up, params.nu, params.shift)
        vol_down = vol_fn(F, K, T, alpha, params.beta, rho_down, params.nu, params.shift)
        
        # Also need to account for alpha change due to rho change
        dalpha_drho = self.dalpha_dtheta(F, T, params, 'rho')
        
        # Chain rule
        alpha_up = alpha + dalpha_drho * eps
        alpha_down = alpha - dalpha_drho * eps
        
        vol_up_full = vol_fn(F, K, T, alpha_up, params.beta, rho_up, params.nu, params.shift)
        vol_down_full = vol_fn(F, K, T, alpha_down, params.beta, rho_down, params.nu, params.shift)
        
        return (vol_up_full - vol_down_full) / (rho_up - rho_down)
    
    def dsigma_dnu(
        self,
        F: float,
        K: float,
        T: float,
        params: SabrParams,
        vol_type: str = "BLACK"
    ) -> float:
        """
        Derivative of implied vol w.r.t. nu (for volga).
        
        Args:
            F: Forward rate
            K: Strike
            T: Time to expiry
            params: SABR parameters
            vol_type: "BLACK" or "NORMAL"
            
        Returns:
            d_sigma/d_nu
        """
        alpha = self.alpha_from_sigma_atm(F, T, params)
        eps = max(params.nu * 0.01, 0.001)
        
        nu_up = params.nu + eps
        nu_down = max(params.nu - eps, 0.001)
        
        # Account for alpha change due to nu change
        dalpha_dnu = self.dalpha_dtheta(F, T, params, 'nu')
        
        alpha_up = alpha + dalpha_dnu * eps
        alpha_down = alpha - dalpha_dnu * eps

        vol_type_upper = vol_type.upper()
        if vol_type_upper == "NORMAL":
            vol_fn = hagan_normal_vol
        else:
            vol_fn = hagan_black_vol

        vol_up = vol_fn(F, K, T, alpha_up, params.beta, params.rho, nu_up, params.shift)
        vol_down = vol_fn(F, K, T, alpha_down, params.beta, params.rho, nu_down, params.shift)
        
        return (vol_up - vol_down) / (nu_up - nu_down)
    
    def smile_at_strikes(
        self,
        F: float,
        strikes: list,
        T: float,
        params: SabrParams,
        vol_type: str = "BLACK"
    ) -> dict:
        """
        Compute implied vol smile across strikes.
        
        Args:
            F: Forward rate
            strikes: List of strike values
            T: Time to expiry
            params: SABR parameters
            vol_type: "BLACK" or "NORMAL"
            
        Returns:
            Dict of {strike: implied_vol}
        """
        result = {}
        for K in strikes:
            if vol_type.upper() == "BLACK":
                result[K] = self.implied_vol_black(F, K, T, params)
            else:
                result[K] = self.implied_vol_normal(F, K, T, params)
        return result
