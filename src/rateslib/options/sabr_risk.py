"""
SABR-aware risk engine for options.

Provides model-consistent Greeks with smile-dynamics correction:
- Delta includes dSigma/dF contribution
- Vega/Vanna/Volga for volatility risk
- Risk decomposition: sideways vs backbone

References:
    - "Managing Smile Risk", Hagan et al., Wilmott Magazine, 2002
    - "SABR and SABR LIBOR Market Model in Practice", Antonov et al., 2015
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np

from ..vol.sabr import SabrParams, SabrModel


@dataclass
class RiskReport:
    """Container for SABR option risk metrics."""
    
    # Basic Greeks
    delta_base: float        # Delta without smile adjustment
    delta_sabr: float        # Model-consistent delta with dSigma/dF
    gamma_base: float        # Base gamma
    vega_atm: float          # Vega to ATM vol shift
    
    # Higher-order vol Greeks
    vanna: float             # dDelta/dVol = dVega/dSpot
    volga: float             # dVega/dVol = Gamma in vol space
    
    # Delta decomposition
    delta_sideways: float    # Delta with fixed smile
    delta_backbone: float    # Contribution from smile dynamics
    
    # Model parameters
    forward: float
    strike: float
    expiry: float
    implied_vol: float
    vol_type: str
    sabr_params: SabrParams
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for reporting."""
        return {
            'delta_base': self.delta_base,
            'delta_sabr': self.delta_sabr,
            'gamma_base': self.gamma_base,
            'vega_atm': self.vega_atm,
            'vanna': self.vanna,
            'volga': self.volga,
            'delta_sideways': self.delta_sideways,
            'delta_backbone': self.delta_backbone,
            'forward': self.forward,
            'strike': self.strike,
            'expiry': self.expiry,
            'implied_vol': self.implied_vol
        }


class SabrOptionRisk:
    """
    Risk engine for options priced under SABR.
    
    Computes model-consistent Greeks that account for smile dynamics.
    
    Model-consistent delta:
        Delta_SABR = Delta_base + Vega * dSigma/dF
        
    where dSigma/dF is the SABR smile slope at the strike.
    
    This correction is essential for proper hedging because:
    - Standard Black/Bachelier delta assumes constant vol
    - SABR vol changes with forward rate (smile dynamics)
    - Ignoring this leads to systematic hedging errors
    """
    
    def __init__(self, vol_type: str = "NORMAL"):
        """
        Initialize risk engine.
        
        Args:
            vol_type: "NORMAL" or "LOGNORMAL" for base model
        """
        self.vol_type = vol_type.upper()
        self.model = SabrModel()
    
    def risk_report(
        self,
        F: float,
        K: float,
        T: float,
        sabr_params: SabrParams,
        annuity: float = 1.0,
        is_call: bool = True,
        notional: float = 1.0
    ) -> RiskReport:
        """
        Generate comprehensive risk report.
        
        Args:
            F: Forward rate
            K: Strike
            T: Time to expiry (years)
            sabr_params: SABR model parameters
            annuity: Discount factor / annuity
            is_call: True for call/payer, False for put/receiver
            notional: Notional amount
            
        Returns:
            RiskReport with all risk metrics
        """
        from .base_models import bachelier_greeks, black76_greeks
        
        # Get SABR implied vol at strike
        if self.vol_type == "NORMAL":
            sigma = self.model.implied_vol_normal(F, K, T, sabr_params)
        else:
            sigma = self.model.implied_vol_black(F, K, T, sabr_params)
        
        # Base Greeks from Black/Bachelier
        if self.vol_type == "NORMAL":
            base_greeks = bachelier_greeks(F, K, T, sigma, annuity, is_call)
        else:
            base_greeks = black76_greeks(F, K, T, sigma, annuity, is_call)
        
        delta_base = base_greeks['delta']
        gamma_base = base_greeks['gamma']
        vega_base = base_greeks['vega']
        
        # SABR vol sensitivities
        dsigma_dF = self.model.dsigma_dF(F, K, T, sabr_params, vol_type=self.vol_type)
        dsigma_drho = self.model.dsigma_drho(F, K, T, sabr_params, vol_type=self.vol_type)
        dsigma_dnu = self.model.dsigma_dnu(F, K, T, sabr_params, vol_type=self.vol_type)
        
        # Model-consistent delta
        # Delta_SABR = Delta_base + Vega * dSigma/dF
        delta_sabr = delta_base + vega_base * dsigma_dF
        
        # Vanna: dDelta/dVol = dVega/dSpot
        # Approximate via dSigma_drho (correlation is vol-of-vol mixing)
        vanna = vega_base * dsigma_drho / (sabr_params.nu + 1e-10)
        
        # Volga: d^2V/dSigma^2
        # For Bachelier/Black, Volga = Vega * d2V/dSigma2
        # Approximate using numerical differentiation
        vol_bump = 0.0001  # 1bp bump
        if self.vol_type == "NORMAL":
            greeks_up = bachelier_greeks(F, K, T, sigma + vol_bump, annuity, is_call)
            greeks_dn = bachelier_greeks(F, K, T, sigma - vol_bump, annuity, is_call)
        else:
            greeks_up = black76_greeks(F, K, T, sigma + vol_bump, annuity, is_call)
            greeks_dn = black76_greeks(F, K, T, sigma - vol_bump, annuity, is_call)
        
        volga = (greeks_up['vega'] - greeks_dn['vega']) / (2 * vol_bump)
        
        # Vega ATM (parallel shift in ATM vol)
        # This is vega to sigma_ATM via chain rule: dV/d(sigma_ATM) = dV/d(sigma) * d(sigma)/d(sigma_ATM)
        # At ATM, dsigma/dsigma_ATM ≈ 1
        F_atm = F
        if self.vol_type == "NORMAL":
            sigma_atm = self.model.implied_vol_normal(F, F_atm, T, sabr_params)
            atm_greeks = bachelier_greeks(F, F_atm, T, sigma_atm, annuity, is_call)
        else:
            sigma_atm = self.model.implied_vol_black(F, F_atm, T, sabr_params)
            atm_greeks = black76_greeks(F, F_atm, T, sigma_atm, annuity, is_call)
        
        vega_atm = atm_greeks['vega']
        
        # Delta decomposition: sideways vs backbone
        # Sideways delta: move spot, keep smile fixed relative to spot
        # Backbone delta: contribution from smile moving with spot
        delta_sideways = delta_base
        delta_backbone = delta_sabr - delta_base
        
        return RiskReport(
            delta_base=notional * delta_base,
            delta_sabr=notional * delta_sabr,
            gamma_base=notional * gamma_base,
            vega_atm=notional * vega_atm,
            vanna=notional * vanna,
            volga=notional * volga,
            delta_sideways=notional * delta_sideways,
            delta_backbone=notional * delta_backbone,
            forward=F,
            strike=K,
            expiry=T,
            implied_vol=sigma,
            vol_type=self.vol_type,
            sabr_params=sabr_params
        )
    
    def delta_decomposition(
        self,
        F: float,
        K: float,
        T: float,
        sabr_params: SabrParams,
        annuity: float = 1.0,
        is_call: bool = True
    ) -> Dict[str, float]:
        """
        Decompose delta into sideways and backbone components.
        
        Sideways delta:
            - Move forward, hold smile grid fixed
            - Standard Black/Bachelier delta
            
        Backbone delta:
            - Contribution from smile moving with spot
            - Vega * dSigma/dF
            
        This decomposition matters for:
            - Understanding hedging behavior
            - Risk attribution
            - Stress testing under different scenarios
        
        Args:
            F: Forward rate
            K: Strike
            T: Time to expiry
            sabr_params: SABR parameters
            annuity: Discount factor
            is_call: True for call, False for put
            
        Returns:
            Dict with 'sideways', 'backbone', 'total'
        """
        report = self.risk_report(F, K, T, sabr_params, annuity, is_call, notional=1.0)
        
        return {
            'sideways': report.delta_sideways,
            'backbone': report.delta_backbone,
            'total': report.delta_sabr,
            'backbone_pct': report.delta_backbone / (abs(report.delta_sabr) + 1e-10) * 100
        }
    
    def smile_risk_ladder(
        self,
        F: float,
        strikes: List[float],
        T: float,
        sabr_params: SabrParams,
        annuity: float = 1.0,
        is_call: bool = True,
        notional: float = 1.0
    ) -> List[Dict[str, float]]:
        """
        Compute risk ladder across strikes.
        
        Args:
            F: Forward rate
            strikes: List of strikes
            T: Time to expiry
            sabr_params: SABR parameters
            annuity: Discount factor
            is_call: Option type
            notional: Notional amount
            
        Returns:
            List of risk dicts per strike
        """
        ladder = []
        
        for K in strikes:
            report = self.risk_report(F, K, T, sabr_params, annuity, is_call, notional)
            
            # Moneyness
            if self.vol_type == "NORMAL":
                moneyness = (K - F) * 10000  # in bps
            else:
                moneyness = np.log(K / F)
            
            ladder.append({
                'strike': K,
                'moneyness': moneyness,
                'implied_vol': report.implied_vol,
                'delta_base': report.delta_base,
                'delta_sabr': report.delta_sabr,
                'gamma': report.gamma_base,
                'vega': report.vega_atm,
                'vanna': report.vanna,
                'volga': report.volga
            })
        
        return ladder
    
    def parameter_sensitivities(
        self,
        F: float,
        K: float,
        T: float,
        sabr_params: SabrParams,
        annuity: float = 1.0,
        is_call: bool = True,
        notional: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute sensitivities to SABR parameters.
        
        Args:
            F: Forward rate
            K: Strike
            T: Time to expiry
            sabr_params: SABR parameters
            annuity: Discount factor
            is_call: Option type
            notional: Notional amount
            
        Returns:
            Dict with sensitivities to rho, nu, sigma_atm
        """
        from .base_models import bachelier_greeks, black76_greeks
        
        # Get base vol and vega
        if self.vol_type == "NORMAL":
            sigma = self.model.implied_vol_normal(F, K, T, sabr_params)
            base_greeks = bachelier_greeks(F, K, T, sigma, annuity, is_call)
        else:
            sigma = self.model.implied_vol_black(F, K, T, sabr_params)
            base_greeks = black76_greeks(F, K, T, sigma, annuity, is_call)
        
        vega = base_greeks['vega']
        
        # Vol sensitivities to SABR params
        dsigma_drho = self.model.dsigma_drho(F, K, T, sabr_params, vol_type=self.vol_type)
        dsigma_dnu = self.model.dsigma_dnu(F, K, T, sabr_params, vol_type=self.vol_type)
        
        # dV/d(sigma_atm) ≈ Vega at ATM (for ATM, d_sigma/d_sigma_atm ≈ 1)
        # For off-ATM, need chain rule through alpha
        dalpha_dsigma = self.model.dalpha_dsigma_atm(F, T, sabr_params)
        
        return {
            'dV_drho': notional * vega * dsigma_drho,
            'dV_dnu': notional * vega * dsigma_dnu,
            'dV_dsigma_atm': notional * vega * dalpha_dsigma,
            'dsigma_drho': dsigma_drho,
            'dsigma_dnu': dsigma_dnu,
            'vega': notional * vega
        }


def compute_portfolio_risk(
    positions: List[Dict],
    sabr_params_by_expiry: Dict[str, SabrParams],
    vol_type: str = "NORMAL"
) -> Dict[str, float]:
    """
    Compute aggregate risk for a portfolio of options.
    
    Args:
        positions: List of position dicts with keys:
            - expiry_tenor: str
            - strike: float
            - forward: float
            - annuity: float
            - notional: float
            - is_call: bool
        sabr_params_by_expiry: SABR params keyed by expiry tenor
        vol_type: "NORMAL" or "LOGNORMAL"
        
    Returns:
        Aggregated risk metrics
    """
    from ..dates import DateUtils
    
    engine = SabrOptionRisk(vol_type=vol_type)
    
    total_delta_base = 0.0
    total_delta_sabr = 0.0
    total_gamma = 0.0
    total_vega = 0.0
    total_vanna = 0.0
    total_volga = 0.0
    
    for pos in positions:
        expiry = DateUtils.tenor_to_years(pos['expiry_tenor'])
        sabr_params = sabr_params_by_expiry.get(pos['expiry_tenor'])
        
        if sabr_params is None:
            continue
        
        report = engine.risk_report(
            F=pos['forward'],
            K=pos['strike'],
            T=expiry,
            sabr_params=sabr_params,
            annuity=pos['annuity'],
            is_call=pos['is_call'],
            notional=pos['notional']
        )
        
        total_delta_base += report.delta_base
        total_delta_sabr += report.delta_sabr
        total_gamma += report.gamma_base
        total_vega += report.vega_atm
        total_vanna += report.vanna
        total_volga += report.volga
    
    return {
        'delta_base': total_delta_base,
        'delta_sabr': total_delta_sabr,
        'gamma': total_gamma,
        'vega': total_vega,
        'vanna': total_vanna,
        'volga': total_volga,
        'delta_adjustment': total_delta_sabr - total_delta_base
    }
