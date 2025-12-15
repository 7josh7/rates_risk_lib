"""
Swaption pricing engine.

A swaption is an option to enter into an interest rate swap.
- Payer swaption: right to pay fixed, receive floating
- Receiver swaption: right to receive fixed, pay floating

Pricing:
    V_swaption = Annuity * BaseModel(S, K, T, sigma)
    
where:
    - Annuity = sum of discounted accrual fractions (PV01)
    - S = forward swap rate
    - K = strike rate
    - T = option expiry
    - sigma = implied volatility (from SABR or market)
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Tuple, List

import numpy as np

from ..curves.curve import Curve
from ..conventions import DayCount, year_fraction
from ..dates import DateUtils
from .base_models import (
    bachelier_call, bachelier_put,
    black76_call, black76_put,
    shifted_black_call, shifted_black_put,
    bachelier_greeks, black76_greeks
)


@dataclass
class SwaptionResult:
    """Result from swaption pricing."""
    price: float
    forward_swap_rate: float
    strike: float
    expiry: float
    annuity: float
    implied_vol: float
    vol_type: str
    notional: float
    payer_receiver: str


class SwaptionPricer:
    """
    Swaption pricing engine.
    
    Prices European swaptions using:
    - Bachelier (normal) model for normal vol quotes
    - Black'76 model for lognormal vol quotes
    - Shifted Black for negative rate environments
    
    Forward swap rates are derived from the projection curve.
    Annuity uses the discount curve.
    """
    
    def __init__(
        self,
        discount_curve: Curve,
        projection_curve: Optional[Curve] = None,
        fixed_freq: int = 2,
        float_freq: int = 4
    ):
        """
        Initialize swaption pricer.
        
        Args:
            discount_curve: OIS curve for discounting
            projection_curve: Curve for forward rates (defaults to discount)
            fixed_freq: Fixed leg frequency (2 = semi-annual)
            float_freq: Float leg frequency (4 = quarterly)
        """
        self.discount_curve = discount_curve
        self.projection_curve = projection_curve or discount_curve
        self.fixed_freq = fixed_freq
        self.float_freq = float_freq
    
    def forward_swap_rate(
        self,
        expiry: float,
        tenor: float
    ) -> Tuple[float, float]:
        """
        Compute forward swap rate and annuity.
        
        Args:
            expiry: Time to option expiry (years)
            tenor: Swap tenor from expiry (years)
            
        Returns:
            Tuple of (forward_swap_rate, annuity)
        """
        # Generate fixed leg payment times
        months_per_period = 12 // self.fixed_freq
        n_periods = int(tenor * self.fixed_freq)
        
        # Payment times from expiry
        payment_times = []
        for i in range(1, n_periods + 1):
            t = expiry + i / self.fixed_freq
            payment_times.append(t)
        
        # Compute annuity = sum(delta_i * DF(T_i))
        annuity = 0.0
        for t in payment_times:
            delta_i = 1.0 / self.fixed_freq  # Simplified day count
            df = self.discount_curve.discount_factor(t)
            annuity += delta_i * df
        
        # Forward swap rate = (DF(expiry) - DF(maturity)) / Annuity
        df_start = self.projection_curve.discount_factor(expiry)
        df_end = self.projection_curve.discount_factor(expiry + tenor)
        
        if annuity > 0:
            swap_rate = (df_start - df_end) / annuity
        else:
            swap_rate = 0.0
        
        return swap_rate, annuity
    
    def price(
        self,
        S: float,
        K: float,
        T: float,
        annuity: float,
        vol: float,
        vol_type: str = "NORMAL",
        payer_receiver: str = "PAYER",
        notional: float = 1.0,
        shift: float = 0.0
    ) -> float:
        """
        Price a swaption.
        
        Args:
            S: Forward swap rate
            K: Strike rate
            T: Time to expiry (years)
            annuity: Swap annuity (PV01)
            vol: Implied volatility
            vol_type: "NORMAL" or "LOGNORMAL"
            payer_receiver: "PAYER" or "RECEIVER"
            notional: Notional amount
            shift: Shift for shifted lognormal
            
        Returns:
            Swaption price
        """
        vol_type = vol_type.upper()
        is_payer = payer_receiver.upper() == "PAYER"
        
        if vol_type == "NORMAL":
            if is_payer:
                base_price = bachelier_call(S, K, T, vol, annuity)
            else:
                base_price = bachelier_put(S, K, T, vol, annuity)
        elif vol_type == "LOGNORMAL":
            if shift > 0:
                if is_payer:
                    base_price = shifted_black_call(S, K, T, vol, shift, annuity)
                else:
                    base_price = shifted_black_put(S, K, T, vol, shift, annuity)
            else:
                if is_payer:
                    base_price = black76_call(S, K, T, vol, annuity)
                else:
                    base_price = black76_put(S, K, T, vol, annuity)
        else:
            raise ValueError(f"Unknown vol_type: {vol_type}")
        
        return notional * base_price
    
    def greeks(
        self,
        S: float,
        K: float,
        T: float,
        annuity: float,
        vol: float,
        vol_type: str = "NORMAL",
        payer_receiver: str = "PAYER",
        notional: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute swaption Greeks.
        
        Args:
            S: Forward swap rate
            K: Strike rate
            T: Time to expiry
            annuity: Swap annuity
            vol: Implied volatility
            vol_type: "NORMAL" or "LOGNORMAL"
            payer_receiver: "PAYER" or "RECEIVER"
            notional: Notional amount
            
        Returns:
            Dict with delta, gamma, vega, theta
        """
        vol_type = vol_type.upper()
        is_payer = payer_receiver.upper() == "PAYER"
        
        if vol_type == "NORMAL":
            base_greeks = bachelier_greeks(S, K, T, vol, annuity, is_payer)
        else:
            base_greeks = black76_greeks(S, K, T, vol, annuity, is_payer)
        
        return {
            'delta': notional * base_greeks['delta'],
            'gamma': notional * base_greeks['gamma'],
            'vega': notional * base_greeks['vega'],
            'theta': notional * base_greeks['theta']
        }
    
    def price_from_tenor(
        self,
        expiry_tenor: str,
        swap_tenor: str,
        K: float,
        vol: float,
        vol_type: str = "NORMAL",
        payer_receiver: str = "PAYER",
        notional: float = 1.0,
        shift: float = 0.0
    ) -> SwaptionResult:
        """
        Price swaption from tenor strings.
        
        Args:
            expiry_tenor: Option expiry tenor (e.g., "1Y")
            swap_tenor: Underlying swap tenor (e.g., "5Y")
            K: Strike rate
            vol: Implied volatility
            vol_type: "NORMAL" or "LOGNORMAL"
            payer_receiver: "PAYER" or "RECEIVER"
            notional: Notional amount
            shift: Shift for negative rates
            
        Returns:
            SwaptionResult with price and details
        """
        T_expiry = DateUtils.tenor_to_years(expiry_tenor)
        T_swap = DateUtils.tenor_to_years(swap_tenor)
        
        # Get forward swap rate and annuity
        S, annuity = self.forward_swap_rate(T_expiry, T_swap)
        
        # Price
        price = self.price(
            S=S, K=K, T=T_expiry, annuity=annuity, vol=vol,
            vol_type=vol_type, payer_receiver=payer_receiver,
            notional=notional, shift=shift
        )
        
        return SwaptionResult(
            price=price,
            forward_swap_rate=S,
            strike=K,
            expiry=T_expiry,
            annuity=annuity,
            implied_vol=vol,
            vol_type=vol_type,
            notional=notional,
            payer_receiver=payer_receiver
        )
    
    def price_with_sabr(
        self,
        expiry_tenor: str,
        swap_tenor: str,
        K: float,
        sabr_params,  # SabrParams
        vol_type: str = "NORMAL",
        payer_receiver: str = "PAYER",
        notional: float = 1.0
    ) -> SwaptionResult:
        """
        Price swaption using SABR implied vol.
        
        Args:
            expiry_tenor: Option expiry tenor
            swap_tenor: Underlying swap tenor
            K: Strike rate
            sabr_params: SabrParams object
            vol_type: "NORMAL" or "LOGNORMAL"
            payer_receiver: "PAYER" or "RECEIVER"
            notional: Notional amount
            
        Returns:
            SwaptionResult with price and details
        """
        from ..vol.sabr import SabrModel
        
        T_expiry = DateUtils.tenor_to_years(expiry_tenor)
        T_swap = DateUtils.tenor_to_years(swap_tenor)
        
        # Get forward swap rate and annuity
        S, annuity = self.forward_swap_rate(T_expiry, T_swap)
        
        # Get SABR implied vol
        model = SabrModel()
        if vol_type.upper() == "NORMAL":
            vol = model.implied_vol_normal(S, K, T_expiry, sabr_params)
        else:
            vol = model.implied_vol_black(S, K, T_expiry, sabr_params)
        
        # Price
        price = self.price(
            S=S, K=K, T=T_expiry, annuity=annuity, vol=vol,
            vol_type=vol_type, payer_receiver=payer_receiver,
            notional=notional, shift=sabr_params.shift
        )
        
        return SwaptionResult(
            price=price,
            forward_swap_rate=S,
            strike=K,
            expiry=T_expiry,
            annuity=annuity,
            implied_vol=vol,
            vol_type=vol_type,
            notional=notional,
            payer_receiver=payer_receiver
        )
    
    def par_vol(
        self,
        expiry_tenor: str,
        swap_tenor: str,
        market_price: float,
        vol_type: str = "NORMAL",
        payer_receiver: str = "PAYER",
        notional: float = 1.0
    ) -> float:
        """
        Imply volatility from market price.
        
        Args:
            expiry_tenor: Option expiry tenor
            swap_tenor: Underlying swap tenor
            market_price: Market swaption price
            vol_type: "NORMAL" or "LOGNORMAL"
            payer_receiver: "PAYER" or "RECEIVER"
            notional: Notional amount
            
        Returns:
            Implied volatility
        """
        from .base_models import implied_vol_bachelier, implied_vol_black
        
        T_expiry = DateUtils.tenor_to_years(expiry_tenor)
        T_swap = DateUtils.tenor_to_years(swap_tenor)
        
        S, annuity = self.forward_swap_rate(T_expiry, T_swap)
        K = S  # Assume ATM for simplicity
        
        # Normalize price
        unit_price = market_price / notional
        
        is_payer = payer_receiver.upper() == "PAYER"
        
        if vol_type.upper() == "NORMAL":
            return implied_vol_bachelier(unit_price, S, K, T_expiry, annuity, is_payer)
        else:
            return implied_vol_black(unit_price, S, K, T_expiry, annuity, is_payer)
