"""
Caplet/Floorlet pricing engine.

A caplet is a call option on a forward interest rate.
A floorlet is a put option on a forward interest rate.

Pricing:
    V_caplet = delta_t * DF(T_end) * BaseModel(F, K, T_start, sigma)
    
where:
    - delta_t = T_end - T_start (accrual period)
    - DF(T_end) = discount factor to payment date
    - F = forward rate from T_start to T_end
    - sigma = implied volatility (from SABR or market)
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Tuple

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
class CapletResult:
    """Result from caplet pricing."""
    price: float
    forward: float
    strike: float
    expiry: float
    discount_factor: float
    implied_vol: float
    vol_type: str
    notional: float
    delta_t: float


class CapletPricer:
    """
    Caplet/Floorlet pricing engine.
    
    Prices caplets using:
    - Bachelier (normal) model for normal vol quotes
    - Black'76 model for lognormal vol quotes
    - Shifted Black for negative rate environments
    
    Forward rates are derived from the projection curve.
    Discounting uses the OIS/discount curve.
    """
    
    def __init__(
        self,
        discount_curve: Curve,
        projection_curve: Optional[Curve] = None
    ):
        """
        Initialize caplet pricer.
        
        Args:
            discount_curve: OIS curve for discounting
            projection_curve: Curve for forward rates (defaults to discount)
        """
        self.discount_curve = discount_curve
        self.projection_curve = projection_curve or discount_curve
    
    def forward_rate(
        self,
        start: float,
        end: float
    ) -> float:
        """
        Compute forward rate for period [start, end].
        
        Args:
            start: Start time (years)
            end: End time (years)
            
        Returns:
            Simple forward rate
        """
        if end <= start:
            raise ValueError("End must be greater than start")
        
        df_start = self.projection_curve.discount_factor(start)
        df_end = self.projection_curve.discount_factor(end)
        
        delta_t = end - start
        return (df_start / df_end - 1) / delta_t
    
    def price(
        self,
        F: float,
        K: float,
        T: float,
        df: float,
        vol: float,
        vol_type: str = "NORMAL",
        notional: float = 1.0,
        delta_t: float = 0.25,
        is_cap: bool = True,
        shift: float = 0.0
    ) -> float:
        """
        Price a caplet or floorlet.
        
        Args:
            F: Forward rate
            K: Strike rate
            T: Time to expiry (option expiry, start of rate period)
            df: Discount factor to payment date
            vol: Implied volatility
            vol_type: "NORMAL" or "LOGNORMAL"
            notional: Notional amount
            delta_t: Accrual period (T_end - T_start)
            is_cap: True for caplet, False for floorlet
            shift: Shift for shifted lognormal (only if vol_type="LOGNORMAL")
            
        Returns:
            Caplet/floorlet price
        """
        vol_type = vol_type.upper()
        
        if vol_type == "NORMAL":
            if is_cap:
                base_price = bachelier_call(F, K, T, vol, df)
            else:
                base_price = bachelier_put(F, K, T, vol, df)
        elif vol_type == "LOGNORMAL":
            if shift > 0:
                if is_cap:
                    base_price = shifted_black_call(F, K, T, vol, shift, df)
                else:
                    base_price = shifted_black_put(F, K, T, vol, shift, df)
            else:
                if is_cap:
                    base_price = black76_call(F, K, T, vol, df)
                else:
                    base_price = black76_put(F, K, T, vol, df)
        else:
            raise ValueError(f"Unknown vol_type: {vol_type}")
        
        return notional * delta_t * base_price
    
    def greeks(
        self,
        F: float,
        K: float,
        T: float,
        df: float,
        vol: float,
        vol_type: str = "NORMAL",
        notional: float = 1.0,
        delta_t: float = 0.25,
        is_cap: bool = True
    ) -> Dict[str, float]:
        """
        Compute caplet Greeks.
        
        Args:
            F: Forward rate
            K: Strike rate
            T: Time to expiry
            df: Discount factor
            vol: Implied volatility
            vol_type: "NORMAL" or "LOGNORMAL"
            notional: Notional amount
            delta_t: Accrual period
            is_cap: True for caplet, False for floorlet
            
        Returns:
            Dict with delta, gamma, vega, theta
        """
        vol_type = vol_type.upper()
        scale = notional * delta_t
        
        if vol_type == "NORMAL":
            base_greeks = bachelier_greeks(F, K, T, vol, df, is_cap)
        else:
            base_greeks = black76_greeks(F, K, T, vol, df, is_cap)
        
        return {
            'delta': scale * base_greeks['delta'],
            'gamma': scale * base_greeks['gamma'],
            'vega': scale * base_greeks['vega'],
            'theta': scale * base_greeks['theta']
        }
    
    def price_from_dates(
        self,
        start_date: date,
        end_date: date,
        K: float,
        vol: float,
        vol_type: str = "NORMAL",
        notional: float = 1.0,
        is_cap: bool = True,
        shift: float = 0.0
    ) -> CapletResult:
        """
        Price caplet from dates (derives forward from curves).
        
        Args:
            start_date: Rate fixing date / option expiry
            end_date: Rate payment date
            K: Strike rate
            vol: Implied volatility
            vol_type: "NORMAL" or "LOGNORMAL"
            notional: Notional amount
            is_cap: True for caplet, False for floorlet
            shift: Shift for negative rates
            
        Returns:
            CapletResult with price and details
        """
        anchor = self.discount_curve.anchor_date
        day_count = self.discount_curve.day_count
        
        T_start = year_fraction(anchor, start_date, day_count)
        T_end = year_fraction(anchor, end_date, day_count)
        delta_t = T_end - T_start
        
        # Forward rate
        F = self.forward_rate(T_start, T_end)
        
        # Discount factor to payment
        df = self.discount_curve.discount_factor(T_end)
        
        # Price
        price = self.price(
            F=F, K=K, T=T_start, df=df, vol=vol,
            vol_type=vol_type, notional=notional,
            delta_t=delta_t, is_cap=is_cap, shift=shift
        )
        
        return CapletResult(
            price=price,
            forward=F,
            strike=K,
            expiry=T_start,
            discount_factor=df,
            implied_vol=vol,
            vol_type=vol_type,
            notional=notional,
            delta_t=delta_t
        )
    
    def price_with_sabr(
        self,
        start_date: date,
        end_date: date,
        K: float,
        sabr_params,  # SabrParams
        vol_type: str = "NORMAL",
        notional: float = 1.0,
        is_cap: bool = True
    ) -> CapletResult:
        """
        Price caplet using SABR implied vol.
        
        Args:
            start_date: Rate fixing date
            end_date: Rate payment date
            K: Strike rate
            sabr_params: SabrParams object
            vol_type: "NORMAL" or "LOGNORMAL"
            notional: Notional amount
            is_cap: True for caplet, False for floorlet
            
        Returns:
            CapletResult with price and details
        """
        from ..vol.sabr import SabrModel
        
        anchor = self.discount_curve.anchor_date
        day_count = self.discount_curve.day_count
        
        T_start = year_fraction(anchor, start_date, day_count)
        T_end = year_fraction(anchor, end_date, day_count)
        delta_t = T_end - T_start
        
        # Forward rate
        F = self.forward_rate(T_start, T_end)
        
        # Get SABR implied vol
        model = SabrModel()
        if vol_type.upper() == "NORMAL":
            vol = model.implied_vol_normal(F, K, T_start, sabr_params)
        else:
            vol = model.implied_vol_black(F, K, T_start, sabr_params)
        
        # Discount factor to payment
        df = self.discount_curve.discount_factor(T_end)
        
        # Price
        price = self.price(
            F=F, K=K, T=T_start, df=df, vol=vol,
            vol_type=vol_type, notional=notional,
            delta_t=delta_t, is_cap=is_cap, shift=sabr_params.shift
        )
        
        return CapletResult(
            price=price,
            forward=F,
            strike=K,
            expiry=T_start,
            discount_factor=df,
            implied_vol=vol,
            vol_type=vol_type,
            notional=notional,
            delta_t=delta_t
        )


def price_cap(
    caplet_pricer: CapletPricer,
    start_date: date,
    maturity_date: date,
    K: float,
    vol: float,
    vol_type: str = "NORMAL",
    notional: float = 1.0,
    frequency: int = 4,
    shift: float = 0.0
) -> float:
    """
    Price an interest rate cap (portfolio of caplets).
    
    Args:
        caplet_pricer: CapletPricer instance
        start_date: First fixing date
        maturity_date: Final maturity
        K: Strike rate
        vol: Flat implied vol (or provide SABR per caplet)
        vol_type: "NORMAL" or "LOGNORMAL"
        notional: Notional amount
        frequency: Caplet frequency (4 = quarterly)
        shift: Shift for negative rates
        
    Returns:
        Cap price (sum of caplet prices)
    """
    from ..dates import DateUtils
    
    # Generate schedule
    months_per_period = 12 // frequency
    current = start_date
    total_price = 0.0
    
    while current < maturity_date:
        # Next period end
        next_date = DateUtils.add_tenor(current, f"{months_per_period}M")
        if next_date > maturity_date:
            next_date = maturity_date
        
        # Price caplet
        result = caplet_pricer.price_from_dates(
            start_date=current,
            end_date=next_date,
            K=K,
            vol=vol,
            vol_type=vol_type,
            notional=notional,
            is_cap=True,
            shift=shift
        )
        total_price += result.price
        
        current = next_date
    
    return total_price
