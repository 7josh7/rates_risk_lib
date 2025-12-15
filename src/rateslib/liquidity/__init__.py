"""
Liquidity adjustments for VaR calculations.

Implements:
- Mark-to-liquidation haircuts
- Bid/ask spread adjustments
- Holding period scaling (square-root-of-time rule)
- Position size liquidity factor

Per specification Section 10: LVaR typically adds 5-20% to base VaR.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..curves.curve import Curve


@dataclass
class LiquidityParameters:
    """
    Parameters for liquidity adjustments.
    
    Attributes:
        bid_ask_spreads: Dict of {instrument_id: spread_in_bp}
        holding_period: Days to liquidate position (default 1)
        position_sizes: Dict of {instrument_id: notional}
        adv: Dict of {instrument_id: average_daily_volume}
        stress_multiplier: Multiplier for stressed periods (default 2.0)
    """
    bid_ask_spreads: Dict[str, float] = field(default_factory=dict)
    holding_period: int = 1
    position_sizes: Dict[str, float] = field(default_factory=dict)
    adv: Dict[str, float] = field(default_factory=dict)
    stress_multiplier: float = 2.0


@dataclass
class LiquidityAdjustedVaR:
    """
    Result of liquidity-adjusted VaR calculation.
    
    Attributes:
        base_var: Original VaR before liquidity
        liquidity_cost: Total liquidity adjustment
        lvar: Liquidity-adjusted VaR (LVaR)
        components: Breakdown of liquidity costs
    """
    base_var: float
    liquidity_cost: float
    lvar: float
    components: Dict[str, float] = field(default_factory=dict)
    
    @property
    def liquidity_ratio(self) -> float:
        """Ratio of LVaR to base VaR."""
        if self.base_var == 0:
            return 0
        return self.lvar / self.base_var
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "base_var": self.base_var,
            "liquidity_cost": self.liquidity_cost,
            "lvar": self.lvar,
            "liquidity_ratio": self.liquidity_ratio,
            "components": self.components
        }


# Default bid/ask spreads by instrument type (in basis points)
DEFAULT_SPREADS = {
    "UST_2Y": 0.5,   # Treasuries are very liquid
    "UST_5Y": 0.5,
    "UST_10Y": 0.5,
    "UST_30Y": 1.0,
    "IRS_1Y": 2.0,   # Swaps have wider spreads
    "IRS_2Y": 2.0,
    "IRS_5Y": 2.5,
    "IRS_10Y": 3.0,
    "IRS_30Y": 4.0,
    "OIS_1W": 1.0,   # OIS relatively tight
    "OIS_1M": 1.5,
    "OIS_3M": 2.0,
    "FUT_ED": 0.25,  # Futures very liquid
    "FUT_SOFR": 0.25,
}


class LiquidityEngine:
    """
    Engine for computing liquidity adjustments to VaR.
    
    Combines multiple liquidity factors:
    1. Bid/ask spread cost (half-spread for marking)
    2. Holding period scaling (sqrt-of-time)
    3. Position size impact
    """
    
    def __init__(self, params: Optional[LiquidityParameters] = None):
        """
        Initialize liquidity engine.
        
        Args:
            params: Liquidity parameters (uses defaults if None)
        """
        self.params = params or LiquidityParameters()
    
    def compute_bid_ask_cost(
        self,
        dv01_by_instrument: Dict[str, float],
        is_stressed: bool = False
    ) -> float:
        """
        Compute bid/ask spread cost.
        
        For each instrument: cost = 0.5 * spread * |DV01|
        
        Args:
            dv01_by_instrument: Dict of {instrument: dv01}
            is_stressed: Whether to apply stress multiplier
            
        Returns:
            Total bid/ask cost
        """
        total_cost = 0
        
        multiplier = self.params.stress_multiplier if is_stressed else 1.0
        
        for inst_id, dv01 in dv01_by_instrument.items():
            spread = self.params.bid_ask_spreads.get(
                inst_id, DEFAULT_SPREADS.get(inst_id, 2.0)
            )
            cost = 0.5 * spread * abs(dv01) * multiplier
            total_cost += cost
        
        return total_cost
    
    def compute_holding_period_scaling(
        self,
        base_var: float,
        holding_period: Optional[int] = None
    ) -> float:
        """
        Scale VaR for holding period using square-root-of-time.
        
        Args:
            base_var: 1-day VaR
            holding_period: Days to liquidate (default from params)
            
        Returns:
            Scaled VaR
        """
        hp = holding_period or self.params.holding_period
        
        if hp <= 1:
            return base_var
        
        return base_var * np.sqrt(hp)
    
    def compute_position_impact(
        self,
        dv01_by_instrument: Dict[str, float]
    ) -> float:
        """
        Compute price impact from position size.
        
        Uses a simple model: impact = DV01 * (Position / ADV)^0.5 * impact_factor
        
        Args:
            dv01_by_instrument: Dict of {instrument: dv01}
            
        Returns:
            Position impact cost
        """
        if not self.params.adv:
            return 0
        
        total_impact = 0
        impact_factor = 0.1  # Market impact parameter
        
        for inst_id, dv01 in dv01_by_instrument.items():
            position = self.params.position_sizes.get(inst_id, 0)
            adv = self.params.adv.get(inst_id, float('inf'))
            
            if adv > 0:
                participation = position / adv
                impact = abs(dv01) * np.sqrt(participation) * impact_factor
                total_impact += impact
        
        return total_impact
    
    def compute_lvar(
        self,
        base_var: float,
        dv01_by_instrument: Dict[str, float],
        holding_period: Optional[int] = None,
        is_stressed: bool = False
    ) -> LiquidityAdjustedVaR:
        """
        Compute full liquidity-adjusted VaR.
        
        LVaR = sqrt(VaR_t^2 + LC^2) where:
        - VaR_t is holding-period scaled VaR
        - LC is liquidity cost (bid/ask + position impact)
        
        Args:
            base_var: 1-day VaR
            dv01_by_instrument: DV01 by instrument
            holding_period: Holding period in days
            is_stressed: Whether stressed conditions
            
        Returns:
            LiquidityAdjustedVaR result
        """
        # Scale for holding period
        var_t = self.compute_holding_period_scaling(base_var, holding_period)
        
        # Bid/ask cost
        bid_ask_cost = self.compute_bid_ask_cost(dv01_by_instrument, is_stressed)
        
        # Position impact
        position_impact = self.compute_position_impact(dv01_by_instrument)
        
        # Total liquidity cost
        liquidity_cost = bid_ask_cost + position_impact
        
        # Combine VaR and liquidity cost (assuming independence)
        lvar = np.sqrt(var_t ** 2 + liquidity_cost ** 2)
        
        return LiquidityAdjustedVaR(
            base_var=base_var,
            liquidity_cost=liquidity_cost,
            lvar=lvar,
            components={
                "holding_period_scaled_var": var_t,
                "bid_ask_cost": bid_ask_cost,
                "position_impact": position_impact,
            }
        )


def estimate_liquidation_time(
    position_notional: float,
    adv: float,
    max_participation_rate: float = 0.25
) -> int:
    """
    Estimate days to liquidate a position.
    
    Args:
        position_notional: Total notional to liquidate
        adv: Average daily volume
        max_participation_rate: Max % of ADV per day
        
    Returns:
        Days to liquidate
    """
    if adv <= 0:
        return 10  # Default for illiquid
    
    daily_capacity = adv * max_participation_rate
    days = np.ceil(position_notional / daily_capacity)
    
    return max(1, int(days))


def scale_var_to_horizon(
    one_day_var: float,
    target_horizon: int,
    decay_factor: float = 0.94
) -> float:
    """
    Scale 1-day VaR to longer horizon.
    
    Uses weighted average of:
    - Square-root-of-time rule (standard)
    - Decay-adjusted (for mean reversion)
    
    Args:
        one_day_var: 1-day VaR
        target_horizon: Target horizon in days
        decay_factor: Mean reversion decay (0.94 typical)
        
    Returns:
        Scaled VaR
    """
    # Square-root-of-time
    sqrt_scaling = np.sqrt(target_horizon)
    
    # Decay adjustment (reduces scaling for longer horizons)
    decay_adjustment = np.sqrt((1 - decay_factor ** target_horizon) / (1 - decay_factor))
    
    # Blend (75% sqrt, 25% decay)
    scaling = 0.75 * sqrt_scaling + 0.25 * decay_adjustment
    
    return one_day_var * scaling


__all__ = [
    "LiquidityParameters",
    "LiquidityAdjustedVaR",
    "LiquidityEngine",
    "DEFAULT_SPREADS",
    "estimate_liquidation_time",
    "scale_var_to_horizon",
]
