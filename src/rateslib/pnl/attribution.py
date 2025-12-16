"""
P&L Attribution module.

Implements daily P&L explain comparing yesterday's curve to today's curve.

Components:
1. Carry & Roll-down: Time decay assuming no curve change
2. Curve Move (KRD): Predicted from key-rate DV01 × rate changes
3. Convexity: Second-order correction for parallel moves
4. Residual: Unexplained portion

Formula:
    Realized P&L = PV(t+1, today_curve) - PV(t, yesterday_curve)
    
    Predicted P&L = Carry/Roll + KRD_move + Convexity_effect
    
    Residual = Realized - Predicted

Residual sources:
- Cross-gamma (non-linear interactions)
- Interpolation effects
- Day count / accrual differences
- Timing differences
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..curves.curve import Curve
from ..risk.sensitivities import RiskCalculator, InstrumentRisk
from ..risk.keyrate import KeyRateDV01
from ..market_state import MarketState


@dataclass
class PnLComponents:
    """
    P&L attribution breakdown.
    
    Attributes:
        carry: Income from passage of time
        rolldown: Value change from rolling down the curve
        curve_move_parallel: P&L from parallel curve shift
        curve_move_nonparallel: P&L from non-parallel (KRD) moves
        convexity: Second-order curve effect
        residual: Unexplained portion
    """
    carry: float = 0.0
    rolldown: float = 0.0
    curve_move_parallel: float = 0.0
    curve_move_nonparallel: float = 0.0
    convexity: float = 0.0
    residual: float = 0.0
    
    @property
    def carry_rolldown(self) -> float:
        """Combined carry and roll-down."""
        return self.carry + self.rolldown
    
    @property
    def curve_move_total(self) -> float:
        """Total curve move P&L."""
        return self.curve_move_parallel + self.curve_move_nonparallel
    
    @property
    def predicted_total(self) -> float:
        """Total predicted P&L."""
        return self.carry_rolldown + self.curve_move_total + self.convexity
    
    @property
    def realized_total(self) -> float:
        """Realized P&L (predicted + residual)."""
        return self.predicted_total + self.residual
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "carry": self.carry,
            "rolldown": self.rolldown,
            "carry_rolldown": self.carry_rolldown,
            "curve_move_parallel": self.curve_move_parallel,
            "curve_move_nonparallel": self.curve_move_nonparallel,
            "curve_move_total": self.curve_move_total,
            "convexity": self.convexity,
            "predicted_total": self.predicted_total,
            "residual": self.residual,
            "realized_total": self.realized_total
        }


@dataclass
class PnLAttribution:
    """
    P&L Attribution result for an instrument or portfolio.
    
    Attributes:
        instrument_id: Identifier
        date_t0: Previous date
        date_t1: Current date  
        pv_t0: PV at t0 with t0 curve
        pv_t1: PV at t1 with t1 curve
        components: Breakdown of P&L
    """
    instrument_id: str
    date_t0: date
    date_t1: date
    pv_t0: float
    pv_t1: float
    components: PnLComponents
    key_rate_contributions: Dict[str, float] = field(default_factory=dict)
    
    @property
    def realized_pnl(self) -> float:
        """Realized P&L."""
        return self.pv_t1 - self.pv_t0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting."""
        return {
            "instrument_id": self.instrument_id,
            "date_t0": self.date_t0.isoformat(),
            "date_t1": self.date_t1.isoformat(),
            "pv_t0": self.pv_t0,
            "pv_t1": self.pv_t1,
            "realized_pnl": self.realized_pnl,
            **self.components.to_dict(),
            "key_rate_contributions": self.key_rate_contributions
        }


class PnLAttributionEngine:
    """
    Engine for computing P&L attribution.
    
    Requires curves from two consecutive days and pricer functions.
    """
    
    # Standard key-rate tenors for attribution
    KEY_RATE_TENORS = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    
    def __init__(
        self,
        curve_t0: Curve,
        curve_t1: Curve,
        date_t0: date,
        date_t1: date
    ):
        """
        Initialize attribution engine.
        
        Args:
            curve_t0: Curve at t0 (yesterday)
            curve_t1: Curve at t1 (today)
            date_t0: Date for t0
            date_t1: Date for t1
        """
        self.curve_t0 = curve_t0
        self.curve_t1 = curve_t1
        self.date_t0 = date_t0
        self.date_t1 = date_t1
        
        # Calculate curve changes
        self._compute_curve_changes()
    
    def _compute_curve_changes(self) -> None:
        """Compute rate changes between curves."""
        from ..dates import DateUtils
        
        self.rate_changes = {}
        self.parallel_change = 0.0
        
        for tenor in self.KEY_RATE_TENORS:
            t = DateUtils.tenor_to_years(tenor)
            rate_t0 = self.curve_t0.zero_rate(t) * 10000  # Convert to bp
            rate_t1 = self.curve_t1.zero_rate(t) * 10000
            change_bp = rate_t1 - rate_t0
            self.rate_changes[tenor] = change_bp
        
        # Parallel change is average
        self.parallel_change = np.mean(list(self.rate_changes.values()))
    
    def attribute_pnl(
        self,
        instrument_id: str,
        pricer_func_t0: Callable[[Curve, date], float],
        pricer_func_t1: Callable[[Curve, date], float],
        risk_t0: Optional[InstrumentRisk] = None
    ) -> PnLAttribution:
        """
        Compute P&L attribution for an instrument.
        
        Args:
            instrument_id: Instrument identifier
            pricer_func_t0: Pricer for t0 (takes curve and settle date)
            pricer_func_t1: Pricer for t1
            risk_t0: Optional pre-computed risk at t0
            
        Returns:
            PnLAttribution object
        """
        # Base PVs
        pv_t0 = pricer_func_t0(self.curve_t0, self.date_t0)
        pv_t1 = pricer_func_t1(self.curve_t1, self.date_t1)
        realized_pnl = pv_t1 - pv_t0
        
        # 1. Carry & Roll-down
        # PV at t1 using t0 curve (time passes, curve unchanged)
        pv_t1_old_curve = pricer_func_t1(self.curve_t0, self.date_t1)
        carry_rolldown = pv_t1_old_curve - pv_t0
        
        # 2. Curve move components
        # If we have risk metrics, use them
        curve_move_parallel = 0.0
        curve_move_nonparallel = 0.0
        convexity_effect = 0.0
        kr_contributions = {}
        
        if risk_t0 is not None:
            # DV01-based prediction
            dv01 = risk_t0.dv01
            curve_move_parallel = -dv01 * self.parallel_change
            
            # Key-rate contributions
            for tenor, kr_dv01 in risk_t0.key_rate_dv01.items():
                rate_change = self.rate_changes.get(tenor, 0.0)
                contribution = -kr_dv01 * rate_change
                kr_contributions[tenor] = contribution
            
            # Non-parallel is sum of KRD contributions minus parallel
            curve_move_nonparallel = sum(kr_contributions.values()) - curve_move_parallel
            
            # Convexity effect
            convexity = risk_t0.convexity
            parallel_decimal = self.parallel_change / 10000.0
            convexity_effect = 0.5 * convexity * (parallel_decimal ** 2)
        else:
            # Approximate from repricing
            # PV at t1 with bumped t0 curve
            pv_t1_bumped = pricer_func_t1(self.curve_t1, self.date_t1)
            curve_move_total = pv_t1 - pv_t1_old_curve
            curve_move_parallel = curve_move_total  # Simplified
        
        # 3. Compute residual
        predicted = carry_rolldown + curve_move_parallel + curve_move_nonparallel + convexity_effect
        residual = realized_pnl - predicted
        
        components = PnLComponents(
            carry=carry_rolldown * 0.3,  # Rough split (carry is partial)
            rolldown=carry_rolldown * 0.7,  # Rough split
            curve_move_parallel=curve_move_parallel,
            curve_move_nonparallel=curve_move_nonparallel,
            convexity=convexity_effect,
            residual=residual
        )
        
        return PnLAttribution(
            instrument_id=instrument_id,
            date_t0=self.date_t0,
            date_t1=self.date_t1,
            pv_t0=pv_t0,
            pv_t1=pv_t1,
            components=components,
            key_rate_contributions=kr_contributions
        )


def compute_daily_pnl(
    curve_t0: Curve,
    curve_t1: Curve,
    date_t0: date,
    date_t1: date,
    pricer_func: Callable[[Curve, date], float],
    instrument_id: str = "PORTFOLIO",
    dv01: Optional[float] = None,
    key_rate_dv01: Optional[Dict[str, float]] = None,
    convexity: Optional[float] = None
) -> PnLAttribution:
    """
    Convenience function for daily P&L attribution.
    
    Args:
        curve_t0: Yesterday's curve
        curve_t1: Today's curve
        date_t0: Yesterday's date
        date_t1: Today's date
        pricer_func: Function(curve, date) -> PV
        instrument_id: Identifier
        dv01: Optional DV01 for prediction
        key_rate_dv01: Optional key-rate DV01s
        convexity: Optional dollar convexity
        
    Returns:
        PnLAttribution object
    """
    # Create a mock InstrumentRisk if we have metrics
    risk = None
    if dv01 is not None:
        from ..risk.sensitivities import InstrumentRisk
        risk = InstrumentRisk(
            instrument_id=instrument_id,
            instrument_type="PORTFOLIO",
            pv=pricer_func(curve_t0, date_t0),
            notional=0,
            dv01=dv01,
            modified_duration=0,
            convexity=convexity or 0,
            key_rate_dv01=key_rate_dv01 or {}
        )
    
    engine = PnLAttributionEngine(curve_t0, curve_t1, date_t0, date_t1)
    
    return engine.attribute_pnl(
        instrument_id,
        lambda c, d: pricer_func(c, d),
        lambda c, d: pricer_func(c, d),
        risk
    )


def compute_carry_rolldown(
    curve: Curve,
    pricer_func: Callable[[Curve, date], float],
    date_t0: date,
    holding_days: int = 1
) -> Tuple[float, float, float]:
    """
    Compute carry and roll-down for a position.
    
    Carry: Income assuming curve doesn't change
    Roll-down: Value change from moving down the curve
    
    Args:
        curve: Current yield curve
        pricer_func: Pricer function (curve, settle_date) -> PV
        date_t0: Current date
        holding_days: Holding period in days
        
    Returns:
        Tuple of (carry, rolldown, total)
    """
    date_t1 = date_t0 + timedelta(days=holding_days)
    
    pv_t0 = pricer_func(curve, date_t0)
    pv_t1 = pricer_func(curve, date_t1)  # Same curve, later date
    
    total = pv_t1 - pv_t0
    
    # Split is approximate - proper carry would need accrual calculation
    # Roll-down dominates for bonds, carry for money market
    carry = total * 0.3
    rolldown = total * 0.7
    
    return carry, rolldown, total


def attribute_curve_vs_vol(
    base_state: MarketState,
    shocked_state: MarketState,
    price_func: Callable[[MarketState], float],
    curve_only_state: Optional[MarketState] = None,
    vol_only_state: Optional[MarketState] = None,
) -> Dict[str, float]:
    """
    Attribute P&L into curve, vol, and cross components using bump-and-reprice.

    Args:
        base_state: MarketState at t0
        shocked_state: MarketState after combined shock
        price_func: Callable that takes MarketState -> PV
        curve_only_state: Optional state with curve shocked only
        vol_only_state: Optional state with SABR shocked only

    Returns:
        Dict with curve, vol, cross, residual, total, base_pv, shocked_pv
    """
    base_pv = price_func(base_state)

    curve_state = curve_only_state or base_state.copy(curve=shocked_state.curve, sabr_surface=base_state.sabr_surface)
    vol_state = vol_only_state or base_state.copy(curve=base_state.curve, sabr_surface=shocked_state.sabr_surface)

    curve_pv = price_func(curve_state)
    vol_pv = price_func(vol_state)
    shocked_pv = price_func(shocked_state)

    curve_component = curve_pv - base_pv
    vol_component = vol_pv - base_pv
    cross = shocked_pv - curve_pv - vol_pv + base_pv
    total = shocked_pv - base_pv
    residual = total - (curve_component + vol_component + cross)

    return {
        "base_pv": base_pv,
        "shocked_pv": shocked_pv,
        "curve": curve_component,
        "vol": vol_component,
        "cross": cross,
        "residual": residual,
        "total": total,
    }


__all__ = [
    "PnLAttribution",
    "PnLComponents",
    "PnLAttributionEngine",
    "compute_daily_pnl",
    "compute_carry_rolldown",
    "attribute_curve_vs_vol",
]
