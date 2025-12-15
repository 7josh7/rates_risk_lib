"""
Key-rate duration calculations.

Implements bucketed key-rate sensitivities:
- Bumps specific curve nodes/tenors
- Computes partial durations
- Supports custom tenor buckets

Key-rate durations show sensitivity to non-parallel curve moves
and are essential for hedging curve risk.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..curves.curve import Curve
from ..dates import DateUtils
from .bumping import BumpEngine


# Standard USD key rate tenors
STANDARD_KEY_RATE_TENORS = [
    "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"
]


@dataclass
class KeyRateDV01:
    """
    Key-rate DV01 results.
    
    Attributes:
        as_of_date: Calculation date
        tenors: List of tenors
        dv01s: Dict of {tenor: dv01}
        total_dv01: Sum of key-rate DV01s (should approximate parallel DV01)
        pv: Base present value
    """
    as_of_date: date
    tenors: List[str]
    dv01s: Dict[str, float]
    total_dv01: float
    pv: float
    
    def to_array(self) -> np.ndarray:
        """Return DV01s as numpy array in tenor order."""
        return np.array([self.dv01s.get(t, 0.0) for t in self.tenors])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "pv": self.pv,
            "total_dv01": self.total_dv01,
            "key_rate_dv01": self.dv01s
        }


class KeyRateEngine:
    """
    Engine for computing key-rate durations.
    
    Supports:
    - Standard tenor buckets
    - Custom tenor definitions
    - Interpolated bumps (for smooth profiles)
    - Localized bumps (step function)
    """
    
    def __init__(
        self,
        curve: Curve,
        tenors: Optional[List[str]] = None,
        bump_size: float = 1.0,
        interpolate_bumps: bool = False
    ):
        """
        Initialize key-rate engine.
        
        Args:
            curve: Base yield curve
            tenors: Key rate tenors (default: standard set)
            bump_size: Bump size in bp
            interpolate_bumps: Whether to use triangular interpolation
        """
        self.curve = curve
        self.tenors = tenors or STANDARD_KEY_RATE_TENORS
        self.bump_size = bump_size
        self.interpolate_bumps = interpolate_bumps
        self._tenor_years = [DateUtils.tenor_to_years(t) for t in self.tenors]
    
    def compute_key_rate_dv01(
        self,
        pricer_func: Callable[[Curve], float]
    ) -> KeyRateDV01:
        """
        Compute key-rate DV01 for all tenors.
        
        For each tenor, bump only that point and reprice.
        
        Args:
            pricer_func: Function that takes curve and returns PV
            
        Returns:
            KeyRateDV01 object
        """
        pv_base = pricer_func(self.curve)
        dv01s = {}
        
        for i, tenor in enumerate(self.tenors):
            if self.interpolate_bumps:
                bumped_curve = self._create_interpolated_bump(i)
            else:
                bumped_curve = self._create_localized_bump(tenor)
            
            pv_bumped = pricer_func(bumped_curve)
            dv01 = (pv_bumped - pv_base) / self.bump_size
            dv01s[tenor] = dv01
        
        total_dv01 = sum(dv01s.values())
        
        return KeyRateDV01(
            as_of_date=self.curve.anchor_date,
            tenors=self.tenors,
            dv01s=dv01s,
            total_dv01=total_dv01,
            pv=pv_base
        )
    
    def _create_localized_bump(self, tenor: str) -> Curve:
        """
        Create curve with bump at single tenor.
        
        Args:
            tenor: Tenor to bump
            
        Returns:
            Bumped curve
        """
        return self.curve.bump_tenor(tenor, self.bump_size)
    
    def _create_interpolated_bump(self, tenor_index: int) -> Curve:
        """
        Create curve with triangular bump centered at tenor.
        
        The bump profile is triangular:
        - 0 at previous tenor
        - bump_size at current tenor
        - 0 at next tenor
        
        This provides smoother key-rate sensitivities.
        
        Args:
            tenor_index: Index of center tenor
            
        Returns:
            Bumped curve
        """
        from ..curves.curve import CurveNode
        
        center_years = self._tenor_years[tenor_index]
        
        # Get neighboring tenors for interpolation
        if tenor_index > 0:
            left_years = self._tenor_years[tenor_index - 1]
        else:
            left_years = 0.0
        
        if tenor_index < len(self._tenor_years) - 1:
            right_years = self._tenor_years[tenor_index + 1]
        else:
            right_years = center_years + (center_years - left_years)
        
        # Build bump profile
        def bump_at_time(t: float) -> float:
            """Triangular bump profile."""
            if t <= left_years or t >= right_years:
                return 0.0
            
            if t <= center_years:
                # Rising portion
                if center_years == left_years:
                    return self.bump_size
                return self.bump_size * (t - left_years) / (center_years - left_years)
            else:
                # Falling portion
                if right_years == center_years:
                    return self.bump_size
                return self.bump_size * (right_years - t) / (right_years - center_years)
        
        # Create bumped curve
        new_curve = Curve(
            anchor_date=self.curve.anchor_date,
            currency=self.curve.currency,
            day_count=self.curve.day_count,
            interpolation_method=self.curve.interpolation_method
        )
        
        new_curve._nodes = []
        
        for node in self.curve._nodes:
            if node.time == 0:
                new_curve._nodes.append(node)
            else:
                bump = bump_at_time(node.time) / 10000.0  # Convert bp to decimal
                new_zr = node.zero_rate + bump
                new_node = CurveNode.from_zero_rate(node.time, new_zr)
                new_curve._nodes.append(new_node)
        
        new_curve.build()
        return new_curve
    
    def compute_hedge_ratios(
        self,
        target_dv01: KeyRateDV01,
        hedge_instruments: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute hedge ratios to neutralize key-rate risk.
        
        Args:
            target_dv01: Target portfolio's key-rate DV01
            hedge_instruments: List of hedge instruments with their KR-DV01s
            
        Returns:
            Dict of {instrument_id: hedge_ratio}
            
        Note: This is a simplified single-instrument-per-bucket approach.
        A full implementation would solve a least-squares problem.
        """
        hedge_ratios = {}
        
        for hedge in hedge_instruments:
            inst_id = hedge.get("id", "")
            inst_kr_dv01 = hedge.get("key_rate_dv01", {})
            inst_dv01 = hedge.get("dv01", 0)
            
            if inst_dv01 == 0:
                continue
            
            # Find dominant tenor
            dominant_tenor = max(inst_kr_dv01.keys(), 
                                key=lambda t: abs(inst_kr_dv01.get(t, 0)))
            
            target_kr = target_dv01.dv01s.get(dominant_tenor, 0)
            hedge_kr = inst_kr_dv01.get(dominant_tenor, 0)
            
            if hedge_kr != 0:
                ratio = -target_kr / hedge_kr
                hedge_ratios[inst_id] = ratio
        
        return hedge_ratios


def compute_key_rate_contributions(
    key_rate_dv01: KeyRateDV01,
    rate_changes: Dict[str, float]
) -> Tuple[Dict[str, float], float]:
    """
    Compute P&L contribution from each key rate.
    
    Args:
        key_rate_dv01: Key-rate DV01 object
        rate_changes: Dict of {tenor: rate_change_in_bp}
        
    Returns:
        Tuple of (contribution_dict, total_pnl)
    """
    contributions = {}
    total_pnl = 0.0
    
    for tenor, dv01 in key_rate_dv01.dv01s.items():
        rate_change = rate_changes.get(tenor, 0.0)
        contribution = -dv01 * rate_change
        contributions[tenor] = contribution
        total_pnl += contribution
    
    return contributions, total_pnl


def bucket_cashflow_risk(
    cashflow_times: List[float],
    cashflow_dv01s: List[float],
    bucket_tenors: List[str] = None
) -> Dict[str, float]:
    """
    Bucket cashflow-level risk into standard key-rate buckets.
    
    Uses linear interpolation to allocate risk between adjacent buckets.
    
    Args:
        cashflow_times: Times (in years) of cashflows
        cashflow_dv01s: DV01 contribution of each cashflow
        bucket_tenors: Target buckets (default: standard)
        
    Returns:
        Dict of {tenor: bucketed_dv01}
    """
    bucket_tenors = bucket_tenors or STANDARD_KEY_RATE_TENORS
    bucket_years = [DateUtils.tenor_to_years(t) for t in bucket_tenors]
    
    bucketed = {t: 0.0 for t in bucket_tenors}
    
    for cf_time, cf_dv01 in zip(cashflow_times, cashflow_dv01s):
        # Find bracketing buckets
        if cf_time <= bucket_years[0]:
            # Before first bucket - assign to first
            bucketed[bucket_tenors[0]] += cf_dv01
        elif cf_time >= bucket_years[-1]:
            # After last bucket - assign to last
            bucketed[bucket_tenors[-1]] += cf_dv01
        else:
            # Interpolate between buckets
            for i in range(len(bucket_years) - 1):
                if bucket_years[i] <= cf_time <= bucket_years[i + 1]:
                    # Linear interpolation
                    t1, t2 = bucket_years[i], bucket_years[i + 1]
                    w2 = (cf_time - t1) / (t2 - t1)
                    w1 = 1 - w2
                    
                    bucketed[bucket_tenors[i]] += cf_dv01 * w1
                    bucketed[bucket_tenors[i + 1]] += cf_dv01 * w2
                    break
    
    return bucketed


__all__ = [
    "KeyRateEngine",
    "KeyRateDV01",
    "STANDARD_KEY_RATE_TENORS",
    "compute_key_rate_contributions",
    "bucket_cashflow_risk",
]
