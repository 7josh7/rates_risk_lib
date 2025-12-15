"""
Curve bumping framework for sensitivity calculations.

Provides a generic bump-and-reprice engine:
- Parallel bumps (all nodes)
- Single node bumps
- Tenor-specific bumps
- Custom bump profiles

Bump types:
- Additive (shift in bp)
- Multiplicative (percentage change)
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..curves.curve import Curve


class BumpType(Enum):
    """Type of curve bump."""
    ADDITIVE = "additive"          # Add bp to rate
    MULTIPLICATIVE = "multiplicative"  # Multiply rate by factor


@dataclass
class BumpResult:
    """Result of a bump operation."""
    original_pv: float
    bumped_pv: float
    bump_size: float
    bump_type: str
    delta_pv: float = field(init=False)
    
    def __post_init__(self):
        self.delta_pv = self.bumped_pv - self.original_pv


@dataclass 
class BumpProfile:
    """
    Defines how to bump a curve.
    
    Attributes:
        tenors: List of tenors to bump (None = all)
        bump_size: Bump size in bp (additive) or % (multiplicative)
        bump_type: Type of bump
        scale_by_tenor: Whether to scale bump by tenor
    """
    tenors: Optional[List[str]] = None
    bump_size: float = 1.0  # 1 bp
    bump_type: BumpType = BumpType.ADDITIVE
    scale_by_tenor: bool = False


class BumpEngine:
    """
    Engine for curve bumping and sensitivity calculation.
    
    Provides methods to:
    1. Create bumped curves
    2. Calculate delta PV from bumps
    3. Compute DV01, convexity, and other sensitivities
    """
    
    def __init__(self, base_curve: Curve):
        """
        Initialize bump engine with base curve.
        
        Args:
            base_curve: The curve to bump
        """
        self.base_curve = base_curve
    
    def parallel_bump(self, bp: float) -> Curve:
        """
        Create parallel-bumped curve.
        
        Args:
            bp: Bump size in basis points
            
        Returns:
            Bumped curve
        """
        return self.base_curve.bump_parallel(bp)
    
    def node_bump(self, node_index: int, bp: float) -> Curve:
        """
        Bump a single node.
        
        Args:
            node_index: Index of node to bump
            bp: Bump size in basis points
            
        Returns:
            Bumped curve
        """
        return self.base_curve.bump_node(node_index, bp)
    
    def tenor_bump(self, tenor: str, bp: float) -> Curve:
        """
        Bump the node closest to a tenor.
        
        Args:
            tenor: Tenor string (e.g., "2Y")
            bp: Bump size in basis points
            
        Returns:
            Bumped curve
        """
        return self.base_curve.bump_tenor(tenor, bp)
    
    def custom_bump(
        self,
        bump_vector: Dict[str, float]
    ) -> Curve:
        """
        Apply custom bumps to multiple tenors.
        
        Args:
            bump_vector: Dict of {tenor: bump_in_bp}
            
        Returns:
            Bumped curve
        """
        from ..dates import DateUtils
        
        # Start with copy of base curve
        bumped = self.base_curve.copy()
        
        # Clear nodes except t=0
        nodes = list(bumped._nodes)
        bumped._nodes = [n for n in nodes if n.time == 0]
        
        # Apply bumps
        for node in nodes:
            if node.time == 0:
                continue
            
            # Find best matching tenor
            best_tenor = None
            best_dist = float('inf')
            
            for tenor in bump_vector.keys():
                target_time = DateUtils.tenor_to_years(tenor)
                dist = abs(node.time - target_time)
                if dist < best_dist:
                    best_dist = dist
                    best_tenor = tenor
            
            # Apply bump if close enough (within 0.5 years)
            if best_tenor and best_dist < 0.5:
                bump = bump_vector[best_tenor] / 10000.0  # Convert bp to decimal
                new_zr = node.zero_rate + bump
                from ..curves.curve import CurveNode
                new_node = CurveNode.from_zero_rate(node.time, new_zr)
                bumped._nodes.append(new_node)
            else:
                bumped._nodes.append(node)
        
        bumped._nodes.sort(key=lambda n: n.time)
        bumped.build()
        return bumped
    
    def compute_dv01(
        self,
        pricer_func: Callable[[Curve], float],
        bump_size: float = 1.0
    ) -> float:
        """
        Compute DV01 using parallel bump.
        
        DV01 = (PV_down - PV_up) / 2
        
        This gives the dollar value change for a 1bp parallel move.
        
        Args:
            pricer_func: Function that takes a curve and returns PV
            bump_size: Bump size in bp (default 1)
            
        Returns:
            DV01 (dollar value of 1bp)
        """
        pv_base = pricer_func(self.base_curve)
        pv_up = pricer_func(self.parallel_bump(bump_size))
        pv_down = pricer_func(self.parallel_bump(-bump_size))
        
        # Central difference
        dv01 = (pv_down - pv_up) / (2 * bump_size)
        return dv01
    
    def compute_convexity(
        self,
        pricer_func: Callable[[Curve], float],
        bump_size: float = 1.0
    ) -> float:
        """
        Compute dollar convexity using second difference.
        
        Convexity = (PV_up + PV_down - 2*PV_base) / (bump^2)
        
        Args:
            pricer_func: Function that takes a curve and returns PV
            bump_size: Bump size in bp
            
        Returns:
            Dollar convexity
        """
        pv_base = pricer_func(self.base_curve)
        pv_up = pricer_func(self.parallel_bump(bump_size))
        pv_down = pricer_func(self.parallel_bump(-bump_size))
        
        # Second difference
        bump_decimal = bump_size / 10000.0
        convexity = (pv_up + pv_down - 2 * pv_base) / (bump_decimal ** 2)
        return convexity
    
    def compute_node_deltas(
        self,
        pricer_func: Callable[[Curve], float],
        bump_size: float = 1.0
    ) -> List[Tuple[float, float]]:
        """
        Compute delta for each curve node.
        
        Args:
            pricer_func: Pricer function
            bump_size: Bump in bp
            
        Returns:
            List of (node_time, delta) tuples
        """
        pv_base = pricer_func(self.base_curve)
        deltas = []
        
        for i, node in enumerate(self.base_curve._nodes):
            if node.time == 0:
                continue
            
            bumped = self.node_bump(i, bump_size)
            pv_bumped = pricer_func(bumped)
            delta = (pv_bumped - pv_base) / bump_size
            deltas.append((node.time, delta))
        
        return deltas
    
    def scenario_pv(
        self,
        pricer_func: Callable[[Curve], float],
        bump_profile: Dict[str, float]
    ) -> BumpResult:
        """
        Calculate PV under a scenario (custom bump profile).
        
        Args:
            pricer_func: Pricer function
            bump_profile: Dict of {tenor: bump_in_bp}
            
        Returns:
            BumpResult with original and scenario PV
        """
        pv_original = pricer_func(self.base_curve)
        bumped_curve = self.custom_bump(bump_profile)
        pv_scenario = pricer_func(bumped_curve)
        
        return BumpResult(
            original_pv=pv_original,
            bumped_pv=pv_scenario,
            bump_size=sum(bump_profile.values()) / len(bump_profile) if bump_profile else 0,
            bump_type="scenario"
        )


def create_parallel_scenario(bp: float) -> Dict[str, float]:
    """Create parallel bump scenario."""
    tenors = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    return {t: bp for t in tenors}


def create_steepener_scenario(short_bp: float, long_bp: float) -> Dict[str, float]:
    """Create steepener/flattener scenario."""
    return {
        "3M": short_bp,
        "6M": short_bp,
        "1Y": short_bp,
        "2Y": short_bp * 0.5 + long_bp * 0.5,
        "5Y": long_bp,
        "10Y": long_bp,
        "30Y": long_bp
    }


def create_twist_scenario(pivot_tenor: str = "5Y", short_bp: float = -25, long_bp: float = 25) -> Dict[str, float]:
    """Create twist scenario around pivot."""
    from ..dates import DateUtils
    
    pivot_years = DateUtils.tenor_to_years(pivot_tenor)
    
    tenors = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    scenario = {}
    
    for t in tenors:
        years = DateUtils.tenor_to_years(t)
        if years < pivot_years:
            # Short end
            weight = (pivot_years - years) / pivot_years
            scenario[t] = short_bp * weight
        else:
            # Long end
            weight = (years - pivot_years) / (30 - pivot_years)
            scenario[t] = long_bp * min(weight, 1.0)
    
    return scenario


__all__ = [
    "BumpEngine",
    "BumpType",
    "BumpResult",
    "BumpProfile",
    "create_parallel_scenario",
    "create_steepener_scenario", 
    "create_twist_scenario",
]
