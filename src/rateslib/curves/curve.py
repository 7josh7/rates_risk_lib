"""
Yield curve representation and operations.

The Curve class provides:
- Discount factor P(0,t)
- Zero rate z(t)
- Forward rate f(t1, t2)
- Instantaneous forward rate f(t)

Internal representation uses year fractions from anchor date and
stores either zero rates or log discount factors for interpolation.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from ..conventions import DayCount, year_fraction, CompoundingConvention
from ..dates import DateUtils
from .interpolation import (
    Interpolator, 
    CubicSplineInterpolator, 
    LinearInterpolator,
    create_interpolator
)


@dataclass
class CurveNode:
    """A single point on the curve."""
    time: float  # Year fraction from anchor
    discount_factor: float
    zero_rate: float  # Continuously compounded
    
    @classmethod
    def from_discount_factor(cls, time: float, df: float) -> "CurveNode":
        """Create node from discount factor."""
        if time <= 0:
            return cls(time=time, discount_factor=df, zero_rate=0.0)
        zr = -np.log(df) / time
        return cls(time=time, discount_factor=df, zero_rate=zr)
    
    @classmethod
    def from_zero_rate(cls, time: float, zr: float) -> "CurveNode":
        """Create node from continuously compounded zero rate."""
        df = np.exp(-zr * time)
        return cls(time=time, discount_factor=df, zero_rate=zr)


class Curve:
    """
    Yield curve with interpolation.
    
    Stores discount factors at discrete nodes and interpolates
    between them using the specified method.
    
    Attributes:
        anchor_date: Valuation date (time 0)
        currency: Currency code (default "USD")
        day_count: Day count for time calculations
        interpolation_method: Name of interpolation method
    
    Conventions:
        - Zero rates are continuously compounded
        - Times are year fractions from anchor date
        - Discount factor at t=0 is 1.0
    """
    
    def __init__(
        self,
        anchor_date: date,
        currency: str = "USD",
        day_count: DayCount = DayCount.ACT_365,
        interpolation_method: str = "cubic_spline"
    ):
        self.anchor_date = anchor_date
        self.currency = currency
        self.day_count = day_count
        self.interpolation_method = interpolation_method
        
        # Curve data
        self._nodes: List[CurveNode] = []
        self._interpolator: Optional[Interpolator] = None
        self._is_fitted = False
        
        # Always have node at t=0
        self._nodes.append(CurveNode(time=0.0, discount_factor=1.0, zero_rate=0.0))
    
    def add_node(self, time: float, discount_factor: float) -> None:
        """
        Add a discount factor node to the curve.
        
        Args:
            time: Year fraction from anchor date
            discount_factor: Discount factor P(0,t)
        """
        if time < 0:
            raise ValueError("Time must be non-negative")
        if discount_factor <= 0 or discount_factor > 1:
            raise ValueError(f"Invalid discount factor: {discount_factor}")
        
        node = CurveNode.from_discount_factor(time, discount_factor)
        
        # Insert in sorted order by time
        idx = 0
        for i, n in enumerate(self._nodes):
            if abs(n.time - time) < 1e-10:
                # Replace existing node
                self._nodes[i] = node
                self._is_fitted = False
                return
            if n.time > time:
                break
            idx = i + 1
        
        self._nodes.insert(idx, node)
        self._is_fitted = False
    
    def add_node_from_date(self, d: date, discount_factor: float) -> None:
        """Add a node using a date instead of year fraction."""
        time = year_fraction(self.anchor_date, d, self.day_count)
        self.add_node(time, discount_factor)
    
    def build(self) -> None:
        """
        Build the interpolator from current nodes.
        
        Must be called after adding nodes and before querying the curve.
        """
        if len(self._nodes) < 2:
            raise ValueError("Need at least 2 nodes to build curve")
        
        times = np.array([n.time for n in self._nodes])
        zero_rates = np.array([n.zero_rate for n in self._nodes])
        
        self._interpolator = create_interpolator(self.interpolation_method)
        self._interpolator.fit(times, zero_rates)
        self._is_fitted = True
    
    def _ensure_fitted(self) -> None:
        """Ensure interpolator is fitted."""
        if not self._is_fitted or self._interpolator is None:
            if len(self._nodes) >= 2:
                self.build()
            else:
                raise RuntimeError("Curve not fitted - add more nodes and call build()")
    
    def discount_factor(self, t: Union[float, date]) -> float:
        """
        Get discount factor P(0,t).
        
        Args:
            t: Year fraction or date
            
        Returns:
            Discount factor
        """
        if isinstance(t, date):
            t = year_fraction(self.anchor_date, t, self.day_count)
        
        if t <= 0:
            return 1.0
        
        self._ensure_fitted()
        zr = self._interpolator.interpolate(t)
        return np.exp(-zr * t)
    
    def zero_rate(
        self, 
        t: Union[float, date],
        compounding: CompoundingConvention = CompoundingConvention.CONTINUOUS
    ) -> float:
        """
        Get zero rate z(t).
        
        Args:
            t: Year fraction or date
            compounding: Compounding convention for output
            
        Returns:
            Zero rate (default continuously compounded)
        """
        if isinstance(t, date):
            t = year_fraction(self.anchor_date, t, self.day_count)
        
        if t <= 0:
            # Return short-term rate
            if len(self._nodes) > 1:
                return self._nodes[1].zero_rate
            return 0.0
        
        self._ensure_fitted()
        zr_cont = self._interpolator.interpolate(t)
        
        # Convert to requested compounding
        if compounding == CompoundingConvention.CONTINUOUS:
            return zr_cont
        elif compounding == CompoundingConvention.ANNUAL:
            return np.exp(zr_cont) - 1
        elif compounding == CompoundingConvention.SEMI_ANNUAL:
            return 2 * (np.exp(zr_cont / 2) - 1)
        elif compounding == CompoundingConvention.QUARTERLY:
            return 4 * (np.exp(zr_cont / 4) - 1)
        elif compounding == CompoundingConvention.SIMPLE:
            return zr_cont  # Approximate
        else:
            return zr_cont
    
    def forward_rate(
        self, 
        t1: Union[float, date], 
        t2: Union[float, date],
        compounding: CompoundingConvention = CompoundingConvention.SIMPLE
    ) -> float:
        """
        Get forward rate f(t1, t2).
        
        Args:
            t1: Start time (year fraction or date)
            t2: End time (year fraction or date)
            compounding: Compounding convention
            
        Returns:
            Forward rate between t1 and t2
        """
        if isinstance(t1, date):
            t1 = year_fraction(self.anchor_date, t1, self.day_count)
        if isinstance(t2, date):
            t2 = year_fraction(self.anchor_date, t2, self.day_count)
        
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")
        
        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        
        delta = t2 - t1
        
        if compounding == CompoundingConvention.SIMPLE:
            return (df1 / df2 - 1) / delta
        elif compounding == CompoundingConvention.CONTINUOUS:
            return -np.log(df2 / df1) / delta
        else:
            # Default to simple
            return (df1 / df2 - 1) / delta
    
    def instantaneous_forward(self, t: Union[float, date]) -> float:
        """
        Get instantaneous forward rate f(t).
        
        This is the derivative of the zero rate curve at t.
        f(t) = -d/dt [log P(0,t)]
             = z(t) + t * dz/dt
        
        Args:
            t: Year fraction or date
            
        Returns:
            Instantaneous forward rate
        """
        if isinstance(t, date):
            t = year_fraction(self.anchor_date, t, self.day_count)
        
        if t <= 0:
            return self.zero_rate(0.001)
        
        self._ensure_fitted()
        zr = self._interpolator.interpolate(t)
        dz_dt = self._interpolator.derivative(t)
        
        return zr + t * dz_dt
    
    def get_nodes(self) -> List[Tuple[float, float, float]]:
        """
        Get all curve nodes.
        
        Returns:
            List of (time, discount_factor, zero_rate) tuples
        """
        return [(n.time, n.discount_factor, n.zero_rate) for n in self._nodes]
    
    def get_node_times(self) -> np.ndarray:
        """Get array of node times."""
        return np.array([n.time for n in self._nodes])
    
    def get_node_dfs(self) -> np.ndarray:
        """Get array of node discount factors."""
        return np.array([n.discount_factor for n in self._nodes])
    
    def get_node_rates(self) -> np.ndarray:
        """Get array of node zero rates."""
        return np.array([n.zero_rate for n in self._nodes])
    
    def bump_parallel(self, bp: float) -> "Curve":
        """
        Create a new curve with parallel bump.
        
        Args:
            bp: Bump size in basis points
            
        Returns:
            New bumped curve
        """
        bump = bp / 10000.0  # Convert bp to decimal
        
        new_curve = Curve(
            anchor_date=self.anchor_date,
            currency=self.currency,
            day_count=self.day_count,
            interpolation_method=self.interpolation_method
        )
        
        # Clear the default t=0 node
        new_curve._nodes = []
        
        for node in self._nodes:
            if node.time == 0:
                new_curve._nodes.append(node)
            else:
                new_zr = node.zero_rate + bump
                new_node = CurveNode.from_zero_rate(node.time, new_zr)
                new_curve._nodes.append(new_node)
        
        if len(new_curve._nodes) >= 2:
            new_curve.build()
        
        return new_curve
    
    def bump_node(self, node_index: int, bp: float) -> "Curve":
        """
        Create a new curve with a single node bumped.
        
        Args:
            node_index: Index of node to bump (0-based)
            bp: Bump size in basis points
            
        Returns:
            New bumped curve
        """
        if node_index < 0 or node_index >= len(self._nodes):
            raise IndexError(f"Invalid node index: {node_index}")
        
        bump = bp / 10000.0
        
        new_curve = Curve(
            anchor_date=self.anchor_date,
            currency=self.currency,
            day_count=self.day_count,
            interpolation_method=self.interpolation_method
        )
        
        new_curve._nodes = []
        
        for i, node in enumerate(self._nodes):
            if i == node_index and node.time > 0:
                new_zr = node.zero_rate + bump
                new_node = CurveNode.from_zero_rate(node.time, new_zr)
                new_curve._nodes.append(new_node)
            else:
                new_curve._nodes.append(CurveNode(
                    time=node.time,
                    discount_factor=node.discount_factor,
                    zero_rate=node.zero_rate
                ))
        
        if len(new_curve._nodes) >= 2:
            new_curve.build()
        
        return new_curve
    
    def bump_tenor(self, tenor: str, bp: float) -> "Curve":
        """
        Create a new curve with the node closest to tenor bumped.
        
        Args:
            tenor: Tenor string (e.g., "2Y", "5Y")
            bp: Bump size in basis points
            
        Returns:
            New bumped curve
        """
        target_time = DateUtils.tenor_to_years(tenor)
        
        # Find closest node
        min_dist = float('inf')
        closest_idx = 0
        
        for i, node in enumerate(self._nodes):
            dist = abs(node.time - target_time)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        return self.bump_node(closest_idx, bp)
    
    def copy(self) -> "Curve":
        """Create a deep copy of the curve."""
        new_curve = Curve(
            anchor_date=self.anchor_date,
            currency=self.currency,
            day_count=self.day_count,
            interpolation_method=self.interpolation_method
        )
        
        new_curve._nodes = [
            CurveNode(time=n.time, discount_factor=n.discount_factor, zero_rate=n.zero_rate)
            for n in self._nodes
        ]
        
        if self._is_fitted:
            new_curve.build()
        
        return new_curve
    
    def __repr__(self) -> str:
        return (f"Curve(anchor={self.anchor_date}, currency={self.currency}, "
                f"nodes={len(self._nodes)}, method={self.interpolation_method})")


def create_flat_curve(
    anchor_date: date,
    rate: float,
    max_tenor_years: float = 30.0,
    currency: str = "USD"
) -> Curve:
    """
    Create a flat yield curve.
    
    Args:
        anchor_date: Valuation date
        rate: Flat continuously compounded rate
        max_tenor_years: Maximum tenor in years
        currency: Currency code
        
    Returns:
        Flat curve
    """
    curve = Curve(anchor_date, currency, interpolation_method="linear")
    
    # Add nodes at key tenors
    for t in [0.25, 0.5, 1, 2, 5, 10, 20, max_tenor_years]:
        df = np.exp(-rate * t)
        curve.add_node(t, df)
    
    curve.build()
    return curve


__all__ = [
    "Curve",
    "CurveNode",
    "create_flat_curve",
]
