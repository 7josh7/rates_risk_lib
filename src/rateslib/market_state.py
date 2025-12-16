"""
Combined market state for pricing and scenarios.

Encapsulates curve and SABR surface states so pricing/risk functions can
accept a single object instead of juggling multiple inputs.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Optional

from .curves.curve import Curve
from .vol.sabr_surface import SabrSurfaceState, SabrBucketParams


@dataclass
class CurveState:
    """
    Container for discount and projection curves.

    projection_curve defaults to discount_curve for convenience.
    """

    discount_curve: Curve
    projection_curve: Optional[Curve] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.projection_curve is None:
            self.projection_curve = self.discount_curve

    def forward_rate(self, t_start: float, t_end: float) -> float:
        """Simple forward rate helper using the projection curve."""
        curve = self.projection_curve or self.discount_curve
        return curve.forward_rate(t_start, t_end)

    def copy(self, discount_curve: Optional[Curve] = None, projection_curve: Optional[Curve] = None) -> "CurveState":
        """Create a shallow copy with optional curve overrides."""
        return CurveState(
            discount_curve=discount_curve or self.discount_curve,
            projection_curve=projection_curve or self.projection_curve,
            metadata=dict(self.metadata),
        )


@dataclass
class MarketState:
    """
    Combined market state used for repricing and scenarios.
    """

    curve: CurveState
    sabr_surface: Optional[SabrSurfaceState] = None
    asof: Any = field(default_factory=datetime.utcnow)

    def get_sabr_params(self, expiry: str, tenor: str, allow_fallback: bool = True) -> Optional[SabrBucketParams]:
        """
        Convenience accessor for SABR parameters by bucket.
        """
        if self.sabr_surface is None:
            return None
        return self.sabr_surface.get_bucket_params(expiry, tenor, allow_fallback=allow_fallback)

    def copy(
        self,
        curve: Optional[CurveState] = None,
        sabr_surface: Optional[SabrSurfaceState] = None,
        asof: Any = None,
    ) -> "MarketState":
        """Shallow copy with optional overrides."""
        return MarketState(
            curve=curve or self.curve,
            sabr_surface=sabr_surface if sabr_surface is not None else self.sabr_surface,
            asof=asof or self.asof,
        )


__all__ = ["CurveState", "MarketState"]
