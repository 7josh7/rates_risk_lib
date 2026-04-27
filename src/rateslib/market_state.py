"""
Combined market state for pricing and scenarios.

Encapsulates curve and SABR surface states so pricing/risk functions can
accept a single object instead of juggling multiple inputs.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, UTC
from typing import Any, Dict, Optional

from .curves.curve import Curve
from .domain import PricingPolicy
from .vol.sabr_surface import SabrSurfaceState, SabrBucketParams, SabrLookupResult


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
    asof: Any = field(default_factory=lambda: datetime.now(UTC))
    pricing_policy: PricingPolicy = field(default_factory=PricingPolicy)

    def resolve_sabr_lookup(
        self,
        expiry: str,
        tenor: str,
        allow_fallback: bool = True,
    ) -> Optional[SabrLookupResult]:
        """
        Resolve a SABR lookup while respecting the pricing policy.
        """
        if self.sabr_surface is None:
            return None
        effective_allow_fallback = (
            allow_fallback and self.pricing_policy.sabr_bucket_fallback != "error"
        )
        return self.sabr_surface.resolve_bucket(
            expiry,
            tenor,
            allow_fallback=effective_allow_fallback,
        )

    def get_sabr_params(self, expiry: str, tenor: str, allow_fallback: bool = True) -> Optional[SabrBucketParams]:
        """
        Convenience accessor for SABR parameters by bucket.
        """
        lookup = self.resolve_sabr_lookup(
            expiry,
            tenor,
            allow_fallback=allow_fallback,
        )
        if lookup is None:
            return None
        return self.sabr_surface.get_bucket_params(
            expiry,
            tenor,
            allow_fallback=allow_fallback and self.pricing_policy.sabr_bucket_fallback != "error",
        )

    def copy(
        self,
        curve: Optional[CurveState] = None,
        sabr_surface: Optional[SabrSurfaceState] = None,
        asof: Any = None,
        pricing_policy: Optional[PricingPolicy] = None,
    ) -> "MarketState":
        """Shallow copy with optional overrides."""
        return MarketState(
            curve=curve or self.curve,
            sabr_surface=sabr_surface if sabr_surface is not None else self.sabr_surface,
            asof=asof or self.asof,
            pricing_policy=pricing_policy or self.pricing_policy,
        )


__all__ = ["CurveState", "MarketState"]
