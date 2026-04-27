"""
SABR surface state and helpers.

Provides a light-weight container for bucketed SABR parameters along with
diagnostics required by the dashboard and risk/reporting layers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .sabr import SabrParams
from ..dates import DateUtils

BucketKey = Tuple[str, str]


def make_bucket_key(expiry: str, tenor: str) -> BucketKey:
    """
    Normalise a SABR bucket key to uppercase strings.
    """
    return (str(expiry).upper(), str(tenor).upper())


def _bucket_axis_to_years(value: Any) -> float:
    """Convert either a tenor label or numeric year value to years."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped)
        except ValueError:
            return DateUtils.tenor_to_years(stripped)
    return DateUtils.tenor_to_years(value)


@dataclass
class SabrBucketParams:
    """SABR parameters and diagnostics for a single bucket."""

    sigma_atm: float
    nu: float
    rho: float
    beta: float
    shift: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_sabr_params(self) -> SabrParams:
        """Convert to SabrParams used by pricers."""
        return SabrParams(
            sigma_atm=self.sigma_atm,
            beta=self.beta,
            rho=self.rho,
            nu=self.nu,
            shift=self.shift,
        )


@dataclass(frozen=True)
class SabrLookupResult:
    """Structured result for SABR bucket resolution."""

    requested_bucket: BucketKey
    used_bucket: Optional[BucketKey]
    params: Optional[SabrBucketParams]
    used_fallback: bool = False
    reason: str = ""


@dataclass
class SabrSurfaceState:
    """
    Bucketed SABR surface calibrated per (expiry x tenor).

    Attributes:
        params_by_bucket: Mapping of bucket key -> SabrBucketParams
        convention: Metadata such as vol_type, beta policy, shift policy
        asof: Calibration timestamp (string or date for portability)
    """

    params_by_bucket: Dict[BucketKey, SabrBucketParams]
    convention: Dict[str, Any] = field(default_factory=dict)
    asof: Optional[str] = None
    missing_bucket_policy: str = "nearest"

    def get_bucket_params(
        self,
        expiry: str,
        tenor: str,
        allow_fallback: bool = True,
    ) -> Optional[SabrBucketParams]:
        """
        Retrieve parameters for the given bucket.

        If no exact bucket exists and allow_fallback=True, uses the nearest
        bucket in (expiry, tenor) space. This keeps pricing/risk stable while
        still surfacing missing bucket information to the caller.
        
        Special handling for ("ALL", "ALL") bucket which acts as a wildcard.
        """
        lookup = self.resolve_bucket(expiry, tenor, allow_fallback=allow_fallback)
        if lookup is None or lookup.params is None:
            return None

        params = lookup.params
        if lookup.used_fallback and lookup.used_bucket is not None:
            params.diagnostics.setdefault("fallback_from", []).append(
                {"requested": lookup.requested_bucket, "used": lookup.used_bucket}
            )
        return params

    def resolve_bucket(
        self,
        expiry: str,
        tenor: str,
        allow_fallback: bool = True,
    ) -> Optional[SabrLookupResult]:
        """
        Resolve a requested bucket without mutating diagnostics state.

        This is useful for pricing paths that need to distinguish between an
        exact bucket hit and a fallback.
        """
        requested = make_bucket_key(expiry, tenor)
        if requested in self.params_by_bucket:
            return SabrLookupResult(
                requested_bucket=requested,
                used_bucket=requested,
                params=self.params_by_bucket[requested],
                used_fallback=False,
                reason="exact",
            )

        if not allow_fallback or not self.params_by_bucket:
            return SabrLookupResult(
                requested_bucket=requested,
                used_bucket=None,
                params=None,
                used_fallback=False,
                reason="fallback_disabled_or_empty_surface",
            )

        all_key = ("ALL", "ALL")
        if all_key in self.params_by_bucket:
            return SabrLookupResult(
                requested_bucket=requested,
                used_bucket=all_key,
                params=self.params_by_bucket[all_key],
                used_fallback=True,
                reason="wildcard",
            )

        if self.missing_bucket_policy != "nearest":
            return SabrLookupResult(
                requested_bucket=requested,
                used_bucket=None,
                params=None,
                used_fallback=False,
                reason=f"unsupported_missing_bucket_policy:{self.missing_bucket_policy}",
            )

        target_expiry = _bucket_axis_to_years(expiry)
        target_tenor = _bucket_axis_to_years(tenor)

        best_key: Optional[BucketKey] = None
        best_dist = float("inf")

        for candidate in self.params_by_bucket.keys():
            if "ALL" in candidate:
                continue
            cand_expiry = _bucket_axis_to_years(candidate[0])
            cand_tenor = _bucket_axis_to_years(candidate[1])
            dist = np.sqrt((cand_expiry - target_expiry) ** 2 + (cand_tenor - target_tenor) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_key = candidate

        if best_key is None:
            return SabrLookupResult(
                requested_bucket=requested,
                used_bucket=None,
                params=None,
                used_fallback=False,
                reason="no_candidate_bucket",
            )

        return SabrLookupResult(
            requested_bucket=requested,
            used_bucket=best_key,
            params=self.params_by_bucket[best_key],
            used_fallback=True,
            reason="nearest",
        )

    def diagnostics_table(self) -> Dict[BucketKey, Dict[str, Any]]:
        """
        Return diagnostics keyed by bucket for downstream reporting.
        """
        table: Dict[BucketKey, Dict[str, Any]] = {}
        for bucket, params in self.params_by_bucket.items():
            table[bucket] = {
                "sigma_atm": params.sigma_atm,
                "nu": params.nu,
                "rho": params.rho,
                "beta": params.beta,
                "shift": params.shift,
                **params.diagnostics,
            }
        return table


__all__ = [
    "SabrBucketParams",
    "SabrLookupResult",
    "SabrSurfaceState",
    "make_bucket_key",
    "BucketKey",
]
