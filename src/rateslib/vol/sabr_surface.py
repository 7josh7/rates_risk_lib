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
        """
        key = make_bucket_key(expiry, tenor)
        if key in self.params_by_bucket:
            return self.params_by_bucket[key]

        if not allow_fallback or not self.params_by_bucket:
            return None

        if self.missing_bucket_policy != "nearest":
            return None

        # Nearest neighbour by Euclidean distance in (expiry years, tenor years)
        target_expiry = DateUtils.tenor_to_years(expiry)
        target_tenor = DateUtils.tenor_to_years(tenor)

        best_key: Optional[BucketKey] = None
        best_dist = float("inf")

        for candidate in self.params_by_bucket.keys():
            cand_expiry = DateUtils.tenor_to_years(candidate[0])
            cand_tenor = DateUtils.tenor_to_years(candidate[1])
            dist = np.sqrt((cand_expiry - target_expiry) ** 2 + (cand_tenor - target_tenor) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_key = candidate

        if best_key is None:
            return None

        params = self.params_by_bucket[best_key]
        # Annotate that a fallback was used
        params.diagnostics.setdefault("fallback_from", []).append({"requested": key, "used": best_key})
        return params

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


__all__ = ["SabrBucketParams", "SabrSurfaceState", "make_bucket_key", "BucketKey"]
