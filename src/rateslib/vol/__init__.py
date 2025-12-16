"""
Volatility module - SABR model and calibration.

Provides:
- SABR stochastic volatility model
- Hagan implied volatility approximation
- Calibration from market vol quotes
- Greeks and risk sensitivities
"""

from .sabr import (
    SabrParams,
    SabrModel,
    hagan_black_vol,
    hagan_normal_vol,
)
from .calibration import SabrCalibrator, calibrate_sabr_bucket, build_sabr_surface
from .quotes import VolQuote, load_vol_quotes, normalize_vol_quotes
from .sabr_surface import SabrSurfaceState, SabrBucketParams, make_bucket_key

__all__ = [
    "SabrParams",
    "SabrModel",
    "SabrCalibrator",
    "VolQuote",
    "load_vol_quotes",
    "normalize_vol_quotes",
    "SabrSurfaceState",
    "SabrBucketParams",
    "make_bucket_key",
    "calibrate_sabr_bucket",
    "build_sabr_surface",
    "hagan_black_vol",
    "hagan_normal_vol",
]
