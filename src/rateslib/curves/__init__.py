"""
Curves package - yield curve construction and manipulation.

Provides:
- Curve: Main curve object with discount factors and interpolation
- OISBootstrapper: Bootstrap OIS curves from market quotes
- NelsonSiegelSvensson: Parametric curve fitting
"""

from .curve import Curve, create_flat_curve
from .bootstrap import OISBootstrapper, BootstrapResult, bootstrap_from_quotes
from .nss import NelsonSiegelSvensson
from .interpolation import (
    Interpolator,
    LinearInterpolator,
    CubicSplineInterpolator,
    LogLinearInterpolator
)
from .instruments import (
    CurveInstrument,
    Deposit,
    OISSwap,
    FRA,
    Future
)

__all__ = [
    "Curve",
    "create_flat_curve",
    "OISBootstrapper",
    "BootstrapResult",
    "bootstrap_from_quotes",
    "NelsonSiegelSvensson",
    "Interpolator",
    "LinearInterpolator",
    "CubicSplineInterpolator",
    "LogLinearInterpolator",
    "CurveInstrument",
    "Deposit",
    "OISSwap",
    "FRA",
    "Future",
]
