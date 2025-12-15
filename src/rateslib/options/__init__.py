"""
Options module - Rates option pricing.

Provides:
- Caplet/floorlet pricing
- Swaption pricing
- Bachelier (normal) and Black'76 models
- SABR-integrated pricing and risk
"""

from .base_models import (
    bachelier_call,
    bachelier_put,
    black76_call,
    black76_put,
    shifted_black_call,
    shifted_black_put,
    bachelier_greeks,
    black76_greeks,
)
from .caplet import CapletPricer
from .swaption import SwaptionPricer
from .sabr_risk import SabrOptionRisk

__all__ = [
    "bachelier_call",
    "bachelier_put",
    "black76_call",
    "black76_put",
    "shifted_black_call",
    "shifted_black_put",
    "bachelier_greeks",
    "black76_greeks",
    "CapletPricer",
    "SwaptionPricer",
    "SabrOptionRisk",
]
