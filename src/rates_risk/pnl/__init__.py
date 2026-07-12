"""
P&L package - attribution and explain.

Provides daily P&L attribution splitting:
- Carry and roll-down
- Curve move (parallel and non-parallel)
- Convexity effect
- Residual/unexplained
"""

from .attribution import (
    PnLAttribution,
    PnLAttributionEngine,
    PnLComponents,
    compute_daily_pnl,
    compute_carry_rolldown,
)

__all__ = [
    "PnLAttribution",
    "PnLAttributionEngine",
    "PnLComponents",
    "compute_daily_pnl",
    "compute_carry_rolldown",
]
