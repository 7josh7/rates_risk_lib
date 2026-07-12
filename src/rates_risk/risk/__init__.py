"""
Risk package - sensitivity calculations and risk metrics.

Provides:
- Bump-and-reprice framework
- DV01 and modified duration
- Key-rate/partial durations
- Convexity calculations
"""

from .bumping import (
    BumpEngine,
    BumpType,
    BumpResult,
)
from .sensitivities import (
    RiskCalculator,
    InstrumentRisk,
    PortfolioRisk,
)
from .keyrate import (
    KeyRateEngine,
    KeyRateDV01,
    STANDARD_KEY_RATE_TENORS,
)
from .limits import (
    LimitDefinition,
    LimitResult,
    DEFAULT_LIMITS,
    evaluate_limits,
    limits_to_table,
)
from .reporting import (
    compute_limit_metrics,
    compute_curve_risk_metrics,
    CurveRiskMetrics,
    DEFAULT_KEYRATE_TENORS,
    EXTENDED_KEYRATE_TENORS,
    build_var_portfolio_pricer,
    VaRCoverageInfo,
)

__all__ = [
    "BumpEngine",
    "BumpType",
    "BumpResult",
    "RiskCalculator",
    "InstrumentRisk",
    "PortfolioRisk",
    "KeyRateEngine",
    "KeyRateDV01",
    "STANDARD_KEY_RATE_TENORS",
    "LimitDefinition",
    "LimitResult",
    "DEFAULT_LIMITS",
    "evaluate_limits",
    "limits_to_table",
    "compute_limit_metrics",
    "compute_curve_risk_metrics",
    "CurveRiskMetrics",
    "DEFAULT_KEYRATE_TENORS",
    "EXTENDED_KEYRATE_TENORS",
    "build_var_portfolio_pricer",
    "VaRCoverageInfo",
]
