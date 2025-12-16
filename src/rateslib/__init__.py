"""
RatesLib: USD Yield Curve Construction & Risk Analytics Library

A modular library for:
- Constructing USD OIS and Treasury yield curves from market quotes
- Pricing core linear rates instruments (bonds, swaps, futures)
- Computing desk-relevant risk metrics (DV01, key-rate, VaR/ES)
- P&L attribution and risk reporting

Scope: USD rates only; linear products only (bonds, vanilla IRS, futures).
"""

__version__ = "0.1.0"

# Core modules
from .conventions import DayCount, BusinessDayConvention, Conventions, year_fraction
from .dates import DateUtils, ScheduleInfo

# Curves
from .curves import (
    Curve,
    OISBootstrapper,
    NelsonSiegelSvensson,
    LinearInterpolator,
    CubicSplineInterpolator,
    LogLinearInterpolator,
)

# Pricers
from .pricers import BondPricer, SwapPricer, FuturesPricer, FuturesContract

# Risk
from .risk import (
    BumpEngine,
    RiskCalculator,
    KeyRateEngine,
    InstrumentRisk,
    PortfolioRisk,
    LimitLevel,
    LimitValue,
    RiskLimits,
    LimitCheckResult,
    RiskLimitChecker,
)

# VaR
from .var import (
    HistoricalSimulation,
    MonteCarloVaR,
    StressedVaR,
    ScenarioEngine,
    STANDARD_SCENARIOS,
)

# P&L
from .pnl import PnLAttributionEngine, PnLAttribution

# Reporting
from .reporting import RiskReport, ReportFormatter, export_to_csv

# Liquidity
from .liquidity import LiquidityEngine, LiquidityAdjustedVaR

# Market State
from .market_state import (
    CurveState,
    SabrSurface,
    MarketState,
    build_sabr_surface_from_quotes,
)

# Volatility (SABR)
from .vol import (
    SabrParams,
    SabrModel,
    SabrCalibrator,
    VolQuote,
    load_vol_quotes,
    hagan_black_vol,
    hagan_normal_vol,
)

# Options
from .options import (
    bachelier_call,
    bachelier_put,
    black76_call,
    black76_put,
    shifted_black_call,
    shifted_black_put,
    bachelier_greeks,
    black76_greeks,
    CapletPricer,
    SwaptionPricer,
    SabrOptionRisk,
)

__all__ = [
    # Version
    "__version__",
    # Conventions
    "DayCount",
    "BusinessDayConvention",
    "Conventions",
    "year_fraction",
    # Dates
    "DateUtils",
    "ScheduleInfo",
    # Curves
    "Curve",
    "OISBootstrapper",
    "NelsonSiegelSvensson",
    "LinearInterpolator",
    "CubicSplineInterpolator",
    "LogLinearInterpolator",
    # Pricers
    "BondPricer",
    "SwapPricer",
    "FuturesPricer",
    "FuturesContract",
    # Risk
    "BumpEngine",
    "RiskCalculator",
    "KeyRateEngine",
    "InstrumentRisk",
    "PortfolioRisk",
    "LimitLevel",
    "LimitValue",
    "RiskLimits",
    "LimitCheckResult",
    "RiskLimitChecker",
    # VaR
    "HistoricalSimulation",
    "MonteCarloVaR",
    "StressedVaR",
    "ScenarioEngine",
    "STANDARD_SCENARIOS",
    # P&L
    "PnLAttributionEngine",
    "PnLAttribution",
    # Reporting
    "RiskReport",
    "ReportFormatter",
    "export_to_csv",
    # Liquidity
    "LiquidityEngine",
    "LiquidityAdjustedVaR",
    # Market State
    "CurveState",
    "SabrSurface",
    "MarketState",
    "build_sabr_surface_from_quotes",
    # Volatility (SABR)
    "SabrParams",
    "SabrModel",
    "SabrCalibrator",
    "VolQuote",
    "load_vol_quotes",
    "hagan_black_vol",
    "hagan_normal_vol",
    # Options
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

