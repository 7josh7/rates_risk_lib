"""
Portfolio module for trade construction from positions.

This module provides production-grade trade builders with:
- Explicit option fields (no inference)
- Consistent sign conventions
- Comprehensive validation
- Clear error messages
"""

from .builders import (
    # Core trade builders
    build_bond_trade,
    build_swap_trade,
    build_swaption_trade,
    build_caplet_trade,
    build_futures_trade,
    build_trade_from_position,
    # Portfolio pricing with failure tracking
    price_portfolio_with_diagnostics,
    PortfolioPricingResult,
    TradeFailure,
    # Sign convention constants
    SIGN_LONG,
    SIGN_SHORT,
    # Validation errors
    PositionValidationError,
    MissingFieldError,
    InvalidOptionError,
)

__all__ = [
    "build_bond_trade",
    "build_swap_trade",
    "build_swaption_trade",
    "build_caplet_trade",
    "build_futures_trade",
    "build_trade_from_position",
    "price_portfolio_with_diagnostics",
    "PortfolioPricingResult",
    "TradeFailure",
    "SIGN_LONG",
    "SIGN_SHORT",
    "PositionValidationError",
    "MissingFieldError",
    "InvalidOptionError",
]
