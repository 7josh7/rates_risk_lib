"""
Pricers package - instrument pricing.

Provides pricing engines for:
- Zero-coupon and coupon bonds
- Vanilla fixed-float interest rate swaps
- Interest rate futures
"""

from .bonds import (
    BondPricer,
    BondCashflows,
    price_zero_coupon_bond,
    price_coupon_bond,
    compute_accrued_interest,
)
from .swaps import (
    SwapPricer,
    SwapCashflows,
    price_vanilla_swap,
    compute_swap_par_rate,
)
from .futures import (
    FuturesPricer,
    FuturesContract,
    price_rate_future,
)
from .dispatcher import price_trade, risk_trade, PricerOutput

__all__ = [
    "BondPricer",
    "BondCashflows",
    "price_zero_coupon_bond",
    "price_coupon_bond",
    "compute_accrued_interest",
    "SwapPricer",
    "SwapCashflows",
    "price_vanilla_swap",
    "compute_swap_par_rate",
    "FuturesPricer",
    "FuturesContract",
    "price_rate_future",
    "price_trade",
    "risk_trade",
    "PricerOutput",
]
