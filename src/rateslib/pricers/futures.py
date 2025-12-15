"""
Interest rate futures pricing engine.

Prices exchange-traded rate futures with simplified assumptions:
- SOFR futures (CME)
- Fed Funds futures
- Legacy Eurodollar futures (historical)

Key assumptions (documented):
1. Futures rate = Forward rate (ignoring convexity adjustment)
2. Settlement is cash-settled against average rate
3. Contract specifications follow CME standards

Convexity Adjustment Note:
    In reality, futures rates exceed forward rates due to the convexity
    adjustment arising from daily mark-to-market. This library ignores
    this adjustment for simplicity. For precise trading, a convexity
    adjustment model (e.g., Hull-White) should be applied.

Contract Specifications (CME SOFR):
    - Contract size: $1,000,000 notional
    - Tick size: 0.0025 for near months, 0.005 for deferred
    - Tick value: $6.25 or $12.50
    - Settlement: Cash settled against average SOFR
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple

import numpy as np

from ..conventions import DayCount, year_fraction
from ..curves.curve import Curve


@dataclass
class FuturesContract:
    """
    Specification for a rate futures contract.
    
    Attributes:
        contract_code: Exchange code (e.g., "SFR" for SOFR)
        expiry: Contract expiry/settlement date
        contract_size: Notional per contract
        tick_size: Minimum price increment
        tick_value: Dollar value per tick
        underlying_tenor: Underlying rate tenor (e.g., "3M")
    """
    contract_code: str
    expiry: date
    contract_size: float = 1_000_000
    tick_size: float = 0.0025  # 0.25 bp
    tick_value: float = 6.25   # $6.25 per 0.25bp per contract
    underlying_tenor: str = "3M"
    
    @classmethod
    def sofr_3m(cls, expiry: date) -> "FuturesContract":
        """Create CME 3-month SOFR futures specification."""
        return cls(
            contract_code="SFR",
            expiry=expiry,
            contract_size=1_000_000,
            tick_size=0.0025,
            tick_value=6.25,
            underlying_tenor="3M"
        )
    
    @classmethod
    def fed_funds_30d(cls, expiry: date) -> "FuturesContract":
        """Create CME 30-day Fed Funds futures specification."""
        return cls(
            contract_code="FF",
            expiry=expiry,
            contract_size=5_000_000,
            tick_size=0.005,
            tick_value=20.835,
            underlying_tenor="1M"
        )


class FuturesPricer:
    """
    Interest rate futures pricing engine.
    
    Provides:
    - Price from implied rate
    - Implied rate from price
    - DV01 calculation
    - Position P&L
    
    Assumptions:
        - Futures rate = Forward rate (no convexity adjustment)
        - Simple interest rate calculations
    """
    
    def __init__(self, curve: Curve):
        """
        Initialize futures pricer.
        
        Args:
            curve: Yield curve for rate calculations
        """
        self.curve = curve
    
    def implied_rate(
        self,
        expiry: date,
        underlying_tenor: str = "3M"
    ) -> float:
        """
        Calculate implied forward rate for futures.
        
        Args:
            expiry: Futures expiry date
            underlying_tenor: Tenor of underlying rate
            
        Returns:
            Implied rate (decimal)
        """
        from ..dates import DateUtils
        
        t1 = year_fraction(self.curve.anchor_date, expiry, DayCount.ACT_360)
        tenor_years = DateUtils.tenor_to_years(underlying_tenor)
        t2 = t1 + tenor_years
        
        if t1 <= 0:
            # Already expired or spot
            return self.curve.zero_rate(t2)
        
        # Forward rate from curve
        fwd = self.curve.forward_rate(t1, t2)
        
        return fwd
    
    def price_from_rate(self, rate: float) -> float:
        """
        Convert rate to futures price.
        
        Args:
            rate: Rate (decimal)
            
        Returns:
            Futures price (100 - rate%)
        """
        return 100.0 - rate * 100.0
    
    def rate_from_price(self, price: float) -> float:
        """
        Convert futures price to implied rate.
        
        Args:
            price: Futures price
            
        Returns:
            Implied rate (decimal)
        """
        return (100.0 - price) / 100.0
    
    def theoretical_price(
        self,
        contract: FuturesContract
    ) -> float:
        """
        Calculate theoretical futures price.
        
        Args:
            contract: Futures contract specification
            
        Returns:
            Theoretical price
        """
        rate = self.implied_rate(contract.expiry, contract.underlying_tenor)
        return self.price_from_rate(rate)
    
    def dv01(
        self,
        contract: FuturesContract,
        num_contracts: int = 1
    ) -> float:
        """
        Calculate DV01 for futures position.
        
        For 3M futures: DV01 = Notional * 0.25 / 10000 per contract
        
        Args:
            contract: Futures contract
            num_contracts: Number of contracts (positive = long)
            
        Returns:
            DV01 in dollars (positive for long position)
        """
        from ..dates import DateUtils
        
        tenor_years = DateUtils.tenor_to_years(contract.underlying_tenor)
        dv01_per_contract = contract.contract_size * tenor_years / 10000.0
        
        # Long futures gains when rates fall (price rises)
        # So DV01 is negative (value falls when rates rise by 1bp)
        return -num_contracts * dv01_per_contract
    
    def position_pv(
        self,
        contract: FuturesContract,
        num_contracts: int,
        trade_price: float
    ) -> Tuple[float, float, float]:
        """
        Calculate futures position P&L.
        
        Args:
            contract: Futures contract
            num_contracts: Number of contracts (positive = long)
            trade_price: Original trade price
            
        Returns:
            Tuple of (current_price, position_pv, p&l)
        """
        current_price = self.theoretical_price(contract)
        
        # Price change in bp terms
        price_change = current_price - trade_price  # In price points
        
        # P&L = price change / tick_size * tick_value * num_contracts
        pnl = (price_change / contract.tick_size) * contract.tick_value * num_contracts
        
        # Position PV is approximately notional * num_contracts
        # (futures have zero PV at inception by design)
        position_value = pnl  # Futures P&L is the position value
        
        return current_price, position_value, pnl


def price_rate_future(
    curve: Curve,
    expiry: date,
    underlying_tenor: str = "3M"
) -> Tuple[float, float]:
    """
    Price a rate future.
    
    Args:
        curve: Yield curve
        expiry: Futures expiry
        underlying_tenor: Underlying rate tenor
        
    Returns:
        Tuple of (futures_price, implied_rate)
    """
    pricer = FuturesPricer(curve)
    rate = pricer.implied_rate(expiry, underlying_tenor)
    price = pricer.price_from_rate(rate)
    return price, rate


def futures_dv01(
    contract_size: float = 1_000_000,
    tenor_months: int = 3,
    num_contracts: int = 1
) -> float:
    """
    Calculate DV01 for rate futures.
    
    Standard formula: DV01 = Notional * tenor_fraction / 10000
    
    Args:
        contract_size: Notional per contract
        tenor_months: Underlying tenor in months
        num_contracts: Number of contracts
        
    Returns:
        DV01 in dollars (negative for long position)
    """
    tenor_fraction = tenor_months / 12.0
    dv01_per_contract = contract_size * tenor_fraction / 10000.0
    return -num_contracts * dv01_per_contract


__all__ = [
    "FuturesPricer",
    "FuturesContract",
    "price_rate_future",
    "futures_dv01",
]
