"""
Risk sensitivity calculations.

Computes standard risk metrics for instruments and portfolios:
- DV01 (dollar value of 1bp)
- Modified duration
- Convexity
- Key-rate durations

Output format follows desk conventions for risk reporting.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..curves.curve import Curve
from .bumping import BumpEngine


@dataclass
class InstrumentRisk:
    """
    Risk metrics for a single instrument.
    
    Attributes:
        instrument_id: Unique identifier
        instrument_type: Type (BOND, IRS, FUT, etc.)
        pv: Present value
        notional: Notional/face amount
        dv01: Dollar value of 1bp parallel shift
        modified_duration: Modified duration (years)
        convexity: Dollar convexity
        key_rate_dv01: Dict of {tenor: dv01}
    """
    instrument_id: str
    instrument_type: str
    pv: float
    notional: float
    dv01: float
    modified_duration: float
    convexity: float
    key_rate_dv01: Dict[str, float] = field(default_factory=dict)
    
    @property
    def dv01_per_million(self) -> float:
        """DV01 per million notional."""
        if self.notional == 0:
            return 0.0
        return self.dv01 * 1_000_000 / abs(self.notional)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "instrument_id": self.instrument_id,
            "instrument_type": self.instrument_type,
            "pv": self.pv,
            "notional": self.notional,
            "dv01": self.dv01,
            "modified_duration": self.modified_duration,
            "convexity": self.convexity,
            "key_rate_dv01": self.key_rate_dv01
        }


@dataclass
class PortfolioRisk:
    """
    Aggregated risk metrics for a portfolio.
    
    Attributes:
        as_of_date: Risk calculation date
        total_pv: Sum of instrument PVs
        total_dv01: Sum of DV01s
        total_convexity: Sum of dollar convexities
        key_rate_dv01: Aggregated key-rate DV01
        instrument_risks: List of individual instrument risks
    """
    as_of_date: date
    total_pv: float
    total_dv01: float
    total_convexity: float
    key_rate_dv01: Dict[str, float]
    instrument_risks: List[InstrumentRisk]
    
    @property
    def weighted_duration(self) -> float:
        """PV-weighted average duration."""
        if self.total_pv == 0:
            return 0.0
        weighted = sum(r.pv * r.modified_duration for r in self.instrument_risks)
        return weighted / self.total_pv
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Summary dictionary for reporting."""
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "total_pv": self.total_pv,
            "total_dv01": self.total_dv01,
            "total_convexity": self.total_convexity,
            "weighted_duration": self.weighted_duration,
            "key_rate_dv01": self.key_rate_dv01,
            "num_instruments": len(self.instrument_risks)
        }


class RiskCalculator:
    """
    Calculator for instrument and portfolio risk metrics.
    
    Supports:
    - Bonds
    - Swaps
    - Futures
    
    All calculations use bump-and-reprice methodology.
    """
    
    # Standard key rate tenors
    KEY_RATE_TENORS = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    
    def __init__(
        self,
        curve: Curve,
        bump_size: float = 1.0
    ):
        """
        Initialize risk calculator.
        
        Args:
            curve: Yield curve for calculations
            bump_size: Bump size in bp for sensitivities
        """
        self.curve = curve
        self.bump_size = bump_size
        self.bump_engine = BumpEngine(curve)
    
    def compute_bond_risk(
        self,
        instrument_id: str,
        settlement: date,
        maturity: date,
        coupon_rate: float,
        notional: float,
        frequency: int = 2
    ) -> InstrumentRisk:
        """
        Compute risk for a bond.
        
        Args:
            instrument_id: Bond identifier
            settlement: Settlement date
            maturity: Maturity date
            coupon_rate: Annual coupon rate
            notional: Face/notional (positive=long, negative=short)
            frequency: Coupon frequency
            
        Returns:
            InstrumentRisk object
        """
        from ..pricers.bonds import BondPricer, Conventions
        from ..conventions import DayCount
        
        conventions = Conventions(
            day_count=DayCount.ACT_ACT,
            payment_frequency=frequency
        )
        
        def pricer_func(curve: Curve) -> float:
            pricer = BondPricer(curve, conventions)
            dirty, _, _ = pricer.price(settlement, maturity, coupon_rate, 100.0, frequency)
            return dirty * notional / 100.0
        
        # Base PV
        pv = pricer_func(self.curve)
        
        # DV01
        dv01 = self.bump_engine.compute_dv01(pricer_func, self.bump_size)
        
        # Convexity
        convexity = self.bump_engine.compute_convexity(pricer_func, self.bump_size)
        
        # Modified duration
        if pv != 0:
            mod_duration = -dv01 * 10000 / pv
        else:
            mod_duration = 0.0
        
        # Key-rate DV01
        kr_dv01 = self._compute_key_rate_dv01(pricer_func)
        
        return InstrumentRisk(
            instrument_id=instrument_id,
            instrument_type="BOND",
            pv=pv,
            notional=notional,
            dv01=dv01,
            modified_duration=mod_duration,
            convexity=convexity,
            key_rate_dv01=kr_dv01
        )
    
    def compute_swap_risk(
        self,
        instrument_id: str,
        effective: date,
        maturity: date,
        fixed_rate: float,
        notional: float,
        pay_receive: str = "PAY"
    ) -> InstrumentRisk:
        """
        Compute risk for an interest rate swap.
        
        Args:
            instrument_id: Swap identifier
            effective: Effective date
            maturity: Maturity date
            fixed_rate: Fixed rate
            notional: Notional (absolute value)
            pay_receive: "PAY" or "RECEIVE" fixed
            
        Returns:
            InstrumentRisk object
        """
        from ..pricers.swaps import SwapPricer
        
        def pricer_func(curve: Curve) -> float:
            pricer = SwapPricer(curve)
            return pricer.present_value(effective, maturity, abs(notional), fixed_rate, pay_receive)
        
        pv = pricer_func(self.curve)
        dv01 = self.bump_engine.compute_dv01(pricer_func, self.bump_size)
        convexity = self.bump_engine.compute_convexity(pricer_func, self.bump_size)
        
        if pv != 0:
            mod_duration = -dv01 * 10000 / pv
        else:
            mod_duration = 0.0
        
        kr_dv01 = self._compute_key_rate_dv01(pricer_func)
        
        return InstrumentRisk(
            instrument_id=instrument_id,
            instrument_type="IRS",
            pv=pv,
            notional=notional,
            dv01=dv01,
            modified_duration=mod_duration,
            convexity=convexity,
            key_rate_dv01=kr_dv01
        )
    
    def compute_futures_risk(
        self,
        instrument_id: str,
        expiry: date,
        num_contracts: int,
        underlying_tenor: str = "3M",
        contract_size: float = 1_000_000
    ) -> InstrumentRisk:
        """
        Compute risk for interest rate futures.
        
        Args:
            instrument_id: Futures identifier
            expiry: Expiry date
            num_contracts: Number of contracts (positive=long)
            underlying_tenor: Underlying rate tenor
            contract_size: Contract size
            
        Returns:
            InstrumentRisk object
        """
        from ..pricers.futures import FuturesPricer, FuturesContract
        from ..dates import DateUtils
        
        contract = FuturesContract(
            contract_code="FUT",
            expiry=expiry,
            contract_size=contract_size,
            underlying_tenor=underlying_tenor
        )
        
        pricer = FuturesPricer(self.curve)
        
        # Futures have zero PV by design (margin)
        pv = 0.0
        
        # DV01 for futures
        tenor_years = DateUtils.tenor_to_years(underlying_tenor)
        dv01 = -num_contracts * contract_size * tenor_years / 10000.0
        
        # Duration not meaningful for futures, convexity is minimal
        mod_duration = 0.0
        convexity = 0.0
        
        # Key-rate: all exposure at one tenor
        kr_dv01 = {t: 0.0 for t in self.KEY_RATE_TENORS}
        
        # Find closest key rate tenor
        from ..dates import DateUtils
        expiry_years = (expiry - self.curve.anchor_date).days / 365.0
        closest_tenor = min(self.KEY_RATE_TENORS, 
                          key=lambda t: abs(DateUtils.tenor_to_years(t) - expiry_years))
        kr_dv01[closest_tenor] = dv01
        
        return InstrumentRisk(
            instrument_id=instrument_id,
            instrument_type="FUT",
            pv=pv,
            notional=num_contracts * contract_size,
            dv01=dv01,
            modified_duration=mod_duration,
            convexity=convexity,
            key_rate_dv01=kr_dv01
        )
    
    def _compute_key_rate_dv01(
        self,
        pricer_func: Callable[[Curve], float]
    ) -> Dict[str, float]:
        """
        Compute key-rate DV01 for all standard tenors.
        
        Args:
            pricer_func: Pricing function
            
        Returns:
            Dict of {tenor: dv01}
        """
        pv_base = pricer_func(self.curve)
        kr_dv01 = {}
        
        for tenor in self.KEY_RATE_TENORS:
            try:
                bumped = self.curve.bump_tenor(tenor, self.bump_size)
                pv_bumped = pricer_func(bumped)
                kr_dv01[tenor] = (pv_bumped - pv_base) / self.bump_size
            except:
                kr_dv01[tenor] = 0.0
        
        return kr_dv01
    
    def aggregate_portfolio(
        self,
        instrument_risks: List[InstrumentRisk],
        as_of_date: date
    ) -> PortfolioRisk:
        """
        Aggregate individual instrument risks into portfolio risk.
        
        Args:
            instrument_risks: List of instrument risks
            as_of_date: Risk date
            
        Returns:
            PortfolioRisk object
        """
        total_pv = sum(r.pv for r in instrument_risks)
        total_dv01 = sum(r.dv01 for r in instrument_risks)
        total_convexity = sum(r.convexity for r in instrument_risks)
        
        # Aggregate key-rate DV01
        kr_dv01 = {t: 0.0 for t in self.KEY_RATE_TENORS}
        for r in instrument_risks:
            for tenor, dv01 in r.key_rate_dv01.items():
                if tenor in kr_dv01:
                    kr_dv01[tenor] += dv01
        
        return PortfolioRisk(
            as_of_date=as_of_date,
            total_pv=total_pv,
            total_dv01=total_dv01,
            total_convexity=total_convexity,
            key_rate_dv01=kr_dv01,
            instrument_risks=instrument_risks
        )


def compute_dv01(
    curve: Curve,
    pricer_func: Callable[[Curve], float],
    bump_size: float = 1.0
) -> float:
    """
    Convenience function to compute DV01.
    
    Args:
        curve: Base curve
        pricer_func: Pricing function
        bump_size: Bump in bp
        
    Returns:
        DV01
    """
    engine = BumpEngine(curve)
    return engine.compute_dv01(pricer_func, bump_size)


def compute_duration_convexity(
    curve: Curve,
    pricer_func: Callable[[Curve], float],
    bump_size: float = 1.0
) -> Tuple[float, float]:
    """
    Compute modified duration and convexity.
    
    Args:
        curve: Base curve
        pricer_func: Pricing function
        bump_size: Bump in bp
        
    Returns:
        Tuple of (modified_duration, convexity)
    """
    engine = BumpEngine(curve)
    pv = pricer_func(curve)
    dv01 = engine.compute_dv01(pricer_func, bump_size)
    convexity = engine.compute_convexity(pricer_func, bump_size)
    
    if pv != 0:
        mod_duration = -dv01 * 10000 / pv
    else:
        mod_duration = 0.0
    
    return mod_duration, convexity


__all__ = [
    "RiskCalculator",
    "InstrumentRisk",
    "PortfolioRisk",
    "compute_dv01",
    "compute_duration_convexity",
]
