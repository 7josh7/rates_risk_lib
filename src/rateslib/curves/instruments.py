"""
Curve instruments for bootstrapping.

Defines the instruments used to build yield curves:
- Deposit: Overnight to short-term deposits
- OISSwap: Overnight Index Swaps
- FRA: Forward Rate Agreements
- Future: Interest rate futures

Each instrument knows how to:
1. Calculate its maturity from trade date
2. Price itself given a curve
3. Imply the discount factor that makes it price to its quote
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

import numpy as np

from ..conventions import DayCount, year_fraction, BusinessDayConvention
from ..dates import DateUtils, adjust_business_day


@dataclass
class CurveInstrument(ABC):
    """
    Abstract base for curve construction instruments.
    
    Attributes:
        tenor: Instrument tenor (e.g., "3M", "2Y")
        quote: Market quote (rate in decimal)
        quote_type: Type of quote ("RATE", "PRICE")
        day_count: Day count convention
    """
    tenor: str
    quote: float
    quote_type: str = "RATE"
    day_count: DayCount = DayCount.ACT_360
    
    @abstractmethod
    def maturity_date(self, anchor: date) -> date:
        """Calculate maturity date from anchor."""
        pass
    
    @abstractmethod
    def maturity_time(self, anchor: date) -> float:
        """Calculate time to maturity in years."""
        pass
    
    @abstractmethod
    def implied_discount_factor(
        self, 
        anchor: date, 
        prior_dfs: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Calculate implied discount factor from quote.
        
        Args:
            anchor: Curve anchor date
            prior_dfs: List of (time, df) pairs for earlier maturities
            
        Returns:
            Tuple of (time, discount_factor)
        """
        pass


@dataclass
class Deposit(CurveInstrument):
    """
    Money market deposit.
    
    Simple interest instrument: the depositor receives (1 + R*tau) at maturity.
    
    Pricing: DF(T) = 1 / (1 + R * tau)
    where tau is the year fraction using the day count convention.
    """
    
    def maturity_date(self, anchor: date) -> date:
        """Get maturity date."""
        return DateUtils.add_tenor(anchor, self.tenor)
    
    def maturity_time(self, anchor: date) -> float:
        """Get time to maturity in years."""
        mat = self.maturity_date(anchor)
        return year_fraction(anchor, mat, self.day_count)
    
    def implied_discount_factor(
        self, 
        anchor: date,
        prior_dfs: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Implied DF from deposit rate.
        
        For a deposit starting at anchor, the discount factor is:
        DF(T) = 1 / (1 + R * tau)
        """
        mat = self.maturity_date(anchor)
        tau = year_fraction(anchor, mat, self.day_count)
        
        if tau <= 0:
            return (0.0, 1.0)
        
        df = 1.0 / (1.0 + self.quote * tau)
        return (tau, df)


@dataclass
class OISSwap(CurveInstrument):
    """
    Overnight Index Swap.
    
    Fixed leg pays fixed rate K at each payment date.
    Floating leg pays compounded overnight rate.
    
    Single-curve pricing (OIS discount = OIS forward):
    Par swap rate: R = (1 - DF(Tn)) / sum(delta_i * DF(Ti))
    
    Bootstrap: solve for DF(Tn) given R and earlier DFs.
    """
    payment_frequency: str = "ANNUAL"  # ANNUAL, SEMI, QUARTERLY
    
    def maturity_date(self, anchor: date) -> date:
        """Get maturity date."""
        return DateUtils.add_tenor(anchor, self.tenor)
    
    def maturity_time(self, anchor: date) -> float:
        """Get time to maturity."""
        mat = self.maturity_date(anchor)
        return year_fraction(anchor, mat, self.day_count)
    
    def _payment_schedule(self, anchor: date) -> List[Tuple[date, float]]:
        """
        Generate payment schedule with accrual fractions.
        
        Returns list of (payment_date, year_fraction) tuples.
        """
        mat = self.maturity_date(anchor)
        
        # Determine payment frequency
        freq_map = {"ANNUAL": 1, "SEMI": 2, "QUARTERLY": 4, "MONTHLY": 12}
        freq = freq_map.get(self.payment_frequency.upper(), 1)
        
        months_per_period = 12 // freq
        
        # Generate payment dates backward from maturity
        payments = []
        current = mat
        
        while current > anchor:
            payments.insert(0, current)
            # Go back by period
            year = current.year
            month = current.month - months_per_period
            while month <= 0:
                month += 12
                year -= 1
            day = min(mat.day, _days_in_month(year, month))
            current = date(year, month, day)
        
        # Calculate accrual fractions
        result = []
        prev = anchor
        for pmt in payments:
            tau = year_fraction(prev, pmt, self.day_count)
            result.append((pmt, tau))
            prev = pmt
        
        return result
    
    def implied_discount_factor(
        self,
        anchor: date,
        prior_dfs: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Implied DF from OIS swap rate.
        
        Swap rate: R = (1 - DF(Tn)) / sum(delta_i * DF(Ti))
        Rearranging: DF(Tn) = (1 - R * sum(delta_i * DF(Ti))) / (1 + R * delta_n)
        
        For intermediate payment dates, we need to interpolate/bootstrap.
        """
        schedule = self._payment_schedule(anchor)
        
        if not schedule:
            return (0.0, 1.0)
        
        # Build lookup from prior DFs
        df_lookup = {t: df for t, df in prior_dfs}
        df_lookup[0.0] = 1.0
        
        R = self.quote
        
        # For single payment (short-dated), simple case
        if len(schedule) == 1:
            pmt_date, tau = schedule[0]
            t = year_fraction(anchor, pmt_date, self.day_count)
            df = 1.0 / (1.0 + R * tau)
            return (t, df)
        
        # For multi-period, accumulate PV of fixed leg
        # and solve for final DF
        pv_fixed_known = 0.0
        final_tau = 0.0
        final_t = 0.0
        
        for i, (pmt_date, tau) in enumerate(schedule):
            t = year_fraction(anchor, pmt_date, self.day_count)
            
            if i < len(schedule) - 1:
                # Intermediate payment - need DF from prior
                # Interpolate if necessary
                df = _interpolate_df(t, prior_dfs)
                pv_fixed_known += R * tau * df
            else:
                # Final payment - this is what we solve for
                final_tau = tau
                final_t = t
        
        # Par swap: 1 - DF(Tn) = R * sum(delta_i * DF(Ti))
        # DF(Tn) = (1 - pv_fixed_known) / (1 + R * final_tau)
        numerator = 1.0 - pv_fixed_known
        denominator = 1.0 + R * final_tau
        
        df_final = numerator / denominator
        
        return (final_t, df_final)


@dataclass
class FRA(CurveInstrument):
    """
    Forward Rate Agreement.
    
    FRA pays at the start of the forward period based on the
    difference between the agreed rate and the fixing rate.
    
    FRA rate: F = (DF(T1)/DF(T2) - 1) / tau
    """
    start_tenor: str = ""  # When the forward period starts (e.g., "3M")
    
    def __post_init__(self):
        if not self.start_tenor:
            # Try to parse from tenor format like "3x6" or "3M6M"
            # Default: start immediately
            self.start_tenor = "0D"
    
    def maturity_date(self, anchor: date) -> date:
        """End of forward period."""
        start = DateUtils.add_tenor(anchor, self.start_tenor)
        return DateUtils.add_tenor(start, self.tenor)
    
    def start_date(self, anchor: date) -> date:
        """Start of forward period."""
        return DateUtils.add_tenor(anchor, self.start_tenor)
    
    def maturity_time(self, anchor: date) -> float:
        mat = self.maturity_date(anchor)
        return year_fraction(anchor, mat, self.day_count)
    
    def start_time(self, anchor: date) -> float:
        start = self.start_date(anchor)
        return year_fraction(anchor, start, self.day_count)
    
    def implied_discount_factor(
        self,
        anchor: date,
        prior_dfs: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Implied DF(T2) from FRA rate.
        
        F = (DF(T1)/DF(T2) - 1) / tau
        DF(T2) = DF(T1) / (1 + F * tau)
        """
        t1 = self.start_time(anchor)
        t2 = self.maturity_time(anchor)
        
        # Get DF(T1) from prior curve
        df1 = _interpolate_df(t1, prior_dfs)
        
        # Forward period
        tau = t2 - t1
        if tau <= 0:
            return (t2, df1)
        
        df2 = df1 / (1.0 + self.quote * tau)
        
        return (t2, df2)


@dataclass
class Future(CurveInstrument):
    """
    Interest rate future (e.g., SOFR future, Eurodollar).
    
    Quote is 100 - rate (in percentage points).
    
    Assumptions (simplified):
    - Ignores convexity adjustment
    - Treats futures rate as forward rate
    - Uses IMM dates
    
    Note: A proper implementation would apply convexity adjustment,
    but this is beyond scope for this library.
    """
    contract_month: Optional[date] = None  # Contract month
    contract_size: float = 1_000_000  # Notional per contract
    tick_value: float = 25.0  # Value of 1bp move per contract
    
    def implied_rate(self) -> float:
        """Convert futures price quote to implied rate."""
        # Quote is 100 - rate
        return (100.0 - self.quote) / 100.0
    
    def maturity_date(self, anchor: date) -> date:
        """Get futures expiry/settlement date."""
        if self.contract_month:
            return self.contract_month
        return DateUtils.add_tenor(anchor, self.tenor)
    
    def maturity_time(self, anchor: date) -> float:
        mat = self.maturity_date(anchor)
        return year_fraction(anchor, mat, self.day_count)
    
    def implied_discount_factor(
        self,
        anchor: date,
        prior_dfs: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Implied DF from futures rate (no convexity adjustment).
        
        Warning: This is simplified. A proper implementation
        would subtract a convexity adjustment from the futures rate.
        """
        # Get implied forward rate (ignoring convexity)
        fwd_rate = self.implied_rate()
        
        # Determine forward period (assume 3M by default)
        t1 = self.maturity_time(anchor)
        t2 = t1 + 0.25  # 3-month forward period
        
        df1 = _interpolate_df(t1, prior_dfs)
        df2 = df1 / (1.0 + fwd_rate * 0.25)
        
        return (t2, df2)


def _interpolate_df(t: float, prior_dfs: List[Tuple[float, float]]) -> float:
    """
    Linearly interpolate/extrapolate discount factor.
    
    Args:
        t: Target time
        prior_dfs: List of (time, df) pairs, sorted by time
        
    Returns:
        Interpolated discount factor
    """
    if not prior_dfs:
        return 1.0
    
    # Sort by time
    sorted_dfs = sorted(prior_dfs, key=lambda x: x[0])
    
    if t <= 0:
        return 1.0
    
    # Check if exact match
    for time, df in sorted_dfs:
        if abs(time - t) < 1e-10:
            return df
    
    # Extrapolate from t=0 if before first point
    if t < sorted_dfs[0][0]:
        t1, df1 = 0.0, 1.0
        t2, df2 = sorted_dfs[0]
        # Log-linear interpolation
        if t2 > t1:
            log_df = np.log(df2) * (t - t1) / (t2 - t1)
            return np.exp(log_df)
        return 1.0
    
    # Extrapolate beyond last point
    if t > sorted_dfs[-1][0]:
        if len(sorted_dfs) < 2:
            t1, df1 = 0.0, 1.0
            t2, df2 = sorted_dfs[0]
        else:
            t1, df1 = sorted_dfs[-2]
            t2, df2 = sorted_dfs[-1]
        
        if t2 > t1:
            # Log-linear extrapolation
            log_df1 = np.log(df1) if df1 > 0 else 0
            log_df2 = np.log(df2) if df2 > 0 else 0
            slope = (log_df2 - log_df1) / (t2 - t1)
            log_df = log_df2 + slope * (t - t2)
            return np.exp(log_df)
        return sorted_dfs[-1][1]
    
    # Find bracketing points
    for i in range(len(sorted_dfs) - 1):
        t1, df1 = sorted_dfs[i]
        t2, df2 = sorted_dfs[i + 1]
        
        if t1 <= t <= t2:
            # Log-linear interpolation
            w = (t - t1) / (t2 - t1) if t2 > t1 else 0
            log_df = (1 - w) * np.log(df1) + w * np.log(df2)
            return np.exp(log_df)
    
    return sorted_dfs[-1][1]


def _days_in_month(year: int, month: int) -> int:
    """Return number of days in a month."""
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif month in (4, 6, 9, 11):
        return 30
    elif month == 2:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return 29
        return 28
    raise ValueError(f"Invalid month: {month}")


__all__ = [
    "CurveInstrument",
    "Deposit",
    "OISSwap",
    "FRA",
    "Future",
]
