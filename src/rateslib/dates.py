"""
Date utilities for rates calculations.

Provides:
- Tenor parsing and date generation
- Schedule generation for coupon bonds and swaps
- Business day adjustments
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional, Tuple
import re

from .conventions import (
    BusinessDayConvention, 
    DayCount,
    adjust_business_day, 
    is_business_day,
    year_fraction
)


class DateUtils:
    """Utility class for date manipulation in rates contexts."""
    
    # Tenor regex pattern: number + unit (D/W/M/Y)
    TENOR_PATTERN = re.compile(r'^(\d+)([DWMY])$', re.IGNORECASE)
    
    @staticmethod
    def parse_tenor(tenor: str) -> Tuple[int, str]:
        """
        Parse a tenor string into (amount, unit).
        
        Args:
            tenor: Tenor string like "1D", "3M", "2Y"
            
        Returns:
            Tuple of (amount, unit) where unit is D/W/M/Y
            
        Raises:
            ValueError: If tenor format is invalid
        """
        match = DateUtils.TENOR_PATTERN.match(tenor.upper().strip())
        if not match:
            raise ValueError(f"Invalid tenor format: {tenor}. Expected format like '3M', '2Y'")
        
        return int(match.group(1)), match.group(2).upper()
    
    @staticmethod
    def add_tenor(start: date, tenor: str, holidays: Optional[set] = None) -> date:
        """
        Add a tenor to a date.
        
        Args:
            start: Starting date
            tenor: Tenor string (e.g., "1D", "3M", "2Y")
            holidays: Optional holiday calendar
            
        Returns:
            End date
        """
        amount, unit = DateUtils.parse_tenor(tenor)
        
        if unit == 'D':
            # Add business days
            result = start
            days_added = 0
            while days_added < amount:
                result += timedelta(days=1)
                if is_business_day(result, holidays):
                    days_added += 1
            return result
        
        elif unit == 'W':
            result = start + timedelta(weeks=amount)
        
        elif unit == 'M':
            # Add months, preserving day of month where possible
            year = start.year + (start.month + amount - 1) // 12
            month = (start.month + amount - 1) % 12 + 1
            day = min(start.day, _days_in_month(year, month))
            result = date(year, month, day)
        
        elif unit == 'Y':
            year = start.year + amount
            # Handle Feb 29 in leap years
            day = min(start.day, _days_in_month(year, start.month))
            result = date(year, start.month, day)
        
        else:
            raise ValueError(f"Unknown tenor unit: {unit}")
        
        return result
    
    @staticmethod
    def tenor_to_years(tenor: str) -> float:
        """
        Convert tenor to approximate year fraction.
        
        Args:
            tenor: Tenor string
            
        Returns:
            Approximate years as float
        """
        amount, unit = DateUtils.parse_tenor(tenor)
        
        if unit == 'D':
            return amount / 365.0
        elif unit == 'W':
            return amount * 7 / 365.0
        elif unit == 'M':
            return amount / 12.0
        elif unit == 'Y':
            return float(amount)
        else:
            raise ValueError(f"Unknown tenor unit: {unit}")
    
    @staticmethod
    def generate_schedule(
        start: date,
        end: date,
        frequency: int,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
        holidays: Optional[set] = None,
        stub: str = "short_front"
    ) -> List[date]:
        """
        Generate a payment schedule between start and end dates.
        
        Args:
            start: Schedule start (accrual start)
            end: Schedule end (maturity)
            frequency: Payments per year (1=annual, 2=semi, 4=quarterly, 12=monthly)
            convention: Business day adjustment
            holidays: Holiday calendar
            stub: Stub period handling ("short_front", "short_back", "long_front", "long_back")
            
        Returns:
            List of payment dates (adjusted for business days)
        """
        if frequency <= 0:
            raise ValueError("Frequency must be positive")
        
        months_per_period = 12 // frequency
        
        # Generate unadjusted dates backward from maturity
        unadjusted = [end]
        current = end
        
        while True:
            # Subtract months
            year = current.year
            month = current.month - months_per_period
            
            while month <= 0:
                month += 12
                year -= 1
            
            day = min(end.day, _days_in_month(year, month))
            prev_date = date(year, month, day)
            
            if prev_date <= start:
                break
            
            unadjusted.insert(0, prev_date)
            current = prev_date
        
        # Adjust for business days
        adjusted = [adjust_business_day(d, convention, holidays) for d in unadjusted]
        
        return adjusted
    
    @staticmethod
    def generate_ois_schedule(
        start: date,
        end: date,
        payment_frequency: str = "ANNUAL",
        holidays: Optional[set] = None
    ) -> List[date]:
        """
        Generate OIS payment schedule.
        
        OIS typically pays annually with daily compounding.
        
        Args:
            start: Effective date
            end: Maturity date
            payment_frequency: "ANNUAL", "SEMI", "QUARTERLY"
            holidays: Holiday calendar
            
        Returns:
            List of payment dates
        """
        freq_map = {"ANNUAL": 1, "SEMI": 2, "QUARTERLY": 4, "DAILY": 252}
        freq = freq_map.get(payment_frequency.upper(), 1)
        
        if freq == 252:  # Daily payment (unusual but supported)
            dates = []
            current = start + timedelta(days=1)
            while current <= end:
                if is_business_day(current, holidays):
                    dates.append(current)
                current += timedelta(days=1)
            return dates
        
        return DateUtils.generate_schedule(
            start, end, freq,
            BusinessDayConvention.MODIFIED_FOLLOWING,
            holidays
        )


@dataclass
class ScheduleInfo:
    """Container for schedule with accrual information."""
    payment_dates: List[date]
    accrual_starts: List[date]
    accrual_ends: List[date]
    year_fractions: List[float]
    day_count: DayCount


def generate_bond_schedule(
    settle: date,
    maturity: date,
    coupon_freq: int,
    day_count: DayCount,
    holidays: Optional[set] = None
) -> ScheduleInfo:
    """
    Generate a bond coupon schedule with accrual periods.
    
    Args:
        settle: Settlement date
        maturity: Maturity date
        coupon_freq: Coupons per year (2 for semi-annual)
        day_count: Day count convention
        holidays: Holiday calendar
        
    Returns:
        ScheduleInfo with payment dates and accrual fractions
    """
    # Generate payment dates
    payment_dates = DateUtils.generate_schedule(
        settle, maturity, coupon_freq,
        BusinessDayConvention.FOLLOWING,
        holidays
    )
    
    # Filter out any dates on or before settlement
    payment_dates = [d for d in payment_dates if d > settle]
    
    if not payment_dates:
        # Bond matures immediately - just return maturity
        return ScheduleInfo(
            payment_dates=[maturity],
            accrual_starts=[settle],
            accrual_ends=[maturity],
            year_fractions=[year_fraction(settle, maturity, day_count)],
            day_count=day_count
        )
    
    # Build accrual periods
    accrual_starts = []
    accrual_ends = []
    yfs = []
    
    # First period starts at settle
    prev = settle
    for pmt_date in payment_dates:
        accrual_starts.append(prev)
        accrual_ends.append(pmt_date)
        yfs.append(year_fraction(prev, pmt_date, day_count))
        prev = pmt_date
    
    return ScheduleInfo(
        payment_dates=payment_dates,
        accrual_starts=accrual_starts,
        accrual_ends=accrual_ends,
        year_fractions=yfs,
        day_count=day_count
    )


def generate_swap_schedule(
    effective: date,
    maturity: date,
    fixed_freq: int,
    float_freq: int,
    fixed_dc: DayCount,
    float_dc: DayCount,
    holidays: Optional[set] = None
) -> Tuple[ScheduleInfo, ScheduleInfo]:
    """
    Generate swap schedules for fixed and floating legs.
    
    Args:
        effective: Swap effective date
        maturity: Swap maturity
        fixed_freq: Fixed leg payments per year
        float_freq: Floating leg payments per year
        fixed_dc: Fixed leg day count
        float_dc: Floating leg day count
        holidays: Holiday calendar
        
    Returns:
        Tuple of (fixed_schedule, float_schedule)
    """
    fixed_schedule = generate_bond_schedule(
        effective, maturity, fixed_freq, fixed_dc, holidays
    )
    
    float_schedule = generate_bond_schedule(
        effective, maturity, float_freq, float_dc, holidays
    )
    
    return fixed_schedule, float_schedule


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
    "DateUtils",
    "ScheduleInfo",
    "generate_bond_schedule",
    "generate_swap_schedule",
]
