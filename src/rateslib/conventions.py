"""
Day count conventions and business day adjustments for rates instruments.

Supported Day Counts:
- ACT/360: Actual days / 360 (money markets, OIS)
- ACT/365: Actual days / 365 
- ACT/ACT: Actual days / actual days in year (Treasuries)
- 30/360: 30 days per month / 360 (some swaps)

Business Day Conventions:
- Modified Following: Move to next business day, unless it falls in next month (then previous)
- Following: Move to next business day
- Preceding: Move to previous business day
"""

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Iterable, Optional, Union
import calendar

from .calendars import BusinessCalendar, calendar_from_input


CalendarInput = Optional[Union[BusinessCalendar, Iterable[date]]]


class DayCount(Enum):
    """Day count convention enumeration."""
    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"
    ACT_ACT = "ACT/ACT"
    THIRTY_360 = "30/360"
    
    @classmethod
    def from_string(cls, s: str) -> "DayCount":
        """Parse day count from string representation."""
        mapping = {
            "ACT/360": cls.ACT_360,
            "ACT360": cls.ACT_360,
            "ACT/365": cls.ACT_365,
            "ACT365": cls.ACT_365,
            "ACT/ACT": cls.ACT_ACT,
            "ACTACT": cls.ACT_ACT,
            "30/360": cls.THIRTY_360,
            "30360": cls.THIRTY_360,
        }
        key = s.upper().replace(" ", "")
        if key in mapping:
            return mapping[key]
        raise ValueError(f"Unknown day count convention: {s}")


class BusinessDayConvention(Enum):
    """Business day adjustment convention."""
    MODIFIED_FOLLOWING = "ModifiedFollowing"
    FOLLOWING = "Following"
    PRECEDING = "Preceding"
    UNADJUSTED = "Unadjusted"


class CompoundingConvention(Enum):
    """Interest rate compounding convention."""
    CONTINUOUS = "Continuous"
    ANNUAL = "Annual"
    SEMI_ANNUAL = "SemiAnnual"
    QUARTERLY = "Quarterly"
    SIMPLE = "Simple"


@dataclass
class Conventions:
    """
    Container for instrument conventions.
    
    Attributes:
        day_count: Day count convention for accrual
        business_day: Business day adjustment rule
        compounding: Rate compounding convention
        payment_frequency: Number of payments per year (1=annual, 2=semi, 4=quarterly)
        settlement_days: Days to settle from trade date
    """
    day_count: DayCount = DayCount.ACT_360
    business_day: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING
    compounding: CompoundingConvention = CompoundingConvention.CONTINUOUS
    payment_frequency: int = 1  # Annual
    settlement_days: int = 2
    holiday_calendar: CalendarInput = None
    end_of_month: bool = False
    stub: str = "short_front"
    
    # Standard USD conventions
    @classmethod
    def usd_ois(cls) -> "Conventions":
        """Standard USD OIS conventions."""
        return cls(
            day_count=DayCount.ACT_360,
            business_day=BusinessDayConvention.MODIFIED_FOLLOWING,
            compounding=CompoundingConvention.ANNUAL,
            payment_frequency=1,
            settlement_days=2
        )
    
    @classmethod
    def usd_treasury(cls) -> "Conventions":
        """Standard USD Treasury bond conventions."""
        return cls(
            day_count=DayCount.ACT_ACT,
            business_day=BusinessDayConvention.FOLLOWING,
            compounding=CompoundingConvention.SEMI_ANNUAL,
            payment_frequency=2,
            settlement_days=1
        )
    
    @classmethod
    def usd_swap(cls) -> "Conventions":
        """Standard USD IRS conventions (fixed leg)."""
        return cls(
            day_count=DayCount.ACT_360,
            business_day=BusinessDayConvention.MODIFIED_FOLLOWING,
            compounding=CompoundingConvention.SEMI_ANNUAL,
            payment_frequency=2,
            settlement_days=2
        )


def year_fraction(start: date, end: date, day_count: DayCount) -> float:
    """
    Calculate year fraction between two dates using specified day count convention.
    
    Args:
        start: Start date
        end: End date
        day_count: Day count convention
        
    Returns:
        Year fraction as float
    
    Conventions:
        ACT/360: (end - start).days / 360
        ACT/365: (end - start).days / 365
        ACT/ACT: Actual days / actual days in period's year(s)
        30/360: Assumes 30 days per month, 360 days per year
    """
    if start >= end:
        return 0.0
    
    actual_days = (end - start).days
    
    if day_count == DayCount.ACT_360:
        return actual_days / 360.0
    
    elif day_count == DayCount.ACT_365:
        return actual_days / 365.0
    
    elif day_count == DayCount.ACT_ACT:
        # ISDA ACT/ACT: split by year boundaries
        if start.year == end.year:
            days_in_year = 366 if calendar.isleap(start.year) else 365
            return actual_days / days_in_year
        else:
            # Accumulate fractions across years
            total = 0.0
            current = start
            for year in range(start.year, end.year + 1):
                year_start = date(year, 1, 1) if year > start.year else start
                year_end = date(year, 12, 31) if year < end.year else end
                days_in_year = 366 if calendar.isleap(year) else 365
                days_in_period = (year_end - year_start).days + (1 if year == end.year else 0)
                if year == start.year:
                    days_in_period = (date(year, 12, 31) - start).days + 1
                if year == end.year:
                    days_in_period = (end - date(year, 1, 1)).days
                total += days_in_period / days_in_year
            return total
    
    elif day_count == DayCount.THIRTY_360:
        # 30/360 US convention
        d1 = min(start.day, 30)
        d2 = end.day if start.day < 30 else min(end.day, 30)
        if start.day == 31:
            d1 = 30
        if end.day == 31 and d1 == 30:
            d2 = 30
        return (360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)) / 360.0
    
    else:
        raise ValueError(f"Unknown day count: {day_count}")


def is_business_day(d: date, holidays: CalendarInput = None) -> bool:
    """
    Check if a date is a business day.
    
    Uses weekend-only calendar by default (Saturday/Sunday are non-business days).
    
    Args:
        d: Date to check
        holidays: Optional set of holiday dates
        
    Returns:
        True if business day, False otherwise
    """
    cal = calendar_from_input(holidays)
    return cal.is_business_day(d)


def advance_business_days(
    start: date,
    business_days: int,
    holidays: CalendarInput = None,
) -> date:
    """
    Advance forward or backward by a number of business days.
    """
    if business_days == 0:
        return start

    step = 1 if business_days > 0 else -1
    remaining = abs(business_days)
    current = start

    while remaining > 0:
        current += timedelta(days=step)
        if is_business_day(current, holidays):
            remaining -= 1

    return current


def adjust_business_day(
    d: date, 
    convention: BusinessDayConvention,
    holidays: CalendarInput = None
) -> date:
    """
    Adjust a date according to business day convention.
    
    Args:
        d: Date to adjust
        convention: Business day adjustment rule
        holidays: Optional set of holiday dates
        
    Returns:
        Adjusted date
    """
    if convention == BusinessDayConvention.UNADJUSTED:
        return d
    
    if is_business_day(d, holidays):
        return d
    
    if convention == BusinessDayConvention.FOLLOWING:
        adjusted = d
        while not is_business_day(adjusted, holidays):
            adjusted += timedelta(days=1)
        return adjusted
    
    elif convention == BusinessDayConvention.PRECEDING:
        adjusted = d
        while not is_business_day(adjusted, holidays):
            adjusted -= timedelta(days=1)
        return adjusted
    
    elif convention == BusinessDayConvention.MODIFIED_FOLLOWING:
        # First try following
        adjusted = d
        while not is_business_day(adjusted, holidays):
            adjusted += timedelta(days=1)
        
        # If we crossed into next month, go preceding instead
        if adjusted.month != d.month:
            adjusted = d
            while not is_business_day(adjusted, holidays):
                adjusted -= timedelta(days=1)
        
        return adjusted
    
    return d


# Re-export key functions
__all__ = [
    "DayCount",
    "BusinessDayConvention", 
    "CompoundingConvention",
    "CalendarInput",
    "Conventions",
    "year_fraction",
    "is_business_day",
    "advance_business_days",
    "adjust_business_day",
]
