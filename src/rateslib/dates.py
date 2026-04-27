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
    CalendarInput,
    DayCount,
    advance_business_days,
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
    def add_business_days(
        start: date,
        business_days: int,
        holidays: CalendarInput = None,
    ) -> date:
        """
        Advance forward or backward by a number of business days.
        """
        return advance_business_days(start, business_days, holidays)

    @staticmethod
    def spot_date(
        trade_date: date,
        settlement_days: int = 2,
        holidays: CalendarInput = None,
    ) -> date:
        """
        Compute a spot/settlement date by advancing business days.
        """
        return DateUtils.add_business_days(trade_date, settlement_days, holidays)
    
    @staticmethod
    def add_tenor(
        start: date,
        tenor: str,
        holidays: CalendarInput = None,
        adjust_to_business_day: bool = False,
        business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
        end_of_month: bool = False,
    ) -> date:
        """
        Add a tenor to a date.
        
        Args:
            start: Starting date
            tenor: Tenor string (e.g., "1D", "3M", "2Y")
            holidays: Optional holiday calendar
            adjust_to_business_day: Whether to adjust the resulting date
            business_day_convention: Convention used when adjusting
            end_of_month: Preserve end-of-month rule for month/year tenors
            
        Returns:
            End date
        """
        amount, unit = DateUtils.parse_tenor(tenor)
        
        if unit == 'D':
            return DateUtils.add_business_days(start, amount, holidays)
        
        elif unit == 'W':
            result = start + timedelta(weeks=amount)
        
        elif unit == 'M':
            result = _shift_months(start, amount, end_of_month=end_of_month)
        
        elif unit == 'Y':
            result = _shift_months(start, amount * 12, end_of_month=end_of_month)
        
        else:
            raise ValueError(f"Unknown tenor unit: {unit}")

        if adjust_to_business_day:
            result = adjust_business_day(result, business_day_convention, holidays)

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
        holidays: CalendarInput = None,
        stub: str = "short_front",
        end_of_month: bool = False,
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
            end_of_month: Preserve end-of-month regular schedule dates
            
        Returns:
            List of payment dates (adjusted for business days)
        """
        if frequency <= 0 or 12 % frequency != 0:
            raise ValueError("Frequency must be positive")
        if end <= start:
            return []

        unadjusted = _generate_unadjusted_schedule_dates(
            start=start,
            end=end,
            frequency=frequency,
            stub=stub,
            end_of_month=end_of_month,
        )
        return [adjust_business_day(d, convention, holidays) for d in unadjusted]
    
    @staticmethod
    def generate_ois_schedule(
        start: date,
        end: date,
        payment_frequency: str = "ANNUAL",
        holidays: CalendarInput = None
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
    unadjusted_payment_dates: Optional[List[date]] = None
    stub_type: str = "short_front"
    end_of_month: bool = False


def generate_bond_schedule(
    settle: date,
    maturity: date,
    coupon_freq: int,
    day_count: DayCount,
    holidays: CalendarInput = None,
    business_day_convention: BusinessDayConvention = BusinessDayConvention.FOLLOWING,
    stub: str = "short_front",
    end_of_month: bool = False,
    accrual_start: Optional[date] = None,
) -> ScheduleInfo:
    """
    Generate a bond coupon schedule with accrual periods.
    
    Args:
        settle: Settlement date
        maturity: Maturity date
        coupon_freq: Coupons per year (2 for semi-annual)
        day_count: Day count convention
        holidays: Holiday calendar
        business_day_convention: Payment-date adjustment convention
        stub: Stub period handling
        end_of_month: Preserve end-of-month regular schedule dates
        accrual_start: Optional contractual accrual start date. When provided,
            past coupon dates are retained internally so the first live accrual
            period can begin before settlement.
        
    Returns:
        ScheduleInfo with payment dates and accrual fractions
    """
    schedule_start = accrual_start or settle

    unadjusted_payment_dates = _generate_unadjusted_schedule_dates(
        start=schedule_start,
        end=maturity,
        frequency=coupon_freq,
        stub=stub,
        end_of_month=end_of_month,
    )
    payment_dates_all = [
        adjust_business_day(d, business_day_convention, holidays)
        for d in unadjusted_payment_dates
    ]

    live_indices = [
        idx for idx, payment_date in enumerate(payment_dates_all)
        if payment_date > settle
    ]

    payment_dates = [payment_dates_all[idx] for idx in live_indices]

    if not payment_dates:
        # Bond matures immediately - just return maturity
        return ScheduleInfo(
            payment_dates=[maturity],
            accrual_starts=[settle],
            accrual_ends=[maturity],
            year_fractions=[year_fraction(settle, maturity, day_count)],
            day_count=day_count,
            unadjusted_payment_dates=[maturity],
            stub_type=stub,
            end_of_month=end_of_month,
        )

    boundaries = [schedule_start] + unadjusted_payment_dates
    accrual_starts = [boundaries[idx] for idx in live_indices]
    accrual_ends = [boundaries[idx + 1] for idx in live_indices]
    yfs = [
        year_fraction(start_date, end_date, day_count)
        for start_date, end_date in zip(accrual_starts, accrual_ends)
    ]

    return ScheduleInfo(
        payment_dates=payment_dates,
        accrual_starts=accrual_starts,
        accrual_ends=accrual_ends,
        year_fractions=yfs,
        day_count=day_count,
        unadjusted_payment_dates=[unadjusted_payment_dates[idx] for idx in live_indices],
        stub_type=stub,
        end_of_month=end_of_month,
    )


def generate_swap_schedule(
    effective: date,
    maturity: date,
    fixed_freq: int,
    float_freq: int,
    fixed_dc: DayCount,
    float_dc: DayCount,
    holidays: CalendarInput = None,
    fixed_holidays: CalendarInput = None,
    float_holidays: CalendarInput = None,
    fixed_business_day: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
    float_business_day: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
    fixed_stub: str = "short_front",
    float_stub: str = "short_front",
    fixed_end_of_month: bool = False,
    float_end_of_month: bool = False,
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
        holidays: Shared holiday calendar fallback for both legs
        fixed_holidays: Optional fixed-leg holiday calendar override
        float_holidays: Optional floating-leg holiday calendar override
        fixed_business_day: Fixed-leg payment-date adjustment convention
        float_business_day: Floating-leg payment-date adjustment convention
        fixed_stub: Fixed-leg stub handling
        float_stub: Floating-leg stub handling
        fixed_end_of_month: Fixed-leg end-of-month flag
        float_end_of_month: Floating-leg end-of-month flag
        
    Returns:
        Tuple of (fixed_schedule, float_schedule)
    """
    fixed_calendar = holidays if fixed_holidays is None else fixed_holidays
    float_calendar = holidays if float_holidays is None else float_holidays

    fixed_schedule = generate_bond_schedule(
        effective,
        maturity,
        fixed_freq,
        fixed_dc,
        holidays=fixed_calendar,
        business_day_convention=fixed_business_day,
        stub=fixed_stub,
        end_of_month=fixed_end_of_month,
        accrual_start=effective,
    )
    
    float_schedule = generate_bond_schedule(
        effective,
        maturity,
        float_freq,
        float_dc,
        holidays=float_calendar,
        business_day_convention=float_business_day,
        stub=float_stub,
        end_of_month=float_end_of_month,
        accrual_start=effective,
    )
    
    return fixed_schedule, float_schedule


def _is_end_of_month(d: date) -> bool:
    return d.day == _days_in_month(d.year, d.month)


def _shift_months(start: date, months: int, end_of_month: bool = False) -> date:
    """Shift a date by a number of months, preserving end-of-month when requested."""
    total_months = (start.year * 12 + (start.month - 1)) + months
    year = total_months // 12
    month = total_months % 12 + 1
    preserve_eom = end_of_month and _is_end_of_month(start)
    if preserve_eom:
        day = _days_in_month(year, month)
    else:
        day = min(start.day, _days_in_month(year, month))
    return date(year, month, day)


def _generate_unadjusted_schedule_dates(
    start: date,
    end: date,
    frequency: int,
    stub: str = "short_front",
    end_of_month: bool = False,
) -> List[date]:
    """Generate unadjusted regular schedule dates including the final maturity."""
    valid_stubs = {"short_front", "long_front", "short_back", "long_back"}
    if stub not in valid_stubs:
        raise ValueError(
            f"Unsupported stub type {stub!r}. Expected one of {sorted(valid_stubs)}"
        )
    if frequency <= 0 or 12 % frequency != 0:
        raise ValueError("Frequency must be a positive divisor of 12")
    if end <= start:
        return []

    months_per_period = 12 // frequency

    if stub in {"short_back", "long_back"}:
        interior_dates: List[date] = []
        current = start
        while True:
            next_date = _shift_months(current, months_per_period, end_of_month=end_of_month)
            if next_date >= end:
                break
            interior_dates.append(next_date)
            current = next_date
        if stub == "long_back" and interior_dates:
            interior_dates.pop()
        return interior_dates + [end]

    reverse_interior: List[date] = []
    current = end
    while True:
        prev_date = _shift_months(current, -months_per_period, end_of_month=end_of_month)
        if prev_date <= start:
            break
        reverse_interior.append(prev_date)
        current = prev_date

    interior_dates = list(reversed(reverse_interior))
    if stub == "long_front" and interior_dates:
        interior_dates = interior_dates[1:]
    return interior_dates + [end]


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
