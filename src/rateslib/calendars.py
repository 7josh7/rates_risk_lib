"""
Business-day calendar utilities for rates workflows.

The existing library mostly used a weekend-only notion of business day.
This module adds explicit calendar objects so pricing and schedule generation
can be configured more realistically without breaking legacy set-of-holidays
inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, Iterable, Optional, Set


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth weekday of a given month."""
    if n <= 0:
        raise ValueError("n must be positive")
    first = date(year, month, 1)
    day_offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=day_offset + 7 * (n - 1))


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Return the last weekday of a given month."""
    if month == 12:
        current = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        current = date(year, month + 1, 1) - timedelta(days=1)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def _observed_holiday(nominal_date: date) -> date:
    """Apply standard US-style weekend observance."""
    if nominal_date.weekday() == 5:
        return nominal_date - timedelta(days=1)
    if nominal_date.weekday() == 6:
        return nominal_date + timedelta(days=1)
    return nominal_date


@dataclass
class BusinessCalendar:
    """
    Base business-day calendar.

    Subclasses can override ``holiday_dates`` to provide generated holiday
    rules. Weekend days default to Saturday and Sunday.
    """

    weekend_days: Set[int] = field(default_factory=lambda: {5, 6})
    _holiday_cache: Dict[int, Set[date]] = field(default_factory=dict, init=False, repr=False)

    def holiday_dates(self, year: int) -> Set[date]:
        """Return holidays observed in the requested year."""
        return set()

    def holidays(self, year: int) -> Set[date]:
        """Cached wrapper around ``holiday_dates``."""
        if year not in self._holiday_cache:
            self._holiday_cache[year] = set(self.holiday_dates(year))
        return set(self._holiday_cache[year])

    def is_business_day(self, d: date) -> bool:
        """Return True when the supplied date is a business day."""
        if d.weekday() in self.weekend_days:
            return False
        return d not in self.holidays(d.year)


@dataclass
class WeekendOnlyCalendar(BusinessCalendar):
    """Calendar with weekends only and no holidays."""

    pass


@dataclass
class ExplicitHolidayCalendar(BusinessCalendar):
    """Calendar backed by an explicit set of holiday dates."""

    holiday_set: Set[date] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.holiday_set = set(self.holiday_set)

    def holiday_dates(self, year: int) -> Set[date]:
        return {d for d in self.holiday_set if d.year == year}


@dataclass
class UnitedStatesHolidayCalendar(BusinessCalendar):
    """
    US federal-holiday style calendar.

    This is a practical calendar for USD rates examples. It is not intended to
    be an exhaustive exchange calendar, but it is materially more realistic
    than a weekend-only calendar for settlement and coupon scheduling.
    """

    include_columbus_day: bool = True
    include_veterans_day: bool = True

    def _nominal_holidays_for_year(self, year: int) -> Set[date]:
        holidays = {
            _observed_holiday(date(year, 1, 1)),
            _nth_weekday_of_month(year, 1, 0, 3),   # Martin Luther King Jr. Day
            _nth_weekday_of_month(year, 2, 0, 3),   # Presidents Day
            _last_weekday_of_month(year, 5, 0),     # Memorial Day
            _observed_holiday(date(year, 7, 4)),
            _nth_weekday_of_month(year, 9, 0, 1),   # Labor Day
            _nth_weekday_of_month(year, 11, 3, 4),  # Thanksgiving
            _observed_holiday(date(year, 12, 25)),
        }

        if year >= 2021:
            holidays.add(_observed_holiday(date(year, 6, 19)))  # Juneteenth

        if self.include_columbus_day:
            holidays.add(_nth_weekday_of_month(year, 10, 0, 2))

        if self.include_veterans_day:
            holidays.add(_observed_holiday(date(year, 11, 11)))

        return holidays

    def holiday_dates(self, year: int) -> Set[date]:
        dates: Set[date] = set()
        for nominal_year in (year - 1, year, year + 1):
            for holiday in self._nominal_holidays_for_year(nominal_year):
                if holiday.year == year:
                    dates.add(holiday)
        return dates


def calendar_from_input(calendar_input: Optional[object]) -> BusinessCalendar:
    """
    Normalize None, explicit holiday iterables, and calendar objects.
    """
    if calendar_input is None:
        return WeekendOnlyCalendar()

    if isinstance(calendar_input, BusinessCalendar):
        return calendar_input

    if hasattr(calendar_input, "is_business_day") and callable(calendar_input.is_business_day):
        return calendar_input  # type: ignore[return-value]

    if isinstance(calendar_input, Iterable):
        return ExplicitHolidayCalendar(set(calendar_input))

    raise TypeError(
        "Calendar input must be None, a BusinessCalendar, or an iterable of dates"
    )


__all__ = [
    "BusinessCalendar",
    "WeekendOnlyCalendar",
    "ExplicitHolidayCalendar",
    "UnitedStatesHolidayCalendar",
    "calendar_from_input",
]
