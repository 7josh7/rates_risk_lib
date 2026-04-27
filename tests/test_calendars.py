from datetime import date

from rateslib.calendars import UnitedStatesHolidayCalendar, WeekendOnlyCalendar
from rateslib.conventions import (
    BusinessDayConvention,
    adjust_business_day,
    advance_business_days,
)
from rateslib.dates import DateUtils


class TestBusinessCalendars:
    def test_weekend_only_calendar(self):
        calendar = WeekendOnlyCalendar()

        assert calendar.is_business_day(date(2024, 4, 19))
        assert not calendar.is_business_day(date(2024, 4, 20))

    def test_us_calendar_observes_july_fourth(self):
        calendar = UnitedStatesHolidayCalendar()

        assert not calendar.is_business_day(date(2024, 7, 4))
        assert not calendar.is_business_day(date(2021, 7, 5))
        assert calendar.is_business_day(date(2024, 7, 5))

    def test_following_adjustment_skips_holiday(self):
        calendar = UnitedStatesHolidayCalendar()

        adjusted = adjust_business_day(
            date(2024, 7, 4),
            BusinessDayConvention.FOLLOWING,
            calendar,
        )

        assert adjusted == date(2024, 7, 5)

    def test_modified_following_respects_month_boundary(self):
        calendar = UnitedStatesHolidayCalendar()

        adjusted = adjust_business_day(
            date(2024, 8, 31),
            BusinessDayConvention.MODIFIED_FOLLOWING,
            calendar,
        )

        assert adjusted == date(2024, 8, 30)

    def test_advance_business_days_skips_holidays_and_weekends(self):
        calendar = UnitedStatesHolidayCalendar()

        assert advance_business_days(date(2024, 7, 3), 1, calendar) == date(2024, 7, 5)
        assert DateUtils.spot_date(date(2024, 7, 3), 2, calendar) == date(2024, 7, 8)
