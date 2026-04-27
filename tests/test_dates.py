"""
Unit tests for dates module.
"""

from datetime import date
import pytest

from rateslib.calendars import UnitedStatesHolidayCalendar
from rateslib.conventions import BusinessDayConvention, DayCount
from rateslib.dates import DateUtils, ScheduleInfo, generate_bond_schedule


class TestDateUtils:
    """Tests for DateUtils class."""
    
    def test_parse_tenor_months(self):
        """Test parsing month tenors."""
        assert DateUtils.parse_tenor("3M") == (3, 'M')
        assert DateUtils.parse_tenor("6M") == (6, 'M')
        assert DateUtils.parse_tenor("12M") == (12, 'M')
    
    def test_parse_tenor_years(self):
        """Test parsing year tenors."""
        assert DateUtils.parse_tenor("1Y") == (1, 'Y')
        assert DateUtils.parse_tenor("5Y") == (5, 'Y')
        assert DateUtils.parse_tenor("10Y") == (10, 'Y')
        assert DateUtils.parse_tenor("30Y") == (30, 'Y')
    
    def test_parse_tenor_weeks(self):
        """Test parsing week tenors."""
        assert DateUtils.parse_tenor("1W") == (1, 'W')
        assert DateUtils.parse_tenor("2W") == (2, 'W')
    
    def test_parse_tenor_days(self):
        """Test parsing day tenors."""
        assert DateUtils.parse_tenor("1D") == (1, 'D')
        assert DateUtils.parse_tenor("30D") == (30, 'D')
    
    def test_parse_tenor_lowercase(self):
        """Test parsing lowercase tenors."""
        assert DateUtils.parse_tenor("3m") == (3, 'M')
        assert DateUtils.parse_tenor("5y") == (5, 'Y')
    
    def test_parse_tenor_invalid(self):
        """Test invalid tenor raises error."""
        with pytest.raises(ValueError):
            DateUtils.parse_tenor("invalid")
        with pytest.raises(ValueError):
            DateUtils.parse_tenor("3X")
    
    def test_add_tenor_months(self):
        """Test adding month tenors."""
        base = date(2024, 1, 15)
        
        result = DateUtils.add_tenor(base, "3M")
        assert result == date(2024, 4, 15)
        
        result = DateUtils.add_tenor(base, "6M")
        assert result == date(2024, 7, 15)
    
    def test_add_tenor_years(self):
        """Test adding year tenors."""
        base = date(2024, 1, 15)
        
        result = DateUtils.add_tenor(base, "1Y")
        assert result == date(2025, 1, 15)
        
        result = DateUtils.add_tenor(base, "5Y")
        assert result == date(2029, 1, 15)
    
    def test_add_tenor_weeks(self):
        """Test adding week tenors."""
        base = date(2024, 1, 15)
        
        result = DateUtils.add_tenor(base, "1W")
        assert result == date(2024, 1, 22)
        
        result = DateUtils.add_tenor(base, "2W")
        assert result == date(2024, 1, 29)
    
    def test_add_tenor_end_of_month(self):
        """Test adding tenor at end of month."""
        base = date(2024, 1, 31)
        
        result = DateUtils.add_tenor(base, "1M")
        # February doesn't have 31 days, should be Feb 29 (leap year)
        assert result.month == 2
        assert result.day <= 29

    def test_add_tenor_end_of_month_with_business_day_adjustment(self):
        """Test month rolling keeps end-of-month behavior before adjusting."""
        base = date(2024, 8, 31)

        result = DateUtils.add_tenor(
            base,
            "1M",
            adjust_to_business_day=True,
            business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING,
            end_of_month=True,
        )

        assert result == date(2024, 9, 30)

    def test_spot_date_skips_holidays(self):
        """Test spot date rolls over holidays and weekends."""
        holidays = UnitedStatesHolidayCalendar()

        result = DateUtils.spot_date(date(2024, 7, 3), settlement_days=2, holidays=holidays)

        assert result == date(2024, 7, 8)
    
    def test_tenor_to_years(self):
        """Test converting tenor to years."""
        assert abs(DateUtils.tenor_to_years("1Y") - 1.0) < 1e-10
        assert abs(DateUtils.tenor_to_years("6M") - 0.5) < 1e-10
        assert abs(DateUtils.tenor_to_years("3M") - 0.25) < 1e-10
        # 1W = 7/365 days (not 1/52 exactly)
        assert abs(DateUtils.tenor_to_years("1W") - 7/365) < 1e-10


class TestScheduleGeneration:
    """Tests for schedule generation."""
    
    def test_generate_bond_schedule(self):
        """Test bond schedule generation."""
        schedule = generate_bond_schedule(
            settle=date(2024, 1, 15),
            maturity=date(2026, 1, 15),
            coupon_freq=2,  # Semi-annual
            day_count=DayCount.ACT_ACT
        )
        
        assert isinstance(schedule, ScheduleInfo)
        assert len(schedule.payment_dates) > 0
        assert schedule.payment_dates[-1] == date(2026, 1, 15)
    
    def test_schedule_frequency_annual(self):
        """Test annual frequency."""
        schedule = generate_bond_schedule(
            settle=date(2024, 1, 15),
            maturity=date(2027, 1, 15),
            coupon_freq=1,
            day_count=DayCount.ACT_ACT
        )
        
        # Should have 3 payments (2025, 2026, 2027)
        assert len(schedule.payment_dates) == 3
    
    def test_schedule_frequency_quarterly(self):
        """Test quarterly frequency."""
        schedule = generate_bond_schedule(
            settle=date(2024, 1, 15),
            maturity=date(2025, 1, 15),
            coupon_freq=4,
            day_count=DayCount.ACT_ACT
        )
        
        # Should have 4 payments
        assert len(schedule.payment_dates) == 4

    def test_generate_schedule_supports_front_and_back_stubs(self):
        start = date(2024, 1, 15)
        end = date(2025, 2, 15)

        short_front = DateUtils.generate_schedule(
            start,
            end,
            4,
            convention=BusinessDayConvention.UNADJUSTED,
            stub="short_front",
        )
        long_front = DateUtils.generate_schedule(
            start,
            end,
            4,
            convention=BusinessDayConvention.UNADJUSTED,
            stub="long_front",
        )
        short_back = DateUtils.generate_schedule(
            start,
            end,
            4,
            convention=BusinessDayConvention.UNADJUSTED,
            stub="short_back",
        )
        long_back = DateUtils.generate_schedule(
            start,
            end,
            4,
            convention=BusinessDayConvention.UNADJUSTED,
            stub="long_back",
        )

        assert short_front == [
            date(2024, 2, 15),
            date(2024, 5, 15),
            date(2024, 8, 15),
            date(2024, 11, 15),
            date(2025, 2, 15),
        ]
        assert long_front == [
            date(2024, 5, 15),
            date(2024, 8, 15),
            date(2024, 11, 15),
            date(2025, 2, 15),
        ]
        assert short_back == [
            date(2024, 4, 15),
            date(2024, 7, 15),
            date(2024, 10, 15),
            date(2025, 1, 15),
            date(2025, 2, 15),
        ]
        assert long_back == [
            date(2024, 4, 15),
            date(2024, 7, 15),
            date(2024, 10, 15),
            date(2025, 2, 15),
        ]

    def test_generate_schedule_end_of_month_rule(self):
        schedule = DateUtils.generate_schedule(
            start=date(2024, 1, 31),
            end=date(2024, 7, 31),
            frequency=4,
            convention=BusinessDayConvention.UNADJUSTED,
            end_of_month=True,
        )

        assert schedule == [date(2024, 4, 30), date(2024, 7, 31)]

    def test_generate_bond_schedule_respects_explicit_accrual_start(self):
        schedule = generate_bond_schedule(
            settle=date(2024, 3, 15),
            maturity=date(2025, 1, 31),
            coupon_freq=2,
            day_count=DayCount.ACT_ACT,
            business_day_convention=BusinessDayConvention.UNADJUSTED,
            end_of_month=True,
            accrual_start=date(2024, 1, 31),
        )

        assert schedule.payment_dates == [date(2024, 7, 31), date(2025, 1, 31)]
        assert schedule.accrual_starts[0] == date(2024, 1, 31)
        assert schedule.accrual_ends[0] == date(2024, 7, 31)
        assert schedule.unadjusted_payment_dates == [date(2024, 7, 31), date(2025, 1, 31)]
