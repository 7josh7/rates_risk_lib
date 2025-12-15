"""
Unit tests for dates module.
"""

from datetime import date
import pytest

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
        from rateslib.conventions import DayCount
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
        from rateslib.conventions import DayCount
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
        from rateslib.conventions import DayCount
        schedule = generate_bond_schedule(
            settle=date(2024, 1, 15),
            maturity=date(2025, 1, 15),
            coupon_freq=4,
            day_count=DayCount.ACT_ACT
        )
        
        # Should have 4 payments
        assert len(schedule.payment_dates) == 4
