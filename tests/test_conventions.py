"""
Unit tests for conventions module.
"""

from datetime import date
import pytest

from rateslib.conventions import (
    DayCount,
    BusinessDayConvention,
    year_fraction,
    Conventions,
)


class TestDayCount:
    """Tests for day count conventions."""
    
    def test_act_360(self):
        """Test ACT/360 day count."""
        start = date(2024, 1, 15)
        end = date(2024, 4, 15)  # 91 days
        
        yf = year_fraction(start, end, DayCount.ACT_360)
        expected = 91 / 360
        
        assert abs(yf - expected) < 1e-10
    
    def test_act_365(self):
        """Test ACT/365 day count."""
        start = date(2024, 1, 15)
        end = date(2024, 4, 15)  # 91 days
        
        yf = year_fraction(start, end, DayCount.ACT_365)
        expected = 91 / 365
        
        assert abs(yf - expected) < 1e-10
    
    def test_act_act(self):
        """Test ACT/ACT day count."""
        start = date(2024, 1, 15)
        end = date(2025, 1, 15)  # Full year including leap year
        
        yf = year_fraction(start, end, DayCount.ACT_ACT)
        # Should be approximately 1 year (allowing for implementation differences)
        assert 0.99 < yf < 1.01
    
    def test_thirty_360(self):
        """Test 30/360 day count."""
        start = date(2024, 1, 15)
        end = date(2024, 4, 15)  # 3 months
        
        yf = year_fraction(start, end, DayCount.THIRTY_360)
        expected = 90 / 360  # 3 months * 30 days
        
        assert abs(yf - expected) < 1e-10
    
    def test_year_fraction_same_date(self):
        """Test year fraction for same date returns 0."""
        d = date(2024, 1, 15)
        yf = year_fraction(d, d, DayCount.ACT_360)
        assert yf == 0.0


class TestConventions:
    """Tests for convention presets."""
    
    def test_usd_ois_preset(self):
        """Test USD OIS conventions."""
        conv = Conventions.usd_ois()
        assert conv.day_count == DayCount.ACT_360
        assert conv.business_day == BusinessDayConvention.MODIFIED_FOLLOWING
        assert conv.settlement_days == 2
    
    def test_usd_treasury_preset(self):
        """Test USD Treasury conventions."""
        conv = Conventions.usd_treasury()
        assert conv.day_count == DayCount.ACT_ACT
        assert conv.settlement_days == 1
