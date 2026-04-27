"""
Bond pricing engine.

Prices zero-coupon and coupon bonds using discount factors from a yield curve.

Features:
- Cashflow schedule generation
- Present value calculation
- Clean and dirty price handling
- Accrued interest calculation

Conventions:
- Prices are expressed per 100 face value
- Yields are continuously compounded unless specified
"""

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

from ..conventions import (
    BusinessDayConvention,
    CalendarInput,
    Conventions,
    DayCount,
    year_fraction,
)
from ..dates import generate_bond_schedule
from ..curves.curve import Curve


@dataclass
class BondCashflow:
    """A single bond cashflow."""
    date: date
    amount: float  # In currency units for face=100
    type: str  # "COUPON", "PRINCIPAL", "COUPON+PRINCIPAL"
    year_fraction: float  # Time from curve anchor


@dataclass
class BondCashflows:
    """Complete set of bond cashflows."""
    cashflows: List[BondCashflow]
    settlement_date: date
    maturity_date: date
    coupon_rate: float
    frequency: int
    face_value: float
    day_count: DayCount
    
    @property
    def total_coupons(self) -> float:
        """Sum of all coupon payments."""
        return sum(cf.amount for cf in self.cashflows if "COUPON" in cf.type and cf.type != "COUPON+PRINCIPAL")
    
    @property
    def final_payment(self) -> float:
        """Final principal + coupon payment."""
        if self.cashflows:
            return self.cashflows[-1].amount
        return 0.0


class BondPricer:
    """
    Bond pricing engine.
    
    Prices bonds by discounting cashflows using a yield curve.
    
    Attributes:
        curve: Yield curve for discounting
        conventions: Bond conventions (day count, frequency, etc.)
    """
    
    def __init__(
        self,
        curve: Curve,
        conventions: Optional[Conventions] = None
    ):
        self.curve = curve
        self.conventions = conventions or Conventions.usd_treasury()
    
    def generate_cashflows(
        self,
        settlement: date,
        maturity: date,
        coupon_rate: float,
        face_value: float = 100.0,
        frequency: int = 2,
        accrual_start: Optional[date] = None,
    ) -> BondCashflows:
        """
        Generate bond cashflows.
        
        Args:
            settlement: Settlement date
            maturity: Maturity date
            coupon_rate: Annual coupon rate (decimal)
            face_value: Face/par value
            frequency: Coupons per year
            accrual_start: Optional contractual accrual start date

        Returns:
            BondCashflows object
        """
        schedule_start = accrual_start or _infer_previous_coupon_date(
            settlement,
            maturity,
            frequency,
            end_of_month=self.conventions.end_of_month,
        )
        schedule = generate_bond_schedule(
            settle=settlement,
            maturity=maturity,
            coupon_freq=frequency,
            day_count=self.conventions.day_count,
            holidays=self.conventions.holiday_calendar,
            business_day_convention=self.conventions.business_day,
            stub=self.conventions.stub,
            end_of_month=self.conventions.end_of_month,
            accrual_start=schedule_start,
        )
        cashflows = []
        
        for i, (pmt_date, yf) in enumerate(zip(schedule.payment_dates, schedule.year_fractions)):
            # Discounting is always from the curve anchor/valuation date.
            t = year_fraction(self.curve.anchor_date, pmt_date, self.curve.day_count)
            coupon_payment = face_value * coupon_rate * yf
            
            if i == len(schedule.payment_dates) - 1:
                # Final payment: coupon + principal
                cf = BondCashflow(
                    date=pmt_date,
                    amount=coupon_payment + face_value,
                    type="COUPON+PRINCIPAL",
                    year_fraction=t
                )
            else:
                cf = BondCashflow(
                    date=pmt_date,
                    amount=coupon_payment,
                    type="COUPON",
                    year_fraction=t
                )
            
            cashflows.append(cf)
        
        return BondCashflows(
            cashflows=cashflows,
            settlement_date=settlement,
            maturity_date=maturity,
            coupon_rate=coupon_rate,
            frequency=frequency,
            face_value=face_value,
            day_count=self.conventions.day_count
        )
    
    def present_value(
        self,
        cashflows: BondCashflows
    ) -> float:
        """
        Calculate present value of bond cashflows.
        
        Args:
            cashflows: Bond cashflows
            
        Returns:
            PV (dirty price) in currency units
        """
        pv = 0.0
        
        for cf in cashflows.cashflows:
            df = self.curve.discount_factor(cf.year_fraction)
            pv += cf.amount * df
        
        return pv
    
    def price(
        self,
        settlement: date,
        maturity: date,
        coupon_rate: float,
        face_value: float = 100.0,
        frequency: int = 2,
        accrual_start: Optional[date] = None,
    ) -> Tuple[float, float, float]:
        """
        Price a coupon bond.
        
        Args:
            settlement: Settlement date
            maturity: Maturity date
            coupon_rate: Annual coupon rate (decimal)
            face_value: Face value
            frequency: Coupons per year
            accrual_start: Optional contractual accrual start date
            
        Returns:
            Tuple of (dirty_price, clean_price, accrued_interest)
        """
        cashflows = self.generate_cashflows(
            settlement,
            maturity,
            coupon_rate,
            face_value,
            frequency,
            accrual_start=accrual_start,
        )
        
        dirty_price = self.present_value(cashflows)
        
        # Calculate accrued interest
        accrued = compute_accrued_interest(
            settlement,
            maturity,
            coupon_rate,
            face_value,
            frequency,
            self.conventions.day_count,
            business_day_convention=self.conventions.business_day,
            holidays=self.conventions.holiday_calendar,
            stub=self.conventions.stub,
            end_of_month=self.conventions.end_of_month,
            accrual_start=accrual_start,
        )
        
        clean_price = dirty_price - accrued
        
        return dirty_price, clean_price, accrued
    
    def price_zero_coupon(
        self,
        settlement: date,
        maturity: date,
        face_value: float = 100.0
    ) -> float:
        """
        Price a zero-coupon bond.
        
        Args:
            settlement: Settlement date
            maturity: Maturity date
            face_value: Face value
            
        Returns:
            Price
        """
        df = self.curve.discount_factor(maturity)
        return face_value * df
    
    def yield_to_maturity(
        self,
        price: float,
        settlement: date,
        maturity: date,
        coupon_rate: float,
        face_value: float = 100.0,
        frequency: int = 2,
        is_clean: bool = True,
        accrual_start: Optional[date] = None,
    ) -> float:
        """
        Compute yield to maturity from price.
        
        Uses Newton-Raphson to find yield that reprices the bond.
        
        Args:
            price: Market price (clean or dirty)
            settlement: Settlement date
            maturity: Maturity date
            coupon_rate: Coupon rate
            face_value: Face value
            frequency: Coupons per year
            is_clean: Whether price is clean (True) or dirty (False)
            accrual_start: Optional contractual accrual start date
            
        Returns:
            Yield to maturity (annual, bond-equivalent)
        """
        from scipy.optimize import brentq
        
        # If clean price, add accrued
        if is_clean:
            accrued = compute_accrued_interest(
                settlement,
                maturity,
                coupon_rate,
                face_value,
                frequency,
                self.conventions.day_count,
                business_day_convention=self.conventions.business_day,
                holidays=self.conventions.holiday_calendar,
                stub=self.conventions.stub,
                end_of_month=self.conventions.end_of_month,
                accrual_start=accrual_start,
            )
            target_dirty = price + accrued
        else:
            target_dirty = price
        
        # Generate cashflows for timing
        cashflows = self.generate_cashflows(
            settlement,
            maturity,
            coupon_rate,
            face_value,
            frequency,
            accrual_start=accrual_start,
        )
        
        def pv_at_yield(y):
            """PV using yield y (semi-annual bond equivalent)."""
            pv = 0.0
            for cf in cashflows.cashflows:
                df = (1 + y / frequency) ** (-cf.year_fraction * frequency)
                pv += cf.amount * df
            return pv - target_dirty
        
        # Solve for yield
        try:
            ytm = brentq(pv_at_yield, -0.5, 1.0, xtol=1e-10)
        except ValueError:
            # Fallback to approximate
            t = year_fraction(settlement, maturity, self.conventions.day_count)
            if t > 0:
                ytm = (face_value / target_dirty) ** (1/t) - 1
            else:
                ytm = 0.0
        
        return ytm
    
    def compute_dv01(
        self,
        settlement: date,
        maturity: date,
        coupon_rate: float,
        face_value: float = 100.0,
        frequency: int = 2,
        notional: float = 1_000_000,
        accrual_start: Optional[date] = None,
    ) -> float:
        """
        Compute bond DV01 using numerical differentiation.
        
        DV01 = -dP/dy * Notional / 10000
        
        Args:
            settlement: Settlement date
            maturity: Maturity date
            coupon_rate: Annual coupon rate (decimal)
            face_value: Face value
            frequency: Coupons per year
            notional: Notional amount
            accrual_start: Optional contractual accrual start date
            
        Returns:
            DV01 in currency units (positive for long position)
        """
        # Get current price
        dirty_price, _, _ = self.price(
            settlement,
            maturity,
            coupon_rate,
            face_value,
            frequency,
            accrual_start=accrual_start,
        )
        
        # Bump curve up by 1bp and reprice
        from ..risk.bumping import BumpEngine
        bump_engine = BumpEngine(self.curve)
        bumped_curve = bump_engine.parallel_bump(1)  # +1bp
        
        # Create new pricer with bumped curve
        bumped_pricer = BondPricer(bumped_curve, self.conventions)
        bumped_dirty, _, _ = bumped_pricer.price(
            settlement,
            maturity,
            coupon_rate,
            face_value,
            frequency,
            accrual_start=accrual_start,
        )
        
        # DV01 = -(P_up - P_base) * notional / face_value
        dv01 = -(bumped_dirty - dirty_price) * notional / face_value
        
        return dv01


def price_zero_coupon_bond(
    curve: Curve,
    settlement: date,
    maturity: date,
    face_value: float = 100.0,
    day_count: DayCount = DayCount.ACT_ACT
) -> float:
    """
    Price a zero-coupon bond.
    
    PV = Face * P(0, T)
    
    Args:
        curve: Discount curve
        settlement: Settlement date
        maturity: Maturity date
        face_value: Face value
        day_count: Day count convention
        
    Returns:
        Bond price
    """
    df = curve.discount_factor(maturity)
    return face_value * df


def price_coupon_bond(
    curve: Curve,
    settlement: date,
    maturity: date,
    coupon_rate: float,
    face_value: float = 100.0,
    frequency: int = 2,
    day_count: DayCount = DayCount.ACT_ACT,
    business_day_convention: BusinessDayConvention = BusinessDayConvention.FOLLOWING,
    holidays: CalendarInput = None,
    stub: str = "short_front",
    end_of_month: bool = False,
    accrual_start: Optional[date] = None,
) -> Tuple[float, float, float]:
    """
    Price a coupon bond.
    
    Args:
        curve: Discount curve
        settlement: Settlement date
        maturity: Maturity date
        coupon_rate: Annual coupon rate
        face_value: Face value
        frequency: Coupons per year
        day_count: Day count convention
        business_day_convention: Payment-date rolling rule
        holidays: Holiday calendar
        stub: Stub period handling
        end_of_month: Preserve end-of-month regular schedule dates
        accrual_start: Optional contractual accrual start date
        
    Returns:
        Tuple of (dirty_price, clean_price, accrued_interest)
    """
    conventions = Conventions(
        day_count=day_count,
        business_day=business_day_convention,
        payment_frequency=frequency,
        holiday_calendar=holidays,
        end_of_month=end_of_month,
        stub=stub,
    )
    pricer = BondPricer(curve, conventions)
    return pricer.price(
        settlement,
        maturity,
        coupon_rate,
        face_value,
        frequency,
        accrual_start=accrual_start,
    )


def compute_accrued_interest(
    settlement: date,
    maturity: date,
    coupon_rate: float,
    face_value: float = 100.0,
    frequency: int = 2,
    day_count: DayCount = DayCount.ACT_ACT,
    business_day_convention: BusinessDayConvention = BusinessDayConvention.FOLLOWING,
    holidays: CalendarInput = None,
    stub: str = "short_front",
    end_of_month: bool = False,
    accrual_start: Optional[date] = None,
) -> float:
    """
    Compute accrued interest.
    
    Args:
        settlement: Settlement date
        maturity: Maturity date
        coupon_rate: Annual coupon rate
        face_value: Face value
        frequency: Coupons per year
        day_count: Day count convention
        business_day_convention: Payment-date rolling rule
        holidays: Holiday calendar
        stub: Stub period handling
        end_of_month: Preserve end-of-month regular schedule dates
        accrual_start: Optional contractual accrual start date
        
    Returns:
        Accrued interest
    """
    if settlement >= maturity or frequency <= 0:
        return 0.0

    schedule_start = accrual_start or _infer_previous_coupon_date(
        settlement,
        maturity,
        frequency,
        end_of_month=end_of_month,
    )
    schedule = generate_bond_schedule(
        settle=settlement,
        maturity=maturity,
        coupon_freq=frequency,
        day_count=day_count,
        holidays=holidays,
        business_day_convention=business_day_convention,
        stub=stub,
        end_of_month=end_of_month,
        accrual_start=schedule_start,
    )

    if not schedule.payment_dates:
        return 0.0

    period_start = schedule.accrual_starts[0]
    period_end = schedule.accrual_ends[0]
    period_fraction = year_fraction(period_start, period_end, day_count)
    accrued_fraction = year_fraction(period_start, settlement, day_count)

    if period_fraction <= 0.0 or accrued_fraction <= 0.0:
        return 0.0

    coupon_payment = face_value * coupon_rate * period_fraction
    return coupon_payment * (accrued_fraction / period_fraction)


def _infer_previous_coupon_date(
    settlement: date,
    maturity: date,
    frequency: int,
    end_of_month: bool = False,
) -> date:
    """Infer the previous coupon boundary from maturity and coupon frequency."""
    if frequency <= 0 or 12 % frequency != 0:
        raise ValueError("Frequency must be a positive divisor of 12")
    if settlement >= maturity:
        return maturity

    months_per_period = 12 // frequency
    previous_boundary = maturity

    while True:
        candidate = _shift_months(
            previous_boundary,
            -months_per_period,
            end_of_month=end_of_month,
        )
        if candidate <= settlement:
            return candidate
        previous_boundary = candidate


def _shift_months(start: date, months: int, end_of_month: bool = False) -> date:
    total_months = (start.year * 12 + (start.month - 1)) + months
    year = total_months // 12
    month = total_months % 12 + 1
    preserve_eom = end_of_month and _is_end_of_month(start)
    if preserve_eom:
        day = _days_in_month(year, month)
    else:
        day = min(start.day, _days_in_month(year, month))
    return date(year, month, day)


def _is_end_of_month(d: date) -> bool:
    return d.day == _days_in_month(d.year, d.month)


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
    "BondPricer",
    "BondCashflow",
    "BondCashflows",
    "price_zero_coupon_bond",
    "price_coupon_bond",
    "compute_accrued_interest",
]
