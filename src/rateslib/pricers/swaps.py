"""
Interest rate swap pricing engine.

Prices vanilla fixed-float interest rate swaps using:
- Single-curve approach (discount = forward curve)
- Multi-curve approach (separate discount and projection curves) - optional

Conventions (USD):
- Fixed leg: Semi-annual, 30/360 or ACT/360
- Floating leg: Quarterly 3M LIBOR/SOFR, ACT/360

Pricing formula (single-curve):
    PV_swap = PV_float - PV_fixed
    
    PV_fixed = K * sum(delta_i * DF(T_i))
    PV_float ≈ 1 - DF(T_n)  (par at inception)
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple

import numpy as np

from ..conventions import DayCount, year_fraction, Conventions
from ..dates import DateUtils, generate_swap_schedule, ScheduleInfo
from ..curves.curve import Curve


@dataclass
class SwapLegCashflow:
    """A single swap leg cashflow."""
    date: date
    amount: float  # Fixed amount or projected floating amount
    accrual_start: date
    accrual_end: date
    year_fraction: float
    discount_factor: float = 1.0
    forward_rate: Optional[float] = None  # For floating leg


@dataclass
class SwapCashflows:
    """Complete swap cashflows for both legs."""
    fixed_leg: List[SwapLegCashflow]
    floating_leg: List[SwapLegCashflow]
    notional: float
    fixed_rate: float
    effective_date: date
    maturity_date: date
    pay_receive: str  # "PAY" or "RECEIVE" (fixed leg)
    
    @property
    def pv_fixed(self) -> float:
        """PV of fixed leg."""
        return sum(cf.amount * cf.discount_factor for cf in self.fixed_leg)
    
    @property
    def pv_floating(self) -> float:
        """PV of floating leg."""
        return sum(cf.amount * cf.discount_factor for cf in self.floating_leg)
    
    @property
    def net_pv(self) -> float:
        """Net PV (positive = value to fixed payer)."""
        if self.pay_receive == "PAY":
            return self.pv_floating - self.pv_fixed
        else:
            return self.pv_fixed - self.pv_floating


class SwapPricer:
    """
    Interest rate swap pricing engine.
    
    Supports:
    - Single-curve pricing (same curve for discounting and projection)
    - Multi-curve pricing (separate discount and projection curves)
    
    Attributes:
        discount_curve: Curve for discounting cashflows
        projection_curve: Curve for projecting forward rates (optional, defaults to discount)
    """
    
    def __init__(
        self,
        discount_curve: Curve,
        projection_curve: Optional[Curve] = None,
        fixed_conventions: Optional[Conventions] = None,
        float_conventions: Optional[Conventions] = None
    ):
        self.discount_curve = discount_curve
        self.projection_curve = projection_curve or discount_curve
        
        # Default USD swap conventions
        self.fixed_conventions = fixed_conventions or Conventions(
            day_count=DayCount.ACT_360,
            payment_frequency=2  # Semi-annual
        )
        self.float_conventions = float_conventions or Conventions(
            day_count=DayCount.ACT_360,
            payment_frequency=4  # Quarterly
        )

    def forward_swap_rate(self, expiry: float, tenor: float) -> Tuple[float, float]:
        """
        Compute the forward par swap rate and annuity for a swap that starts at
        `expiry` (in years) and runs for `tenor` (in years).

        Uses the projection curve for forwards and the discount curve for the
        annuity, aligned with the current fixed leg payment frequency.

        Args:
            expiry: time to start (years)
            tenor: swap length (years) starting from expiry

        Returns:
            (forward_rate, annuity)
        """
        fixed_freq = self.fixed_conventions.payment_frequency
        n_periods = int(tenor * fixed_freq)

        payment_times = [expiry + (i + 1) / fixed_freq for i in range(n_periods)]

        annuity = 0.0
        for t in payment_times:
            delta = 1.0 / fixed_freq  # simple accrual consistent with the frequency
            df = self.discount_curve.discount_factor(t)
            annuity += delta * df

        df_start = self.projection_curve.discount_factor(expiry)
        df_end = self.projection_curve.discount_factor(expiry + tenor)

        if annuity > 0:
            forward_rate = (df_start - df_end) / annuity
        else:
            forward_rate = 0.0

        return forward_rate, annuity
    
    def generate_cashflows(
        self,
        effective: date,
        maturity: date,
        notional: float,
        fixed_rate: float,
        pay_receive: str = "PAY"
    ) -> SwapCashflows:
        """
        Generate swap cashflows for both legs.
        
        Args:
            effective: Swap effective date
            maturity: Swap maturity
            notional: Notional amount
            fixed_rate: Fixed rate (decimal)
            pay_receive: "PAY" or "RECEIVE" fixed
            
        Returns:
            SwapCashflows object with both legs
        """
        # Generate schedules
        fixed_schedule, float_schedule = generate_swap_schedule(
            effective, maturity,
            self.fixed_conventions.payment_frequency,
            self.float_conventions.payment_frequency,
            self.fixed_conventions.day_count,
            self.float_conventions.day_count
        )
        
        # Fixed leg cashflows
        fixed_cfs = []
        for i, pmt_date in enumerate(fixed_schedule.payment_dates):
            yf = fixed_schedule.year_fractions[i]
            t = year_fraction(effective, pmt_date, self.fixed_conventions.day_count)
            df = self.discount_curve.discount_factor(t)
            
            amount = notional * fixed_rate * yf
            
            fixed_cfs.append(SwapLegCashflow(
                date=pmt_date,
                amount=amount,
                accrual_start=fixed_schedule.accrual_starts[i],
                accrual_end=fixed_schedule.accrual_ends[i],
                year_fraction=yf,
                discount_factor=df
            ))
        
        # Floating leg cashflows
        float_cfs = []
        for i, pmt_date in enumerate(float_schedule.payment_dates):
            yf = float_schedule.year_fractions[i]
            t = year_fraction(effective, pmt_date, self.float_conventions.day_count)
            df = self.discount_curve.discount_factor(t)
            
            # Get forward rate for this period
            t_start = year_fraction(effective, float_schedule.accrual_starts[i], 
                                   self.float_conventions.day_count)
            t_end = year_fraction(effective, float_schedule.accrual_ends[i],
                                 self.float_conventions.day_count)
            
            if t_start >= 0:
                fwd_rate = self.projection_curve.forward_rate(t_start, t_end)
            else:
                # First fixing may be historical
                fwd_rate = self.projection_curve.zero_rate(t_end)
            
            amount = notional * fwd_rate * yf
            
            float_cfs.append(SwapLegCashflow(
                date=pmt_date,
                amount=amount,
                accrual_start=float_schedule.accrual_starts[i],
                accrual_end=float_schedule.accrual_ends[i],
                year_fraction=yf,
                discount_factor=df,
                forward_rate=fwd_rate
            ))
        
        return SwapCashflows(
            fixed_leg=fixed_cfs,
            floating_leg=float_cfs,
            notional=notional,
            fixed_rate=fixed_rate,
            effective_date=effective,
            maturity_date=maturity,
            pay_receive=pay_receive.upper()
        )
    
    def present_value(
        self,
        effective: date,
        maturity: date,
        notional: float,
        fixed_rate: float,
        pay_receive: str = "PAY"
    ) -> float:
        """
        Calculate swap present value.
        
        Args:
            effective: Effective date
            maturity: Maturity date
            notional: Notional amount
            fixed_rate: Fixed rate (decimal)
            pay_receive: "PAY" or "RECEIVE" fixed
            
        Returns:
            Swap PV (positive = in-the-money for the specified direction)
        """
        cashflows = self.generate_cashflows(
            effective, maturity, notional, fixed_rate, pay_receive
        )
        return cashflows.net_pv
    
    def par_rate(
        self,
        effective: date,
        maturity: date
    ) -> float:
        """
        Calculate par swap rate (rate at which PV = 0).
        
        For single-curve: R = (1 - DF(Tn)) / Annuity
        
        Args:
            effective: Effective date
            maturity: Maturity date
            
        Returns:
            Par swap rate (decimal)
        """
        # Generate fixed leg schedule for annuity calculation
        fixed_schedule, _ = generate_swap_schedule(
            effective, maturity,
            self.fixed_conventions.payment_frequency,
            self.float_conventions.payment_frequency,
            self.fixed_conventions.day_count,
            self.float_conventions.day_count
        )
        
        # Calculate annuity
        annuity = 0.0
        final_t = 0.0
        
        for i, pmt_date in enumerate(fixed_schedule.payment_dates):
            yf = fixed_schedule.year_fractions[i]
            t = year_fraction(effective, pmt_date, self.fixed_conventions.day_count)
            df = self.discount_curve.discount_factor(t)
            annuity += yf * df
            final_t = t
        
        # Par rate for single-curve
        final_df = self.discount_curve.discount_factor(final_t)
        
        if annuity <= 0:
            return 0.0
        
        par_rate = (1.0 - final_df) / annuity
        return par_rate
    
    def dv01(
        self,
        effective: date,
        maturity: date,
        notional: float,
        fixed_rate: float,
        pay_receive: str = "PAY"
    ) -> float:
        """
        Calculate swap DV01 (dollar value of 1bp).
        
        Approximation: DV01 ≈ Notional * Annuity / 10000
        
        Args:
            effective: Effective date
            maturity: Maturity date
            notional: Notional amount
            fixed_rate: Fixed rate
            pay_receive: Direction
            
        Returns:
            DV01 in currency units
        """
        fixed_schedule, _ = generate_swap_schedule(
            effective, maturity,
            self.fixed_conventions.payment_frequency,
            self.float_conventions.payment_frequency,
            self.fixed_conventions.day_count,
            self.float_conventions.day_count
        )
        
        # Calculate annuity
        annuity = 0.0
        for i, pmt_date in enumerate(fixed_schedule.payment_dates):
            yf = fixed_schedule.year_fractions[i]
            t = year_fraction(effective, pmt_date, self.fixed_conventions.day_count)
            df = self.discount_curve.discount_factor(t)
            annuity += yf * df
        
        # DV01 = Notional * Annuity / 10000
        dv01 = notional * annuity / 10000
        
        # Sign convention: pay fixed means losing value when rates rise
        if pay_receive.upper() == "PAY":
            return -dv01
        else:
            return dv01


def price_vanilla_swap(
    discount_curve: Curve,
    effective: date,
    maturity: date,
    notional: float,
    fixed_rate: float,
    pay_receive: str = "PAY",
    projection_curve: Optional[Curve] = None
) -> float:
    """
    Price a vanilla interest rate swap.
    
    Args:
        discount_curve: Curve for discounting
        effective: Effective date
        maturity: Maturity date
        notional: Notional amount
        fixed_rate: Fixed rate (decimal)
        pay_receive: "PAY" or "RECEIVE" fixed
        projection_curve: Curve for forward rates (optional)
        
    Returns:
        Swap PV
    """
    pricer = SwapPricer(discount_curve, projection_curve)
    return pricer.present_value(effective, maturity, notional, fixed_rate, pay_receive)


def compute_swap_par_rate(
    curve: Curve,
    effective: date,
    maturity: date
) -> float:
    """
    Compute par swap rate.
    
    Args:
        curve: Discount/projection curve
        effective: Effective date
        maturity: Maturity date
        
    Returns:
        Par rate (decimal)
    """
    pricer = SwapPricer(curve)
    return pricer.par_rate(effective, maturity)


__all__ = [
    "SwapPricer",
    "SwapLegCashflow",
    "SwapCashflows",
    "price_vanilla_swap",
    "compute_swap_par_rate",
]
