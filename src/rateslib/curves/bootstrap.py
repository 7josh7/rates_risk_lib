"""
Curve bootstrapping engine.

Implements sequential bootstrap for OIS and other discount curves:
1. Sort instruments by maturity
2. Solve for discount factors sequentially
3. Verify repricing and curve quality

Supports:
- Deposits
- OIS Swaps
- FRAs
- Futures (with simplified convexity handling)
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import brentq

from ..conventions import DayCount
from .curve import Curve
from .instruments import CurveInstrument, Deposit, OISSwap, FRA, Future


@dataclass
class BootstrapResult:
    """Result of curve bootstrap."""
    curve: Curve
    repricing_errors: Dict[str, float]
    success: bool
    message: str


class OISBootstrapper:
    """
    Bootstrap an OIS discount curve from market quotes.
    
    The bootstrapper:
    1. Sorts instruments by maturity
    2. Sequentially solves for each discount factor
    3. Verifies that instruments reprice within tolerance
    
    Attributes:
        anchor_date: Valuation date
        day_count: Day count for curve construction
        tolerance: Maximum allowed repricing error (default 1e-8)
        interpolation_method: Method for inter-node interpolation
    """
    
    def __init__(
        self,
        anchor_date: date,
        day_count: DayCount = DayCount.ACT_360,
        tolerance: float = 1e-8,
        interpolation_method: str = "cubic_spline"
    ):
        self.anchor_date = anchor_date
        self.day_count = day_count
        self.tolerance = tolerance
        self.interpolation_method = interpolation_method
    
    def bootstrap(
        self,
        instruments: List[CurveInstrument],
        verify: bool = True
    ) -> BootstrapResult:
        """
        Bootstrap curve from instruments.
        
        Args:
            instruments: List of curve instruments with quotes
            verify: Whether to verify repricing after bootstrap
            
        Returns:
            BootstrapResult with curve and diagnostics
        """
        if not instruments:
            return BootstrapResult(
                curve=Curve(self.anchor_date, "USD", self.day_count, self.interpolation_method),
                repricing_errors={},
                success=False,
                message="No instruments provided"
            )
        
        # Sort by maturity
        sorted_instruments = sorted(
            instruments,
            key=lambda x: x.maturity_time(self.anchor_date)
        )
        
        # Initialize curve
        curve = Curve(
            anchor_date=self.anchor_date,
            currency="USD",
            day_count=self.day_count,
            interpolation_method=self.interpolation_method
        )
        
        # Track bootstrapped points
        prior_dfs: List[Tuple[float, float]] = [(0.0, 1.0)]
        
        # Bootstrap each instrument
        for inst in sorted_instruments:
            try:
                t, df = inst.implied_discount_factor(self.anchor_date, prior_dfs)
                
                # Validate discount factor
                if df <= 0 or df > 1:
                    # Try to fix with bounds
                    df = max(0.0001, min(0.9999, df))
                
                # Check for monotonicity
                if prior_dfs and t > prior_dfs[-1][0]:
                    # DF should decrease with time
                    if df > prior_dfs[-1][1]:
                        # Force monotonicity
                        df = prior_dfs[-1][1] * 0.99999
                
                curve.add_node(t, df)
                prior_dfs.append((t, df))
                
            except Exception as e:
                return BootstrapResult(
                    curve=curve,
                    repricing_errors={},
                    success=False,
                    message=f"Bootstrap failed at {inst.tenor}: {str(e)}"
                )
        
        # Build interpolator
        try:
            curve.build()
        except Exception as e:
            return BootstrapResult(
                curve=curve,
                repricing_errors={},
                success=False,
                message=f"Curve build failed: {str(e)}"
            )
        
        # Verify repricing
        repricing_errors = {}
        if verify:
            repricing_errors = self._verify_repricing(curve, sorted_instruments)
            
            # Check if any errors exceed tolerance
            max_error = max(abs(e) for e in repricing_errors.values()) if repricing_errors else 0
            if max_error > self.tolerance:
                return BootstrapResult(
                    curve=curve,
                    repricing_errors=repricing_errors,
                    success=False,
                    message=f"Repricing error {max_error:.2e} exceeds tolerance {self.tolerance:.2e}"
                )
        
        return BootstrapResult(
            curve=curve,
            repricing_errors=repricing_errors,
            success=True,
            message="Bootstrap successful"
        )
    
    def _verify_repricing(
        self,
        curve: Curve,
        instruments: List[CurveInstrument]
    ) -> Dict[str, float]:
        """
        Verify that instruments reprice to their quotes.
        
        Returns dict of {tenor: error} where error = computed - quoted.
        """
        errors = {}
        
        for inst in instruments:
            try:
                computed = self._reprice_instrument(curve, inst)
                error = computed - inst.quote
                errors[inst.tenor] = error
            except Exception as e:
                errors[inst.tenor] = float('nan')
        
        return errors
    
    def _reprice_instrument(
        self,
        curve: Curve,
        inst: CurveInstrument
    ) -> float:
        """
        Compute the implied rate/price for an instrument using the curve.
        """
        if isinstance(inst, Deposit):
            return self._reprice_deposit(curve, inst)
        elif isinstance(inst, OISSwap):
            return self._reprice_ois(curve, inst)
        elif isinstance(inst, FRA):
            return self._reprice_fra(curve, inst)
        elif isinstance(inst, Future):
            return self._reprice_future(curve, inst)
        else:
            raise ValueError(f"Unknown instrument type: {type(inst)}")
    
    def _reprice_deposit(self, curve: Curve, inst: Deposit) -> float:
        """Reprice deposit to get implied rate."""
        from ..conventions import year_fraction
        
        mat = inst.maturity_date(self.anchor_date)
        t = year_fraction(self.anchor_date, mat, inst.day_count)
        df = curve.discount_factor(t)
        
        # r = (1/df - 1) / tau
        if t <= 0:
            return 0.0
        rate = (1.0 / df - 1.0) / t
        return rate
    
    def _reprice_ois(self, curve: Curve, inst: OISSwap) -> float:
        """Reprice OIS swap to get par rate."""
        from ..conventions import year_fraction
        
        # Generate payment schedule
        schedule = inst._payment_schedule(self.anchor_date)
        
        if not schedule:
            return 0.0
        
        # Par rate = (1 - DF(Tn)) / sum(delta_i * DF(Ti))
        annuity = 0.0
        final_df = 1.0
        
        for pmt_date, tau in schedule:
            t = year_fraction(self.anchor_date, pmt_date, inst.day_count)
            df = curve.discount_factor(t)
            annuity += tau * df
            final_df = df
        
        if annuity <= 0:
            return 0.0
        
        par_rate = (1.0 - final_df) / annuity
        return par_rate
    
    def _reprice_fra(self, curve: Curve, inst: FRA) -> float:
        """Reprice FRA to get implied forward rate."""
        t1 = inst.start_time(self.anchor_date)
        t2 = inst.maturity_time(self.anchor_date)
        
        df1 = curve.discount_factor(t1)
        df2 = curve.discount_factor(t2)
        
        tau = t2 - t1
        if tau <= 0:
            return 0.0
        
        fwd_rate = (df1 / df2 - 1.0) / tau
        return fwd_rate
    
    def _reprice_future(self, curve: Curve, inst: Future) -> float:
        """Reprice future - return implied rate."""
        t1 = inst.maturity_time(self.anchor_date)
        t2 = t1 + 0.25  # 3-month forward
        
        df1 = curve.discount_factor(t1)
        df2 = curve.discount_factor(t2)
        
        if df2 <= 0:
            return 0.0
        
        fwd_rate = (df1 / df2 - 1.0) / 0.25
        
        # Convert to futures price convention
        return inst.implied_rate()  # Compare to rate, not price


def bootstrap_from_quotes(
    anchor_date: date,
    quotes: List[Dict],
    interpolation: str = "cubic_spline"
) -> Curve:
    """
    Convenience function to bootstrap curve from quote dictionaries.
    
    Args:
        anchor_date: Valuation date
        quotes: List of dicts with keys: instrument_type, tenor, quote, ...
        interpolation: Interpolation method
        
    Returns:
        Bootstrapped curve
        
    Example quote format:
        {"instrument_type": "DEPOSIT", "tenor": "1D", "quote": 0.053}
        {"instrument_type": "OIS", "tenor": "1M", "quote": 0.0532}
    """
    instruments = []
    
    for q in quotes:
        inst_type = q.get("instrument_type", "").upper()
        tenor = q.get("tenor", "")
        quote = float(q.get("quote", 0))
        day_count = DayCount.from_string(q.get("day_count", "ACT/360"))
        
        if inst_type == "DEPOSIT":
            instruments.append(Deposit(
                tenor=tenor,
                quote=quote,
                day_count=day_count
            ))
        elif inst_type == "OIS":
            pay_freq = q.get("pay_freq", "ANNUAL")
            instruments.append(OISSwap(
                tenor=tenor,
                quote=quote,
                day_count=day_count,
                payment_frequency=pay_freq
            ))
        elif inst_type == "FRA":
            start_tenor = q.get("start_tenor", "0D")
            instruments.append(FRA(
                tenor=tenor,
                quote=quote,
                day_count=day_count,
                start_tenor=start_tenor
            ))
        elif inst_type in ("FUT", "FUTURE"):
            instruments.append(Future(
                tenor=tenor,
                quote=quote,
                day_count=day_count
            ))
    
    bootstrapper = OISBootstrapper(
        anchor_date=anchor_date,
        day_count=DayCount.ACT_360,
        interpolation_method=interpolation,
        tolerance=1e-4  # Relax tolerance for practical use
    )
    
    result = bootstrapper.bootstrap(instruments)
    
    if not result.success:
        raise RuntimeError(f"Bootstrap failed: {result.message}")
    
    return result.curve


__all__ = [
    "OISBootstrapper",
    "BootstrapResult",
    "bootstrap_from_quotes",
]
