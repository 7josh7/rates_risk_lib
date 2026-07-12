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

from dataclasses import dataclass, field
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
    diagnostics: Dict[str, object] = field(default_factory=dict)


class CurveBootstrapError(ValueError):
    """Raised when market inputs cannot produce a trustworthy curve."""


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
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Bootstrap tolerance must be positive and finite")
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

        try:
            self._validate_instruments(instruments)
        except (TypeError, ValueError) as exc:
            return BootstrapResult(
                curve=Curve(self.anchor_date, "USD", self.day_count, self.interpolation_method),
                repricing_errors={},
                success=False,
                message=f"Invalid bootstrap inputs: {exc}",
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
                
                if not np.isfinite(t) or t <= 0:
                    raise CurveBootstrapError(
                        f"{type(inst).__name__} {inst.tenor} produced invalid maturity {t!r}"
                    )
                if not np.isfinite(df) or df <= 0:
                    raise CurveBootstrapError(
                        f"{type(inst).__name__} {inst.tenor} produced invalid discount factor {df!r}"
                    )
                if prior_dfs and t <= prior_dfs[-1][0] + 1e-12:
                    raise CurveBootstrapError(
                        f"Duplicate or non-increasing curve maturity at {inst.tenor} (t={t})"
                    )
                
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
        
        try:
            diagnostics = self._curve_diagnostics(curve)
        except (TypeError, ValueError, FloatingPointError) as exc:
            return BootstrapResult(
                curve=curve,
                repricing_errors={},
                success=False,
                message=f"Curve diagnostics failed: {exc}",
            )

        # Verify repricing
        repricing_errors = {}
        if verify:
            repricing_errors = self._verify_repricing(curve, sorted_instruments)
            
            non_finite = {
                tenor: error
                for tenor, error in repricing_errors.items()
                if not np.isfinite(error)
            }
            if non_finite:
                return BootstrapResult(
                    curve=curve,
                    repricing_errors=repricing_errors,
                    success=False,
                    message=f"Non-finite repricing errors for {sorted(non_finite)}",
                    diagnostics=diagnostics,
                )
            max_error = max(abs(e) for e in repricing_errors.values()) if repricing_errors else 0.0
            if max_error > self.tolerance:
                return BootstrapResult(
                    curve=curve,
                    repricing_errors=repricing_errors,
                    success=False,
                    message=f"Repricing error {max_error:.2e} exceeds tolerance {self.tolerance:.2e}",
                    diagnostics=diagnostics,
                )
        
        return BootstrapResult(
            curve=curve,
            repricing_errors=repricing_errors,
            success=True,
            message="Bootstrap successful",
            diagnostics=diagnostics,
        )

    def _validate_instruments(self, instruments: List[CurveInstrument]) -> None:
        seen: set[tuple[str, str, str]] = set()
        for instrument in instruments:
            if not isinstance(instrument, CurveInstrument):
                raise TypeError(f"Expected CurveInstrument, got {type(instrument).__name__}")
            if not instrument.tenor.strip():
                raise ValueError("Instrument tenor cannot be empty")
            if not np.isfinite(instrument.quote):
                raise ValueError(f"Quote for {instrument.tenor} must be finite")
            maturity = instrument.maturity_time(self.anchor_date)
            if not np.isfinite(maturity) or maturity <= 0:
                raise ValueError(f"Maturity for {instrument.tenor} must be positive and finite")
            key = (type(instrument).__name__, instrument.tenor, getattr(instrument, "start_tenor", ""))
            if key in seen:
                raise ValueError(f"Duplicate instrument quote: {key}")
            seen.add(key)

    def _curve_diagnostics(self, curve: Curve) -> Dict[str, object]:
        times = curve.get_node_times()
        horizon = float(times[-1])
        grid = np.linspace(0.0, horizon, max(25, len(times) * 5))
        dfs = np.array([curve.discount_factor(float(t)) for t in grid], dtype=float)
        forwards = np.array(
            [curve.forward_rate(float(grid[i]), float(grid[i + 1])) for i in range(len(grid) - 1)],
            dtype=float,
        )
        if not np.all(np.isfinite(dfs)) or np.any(dfs <= 0):
            raise CurveBootstrapError("Curve interpolation produced invalid discount factors")
        if not np.all(np.isfinite(forwards)):
            raise CurveBootstrapError("Curve interpolation produced non-finite forward rates")
        return {
            "node_count": len(times),
            "min_discount_factor": float(np.min(dfs)),
            "max_discount_factor": float(np.max(dfs)),
            "min_simple_forward": float(np.min(forwards)),
            "max_simple_forward": float(np.max(forwards)),
            "supports_negative_rates": bool(np.max(dfs) > 1.0 or np.min(forwards) < 0.0),
        }
    
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
        """Reprice future using the curve and return the quote in the instrument's convention."""
        t1 = inst.maturity_time(self.anchor_date)
        t2 = t1 + 0.25  # 3-month forward
        
        df1 = curve.discount_factor(t1)
        df2 = curve.discount_factor(t2)
        
        if df2 <= 0:
            return 0.0
        
        fwd_rate = (df1 / df2 - 1.0) / 0.25

        # Match the input quote convention so verification compares like-with-like.
        if str(inst.quote_type).upper() == "RATE" and inst.quote <= 1.0:
            return fwd_rate

        return 100.0 - fwd_rate * 100.0


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
    if not quotes:
        raise CurveBootstrapError("At least one market quote is required")

    instruments = []
    seen_quotes: set[tuple[str, str, str]] = set()
    
    for row_number, q in enumerate(quotes, start=1):
        if not isinstance(q, dict):
            raise CurveBootstrapError(f"Quote row {row_number} must be a dictionary")
        inst_type = str(q.get("instrument_type", "") or "").strip().upper()
        tenor = str(q.get("tenor", "")).strip().upper()
        if not inst_type:
            raise CurveBootstrapError(f"Quote row {row_number} is missing instrument_type")
        if not tenor:
            raise CurveBootstrapError(f"Quote row {row_number} is missing tenor")
        if "quote" not in q:
            raise CurveBootstrapError(f"Quote row {row_number} is missing quote")
        try:
            quote = float(q["quote"])
        except (TypeError, ValueError) as exc:
            raise CurveBootstrapError(
                f"Quote row {row_number} has a non-numeric quote: {q['quote']!r}"
            ) from exc
        if not np.isfinite(quote):
            raise CurveBootstrapError(
                f"Quote row {row_number} for {inst_type} {tenor} must be finite"
            )
        try:
            day_count = DayCount.from_string(q.get("day_count", "ACT/360"))
        except (TypeError, ValueError) as exc:
            raise CurveBootstrapError(
                f"Quote row {row_number} has invalid day_count {q.get('day_count')!r}"
            ) from exc

        start_tenor = str(q.get("start_tenor", "")).strip().upper()
        quote_key = (inst_type, tenor, start_tenor)
        if quote_key in seen_quotes:
            raise CurveBootstrapError(f"Duplicate market quote at row {row_number}: {quote_key}")
        seen_quotes.add(quote_key)
        
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
            start_tenor = start_tenor or "0D"
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
                day_count=day_count,
                quote_type=str(q.get("quote_type", "RATE" if quote <= 1.0 else "PRICE")),
            ))
        else:
            raise CurveBootstrapError(
                f"Unsupported instrument_type {inst_type!r} at quote row {row_number}"
            )
    
    bootstrapper = OISBootstrapper(
        anchor_date=anchor_date,
        day_count=DayCount.ACT_360,
        interpolation_method=interpolation,
        tolerance=1e-4  # Relax tolerance for practical use
    )
    
    result = bootstrapper.bootstrap(instruments)
    
    if not result.success:
        raise CurveBootstrapError(f"Bootstrap failed: {result.message}")
    
    return result.curve


__all__ = [
    "OISBootstrapper",
    "BootstrapResult",
    "CurveBootstrapError",
    "bootstrap_from_quotes",
]
