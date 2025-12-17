"""
Stressed VaR calculation.

Computes VaR using a specified historical stress period
(e.g., 2020-2022 COVID/rate hiking period).

Regulatory context:
- Basel III requires stressed VaR using a 12-month period
  of significant financial stress relevant to the portfolio
- Stressed VaR is typically higher than regular VaR

Common stress periods for USD rates:
- 2008-2009: Financial crisis
- 2013: Taper tantrum
- 2020: COVID crash
- 2022: Rate hiking cycle
"""

from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..curves.curve import Curve
from .historical import HistoricalSimulation, HistoricalVaRResult


@dataclass
class StressResult:
    """
    Stressed VaR result.
    
    Attributes:
        stressed_var_95: 95% VaR from stress period
        stressed_var_99: 99% VaR from stress period
        regular_var_95: Regular VaR for comparison
        regular_var_99: Regular VaR for comparison
        stress_period_start: Start of stress window
        stress_period_end: End of stress window
        num_stress_scenarios: Number of scenarios in stress period
    """
    stressed_var_95: float
    stressed_var_99: float
    stressed_es_95: float
    stressed_es_99: float
    regular_var_95: float
    regular_var_99: float
    stress_period_start: date
    stress_period_end: date
    num_stress_scenarios: int
    stress_multiplier_95: float = 0.0
    stress_multiplier_99: float = 0.0
    
    def __post_init__(self):
        if self.regular_var_95 > 0:
            self.stress_multiplier_95 = self.stressed_var_95 / self.regular_var_95
        if self.regular_var_99 > 0:
            self.stress_multiplier_99 = self.stressed_var_99 / self.regular_var_99
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "stressed_var_95": self.stressed_var_95,
            "stressed_var_99": self.stressed_var_99,
            "stressed_es_95": self.stressed_es_95,
            "stressed_es_99": self.stressed_es_99,
            "regular_var_95": self.regular_var_95,
            "regular_var_99": self.regular_var_99,
            "stress_period_start": self.stress_period_start.isoformat(),
            "stress_period_end": self.stress_period_end.isoformat(),
            "num_stress_scenarios": self.num_stress_scenarios,
            "stress_multiplier_95": self.stress_multiplier_95,
            "stress_multiplier_99": self.stress_multiplier_99
        }


class StressedVaR:
    """
    Stressed VaR calculator.
    
    Computes VaR using only historical scenarios from
    a specified stress period.
    """
    
    # Predefined stress periods
    STRESS_PERIODS = {
        "COVID_2020": (date(2020, 2, 1), date(2020, 12, 31)),
        "RATE_HIKE_2022": (date(2022, 1, 1), date(2022, 12, 31)),
        "TAPER_2013": (date(2013, 5, 1), date(2013, 9, 30)),
        "GFC_2008": (date(2008, 9, 1), date(2009, 3, 31)),
        "FULL_2020_2022": (date(2020, 3, 1), date(2022, 12, 31)),
    }
    
    def __init__(
        self,
        base_curve: Curve,
        historical_data: pd.DataFrame,
        pricer_func: Callable[[Curve], float],
        stress_period_start: date,
        stress_period_end: date
    ):
        """
        Initialize stressed VaR calculator.
        
        Args:
            base_curve: Current yield curve
            historical_data: Full historical data
            pricer_func: Portfolio pricer
            stress_period_start: Start of stress period
            stress_period_end: End of stress period
        """
        self.base_curve = base_curve
        self.full_historical_data = historical_data
        self.pricer_func = pricer_func
        self.stress_period_start = stress_period_start
        self.stress_period_end = stress_period_end
        
        # Filter to stress period
        self._filter_stress_period()
    
    def _filter_stress_period(self) -> None:
        """Filter historical data to stress period."""
        df = self.full_historical_data.copy()
        self.used_fallback_full_history = False
        if "rate" not in df.columns and "date" in df.columns:
            value_cols = [c for c in df.columns if c.lower() != "date"]
            df = df.melt(id_vars="date", value_vars=value_cols, var_name="tenor", value_name="rate")

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            mask = (df['date'] >= pd.Timestamp(self.stress_period_start)) & \
                   (df['date'] <= pd.Timestamp(self.stress_period_end))
            self.stress_data = df[mask].copy()
        else:
            # Assume index is date
            df.index = pd.to_datetime(df.index)
            mask = (df.index >= pd.Timestamp(self.stress_period_start)) & \
                   (df.index <= pd.Timestamp(self.stress_period_end))
            self.stress_data = df[mask].copy()

        # If no data in the selected stress window, fall back to full history
        if self.stress_data.empty:
            self.stress_data = df.copy()
            self.used_fallback_full_history = True
    
    def compute_stressed_var(self) -> StressResult:
        """
        Compute stressed VaR.
        
        Returns:
            StressResult with stressed and regular VaR
        """
        # Stressed VaR
        stressed_hs = HistoricalSimulation(
            self.base_curve,
            self.stress_data,
            self.pricer_func
        )
        stressed_result = stressed_hs.run_simulation()
        
        # Regular VaR (all data)
        regular_hs = HistoricalSimulation(
            self.base_curve,
            self.full_historical_data,
            self.pricer_func
        )
        regular_result = regular_hs.run_simulation()

        if stressed_result.num_scenarios == 0:
            raise ValueError("No scenarios in selected stress period. Choose a period with available data.")
        
        return StressResult(
            stressed_var_95=stressed_result.var_95,
            stressed_var_99=stressed_result.var_99,
            stressed_es_95=stressed_result.es_95,
            stressed_es_99=stressed_result.es_99,
            regular_var_95=regular_result.var_95,
            regular_var_99=regular_result.var_99,
            stress_period_start=self.stress_period_start,
            stress_period_end=self.stress_period_end,
            num_stress_scenarios=stressed_result.num_scenarios
        )
    
    @classmethod
    def from_predefined_period(
        cls,
        base_curve: Curve,
        historical_data: pd.DataFrame,
        pricer_func: Callable[[Curve], float],
        period_name: str
    ) -> "StressedVaR":
        """
        Create StressedVaR from predefined stress period.
        
        Args:
            base_curve: Current curve
            historical_data: Historical data
            pricer_func: Portfolio pricer
            period_name: One of "COVID_2020", "RATE_HIKE_2022", etc.
            
        Returns:
            StressedVaR instance
        """
        if period_name not in cls.STRESS_PERIODS:
            raise ValueError(f"Unknown stress period: {period_name}. "
                           f"Available: {list(cls.STRESS_PERIODS.keys())}")
        
        start, end = cls.STRESS_PERIODS[period_name]
        return cls(base_curve, historical_data, pricer_func, start, end)


def compute_stressed_var(
    base_curve: Curve,
    historical_data: pd.DataFrame,
    pricer_func: Callable[[Curve], float],
    stress_start: date,
    stress_end: date,
    confidence: float = 0.99
) -> float:
    """
    Convenience function to compute stressed VaR.
    
    Args:
        base_curve: Current curve
        historical_data: Historical data
        pricer_func: Portfolio pricer
        stress_start: Stress period start
        stress_end: Stress period end
        confidence: Confidence level
        
    Returns:
        Stressed VaR
    """
    stressed = StressedVaR(
        base_curve, historical_data, pricer_func,
        stress_start, stress_end
    )
    result = stressed.compute_stressed_var()
    
    if confidence == 0.95:
        return result.stressed_var_95
    else:
        return result.stressed_var_99


def identify_stress_periods(
    historical_data: pd.DataFrame,
    volatility_threshold: float = 2.0,
    min_period_days: int = 20
) -> List[Tuple[date, date]]:
    """
    Automatically identify stress periods from data.
    
    Looks for periods where rate volatility exceeds threshold
    times the average volatility.
    
    Args:
        historical_data: Historical rates
        volatility_threshold: Multiplier for identifying stress
        min_period_days: Minimum stress period length
        
    Returns:
        List of (start_date, end_date) tuples
    """
    df = historical_data.copy()
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        pivot = df.pivot_table(index='date', columns='tenor', values='rate')
    else:
        pivot = df
    
    # Compute rolling volatility
    changes = pivot.diff() * 10000
    rolling_vol = changes.abs().mean(axis=1).rolling(window=21).mean()
    
    # Average volatility
    avg_vol = rolling_vol.mean()
    
    # Identify high-vol periods
    high_vol = rolling_vol > (volatility_threshold * avg_vol)
    
    # Find contiguous periods
    stress_periods = []
    in_stress = False
    start_date = None
    
    for date_idx, is_high in high_vol.items():
        if is_high and not in_stress:
            in_stress = True
            start_date = date_idx
        elif not is_high and in_stress:
            in_stress = False
            if start_date is not None:
                period_days = (date_idx - start_date).days
                if period_days >= min_period_days:
                    stress_periods.append((start_date.date(), date_idx.date()))
    
    # Handle case where still in stress at end
    if in_stress and start_date is not None:
        end_date = pivot.index[-1]
        period_days = (end_date - start_date).days
        if period_days >= min_period_days:
            stress_periods.append((start_date.date(), end_date.date()))
    
    return stress_periods


__all__ = [
    "StressedVaR",
    "StressResult",
    "compute_stressed_var",
    "identify_stress_periods",
]
