"""
Historical Simulation VaR and Expected Shortfall.

Implements non-parametric VaR using historical rate changes:
1. Load historical rate changes from a lookback window
2. Apply each historical move to today's curve
3. Reprice portfolio under each shocked curve
4. Compute VaR as empirical quantile of P&L distribution
5. Compute ES as average of tail losses

Advantages:
- Captures actual historical distribution (fat tails, skew)
- No distributional assumptions
- Naturally incorporates correlations

Limitations:
- Limited by historical window
- Assumes past is representative of future
- Computationally intensive for large portfolios
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..curves.curve import Curve
from ..dates import DateUtils


@dataclass
class HistoricalVaRResult:
    """
    Result from historical simulation VaR.
    
    Attributes:
        var_95: 95% VaR (loss at 5th percentile)
        var_99: 99% VaR (loss at 1st percentile)
        es_95: 95% Expected Shortfall
        es_99: 99% Expected Shortfall
        num_scenarios: Number of historical scenarios used
        worst_loss: Maximum loss
        best_gain: Maximum gain
        mean_pnl: Average P&L
        pnl_distribution: Full P&L distribution
    """
    var_95: float
    var_99: float
    es_95: float
    es_99: float
    num_scenarios: int
    worst_loss: float
    best_gain: float
    mean_pnl: float
    pnl_distribution: np.ndarray = field(repr=False)
    scenario_dates: Optional[List[date]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting."""
        return {
            "var_95": self.var_95,
            "var_99": self.var_99,
            "es_95": self.es_95,
            "es_99": self.es_99,
            "num_scenarios": self.num_scenarios,
            "worst_loss": self.worst_loss,
            "best_gain": self.best_gain,
            "mean_pnl": self.mean_pnl
        }


class HistoricalSimulation:
    """
    Historical Simulation VaR engine.
    
    Uses historical rate changes to generate scenarios and
    computes VaR/ES from the resulting P&L distribution.
    """
    
    # Standard tenors for risk factors
    RISK_FACTOR_TENORS = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    
    def __init__(
        self,
        base_curve: Curve,
        historical_data: pd.DataFrame,
        pricer_func: Callable[[Curve], float],
        tenors: Optional[List[str]] = None
    ):
        """
        Initialize historical simulation.
        
        Args:
            base_curve: Current yield curve
            historical_data: DataFrame with columns [date, tenor, rate]
            pricer_func: Function that takes curve and returns portfolio PV
            tenors: Risk factor tenors (default: standard set)
        """
        self.base_curve = base_curve
        self.historical_data = historical_data
        self.pricer_func = pricer_func
        self.tenors = tenors or self.RISK_FACTOR_TENORS
        
        # Process historical data
        self._process_historical_data()
    
    def _process_historical_data(self) -> None:
        """Convert historical data to rate changes."""
        # Pivot to get rates by date and tenor
        df = self.historical_data.copy()
        # Accept wide format (date + tenor columns) by melting to long
        if "rate" not in df.columns and "date" in df.columns:
            value_cols = [c for c in df.columns if c.lower() != "date"]
            df = df.melt(id_vars="date", value_vars=value_cols, var_name="tenor", value_name="rate")
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            pivot = df.pivot_table(index='date', columns='tenor', values='rate')
        else:
            pivot = df
        
        # Filter to tenors we care about
        available_tenors = [t for t in self.tenors if t in pivot.columns]
        pivot = pivot[available_tenors].dropna()
        
        # Compute 1-day changes in basis points
        self.rate_changes = pivot.diff() * 10000  # Convert to bp
        self.rate_changes = self.rate_changes.dropna()
        
        self.available_tenors = available_tenors
        self.scenario_dates = self.rate_changes.index.tolist()
    
    def _create_shocked_curve(self, rate_changes: pd.Series) -> Curve:
        """
        Create a curve shocked by historical rate changes.
        
        Args:
            rate_changes: Series of rate changes by tenor (in bp)
            
        Returns:
            Shocked curve
        """
        # Build bump profile
        bump_profile = {}
        for tenor in self.available_tenors:
            if tenor in rate_changes.index:
                bump_profile[tenor] = rate_changes[tenor]
        
        # Apply bumps
        from ..risk.bumping import BumpEngine
        engine = BumpEngine(self.base_curve)
        return engine.custom_bump(bump_profile)
    
    def run_simulation(
        self,
        lookback_days: Optional[int] = None
    ) -> HistoricalVaRResult:
        """
        Run historical simulation.
        
        Args:
            lookback_days: Number of days to use (default: all available)
            
        Returns:
            HistoricalVaRResult
        """
        # Limit lookback if specified
        changes = self.rate_changes
        if lookback_days and len(changes) > lookback_days:
            changes = changes.tail(lookback_days)
        
        # Base PV
        base_pv = self.pricer_func(self.base_curve)
        
        # Run scenarios
        pnl_distribution = []
        scenario_dates = []
        
        for idx, row in changes.iterrows():
            try:
                shocked_curve = self._create_shocked_curve(row)
                shocked_pv = self.pricer_func(shocked_curve)
                pnl = shocked_pv - base_pv
                pnl_distribution.append(pnl)
                scenario_dates.append(idx)
            except Exception as e:
                # Skip problematic scenarios
                continue
        
        pnl_array = np.array(pnl_distribution)
        if len(pnl_array) == 0:
            raise ValueError("No historical scenarios available after filtering; check data and lookback window.")
        
        # Compute statistics
        # VaR is the loss at the specified percentile (negative P&L)
        # Convention: VaR is reported as positive number (loss)
        var_95 = -np.percentile(pnl_array, 5)  # 5th percentile
        var_99 = -np.percentile(pnl_array, 1)  # 1st percentile
        
        # ES is the average of losses beyond VaR
        losses_beyond_95 = pnl_array[pnl_array <= -var_95]
        losses_beyond_99 = pnl_array[pnl_array <= -var_99]
        
        es_95 = -np.mean(losses_beyond_95) if len(losses_beyond_95) > 0 else var_95
        es_99 = -np.mean(losses_beyond_99) if len(losses_beyond_99) > 0 else var_99
        
        return HistoricalVaRResult(
            var_95=var_95,
            var_99=var_99,
            es_95=es_95,
            es_99=es_99,
            num_scenarios=len(pnl_array),
            worst_loss=-np.min(pnl_array),
            best_gain=np.max(pnl_array),
            mean_pnl=np.mean(pnl_array),
            pnl_distribution=pnl_array,
            scenario_dates=scenario_dates
        )
    
    def run_parametric_var(
        self,
        lookback_days: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Quick parametric VaR using sensitivity approximation.
        
        Uses DV01 × historical rate changes instead of full repricing.
        Much faster but less accurate for non-linear portfolios.
        
        Args:
            lookback_days: Lookback period
            
        Returns:
            Tuple of (VaR_95, VaR_99)
        """
        from ..risk.bumping import BumpEngine
        
        # Compute parallel DV01
        engine = BumpEngine(self.base_curve)
        dv01 = engine.compute_dv01(self.pricer_func, 1.0)
        
        # Use parallel rate changes (average across tenors)
        changes = self.rate_changes
        if lookback_days and len(changes) > lookback_days:
            changes = changes.tail(lookback_days)
        
        parallel_changes = changes.mean(axis=1)
        
        # Approximate P&L
        pnl_approx = -dv01 * parallel_changes
        
        var_95 = -np.percentile(pnl_approx, 5)
        var_99 = -np.percentile(pnl_approx, 1)
        
        return var_95, var_99


def compute_historical_var(
    base_curve: Curve,
    historical_data: pd.DataFrame,
    pricer_func: Callable[[Curve], float],
    confidence: float = 0.95,
    lookback_days: Optional[int] = None
) -> float:
    """
    Convenience function to compute historical VaR.
    
    Args:
        base_curve: Current curve
        historical_data: Historical rates DataFrame
        pricer_func: Portfolio pricer
        confidence: Confidence level (default 95%)
        lookback_days: Lookback period
        
    Returns:
        VaR (as positive number representing potential loss)
    """
    hs = HistoricalSimulation(base_curve, historical_data, pricer_func)
    result = hs.run_simulation(lookback_days)
    
    if confidence == 0.95:
        return result.var_95
    elif confidence == 0.99:
        return result.var_99
    else:
        percentile = (1 - confidence) * 100
        return -np.percentile(result.pnl_distribution, percentile)


def compute_historical_es(
    base_curve: Curve,
    historical_data: pd.DataFrame,
    pricer_func: Callable[[Curve], float],
    confidence: float = 0.95,
    lookback_days: Optional[int] = None
) -> float:
    """
    Convenience function to compute historical Expected Shortfall.
    
    Args:
        base_curve: Current curve
        historical_data: Historical rates DataFrame
        pricer_func: Portfolio pricer
        confidence: Confidence level
        lookback_days: Lookback period
        
    Returns:
        ES (as positive number)
    """
    hs = HistoricalSimulation(base_curve, historical_data, pricer_func)
    result = hs.run_simulation(lookback_days)
    
    if confidence == 0.95:
        return result.es_95
    elif confidence == 0.99:
        return result.es_99
    else:
        var = -np.percentile(result.pnl_distribution, (1 - confidence) * 100)
        losses_beyond = result.pnl_distribution[result.pnl_distribution <= -var]
        return -np.mean(losses_beyond) if len(losses_beyond) > 0 else var


def generate_historical_scenarios_from_csv(
    filepath: str,
    date_column: str = "date",
    tenor_column: str = "tenor",
    rate_column: str = "rate"
) -> pd.DataFrame:
    """
    Load historical data from CSV file.
    
    Expected format:
        date, tenor, rate
        2022-01-03, 2Y, 0.0075
        2022-01-03, 5Y, 0.0137
        ...
    
    Args:
        filepath: Path to CSV file
        date_column: Name of date column
        tenor_column: Name of tenor column
        rate_column: Name of rate column
        
    Returns:
        DataFrame ready for HistoricalSimulation
    """
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        date_column: "date",
        tenor_column: "tenor",
        rate_column: "rate"
    })
    df["date"] = pd.to_datetime(df["date"])
    return df


__all__ = [
    "HistoricalSimulation",
    "HistoricalVaRResult",
    "compute_historical_var",
    "compute_historical_es",
    "generate_historical_scenarios_from_csv",
]
