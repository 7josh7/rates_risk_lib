"""
Monte Carlo VaR and Expected Shortfall.

Implements parametric VaR using Monte Carlo simulation:
1. Estimate covariance matrix of rate changes from history
2. Simulate correlated rate changes from multivariate normal
3. Apply simulated shocks to today's curve
4. Reprice portfolio under each simulated scenario
5. Compute VaR/ES from simulated P&L distribution

Assumptions:
- Rate changes follow multivariate normal distribution
- Covariance is stable over time
- Linear and non-linear effects captured through repricing

Limitations:
- Normality assumption may underestimate tail risk
- Covariance estimation sensitive to lookback window
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..curves.curve import Curve
from ..dates import DateUtils


@dataclass
class MonteCarloResult:
    """
    Result from Monte Carlo VaR simulation.
    
    Attributes:
        var_95: 95% VaR
        var_99: 99% VaR
        es_95: 95% Expected Shortfall
        es_99: 99% Expected Shortfall
        num_paths: Number of simulation paths
        pnl_distribution: Simulated P&L distribution
        mean_pnl: Average simulated P&L
        std_pnl: Standard deviation of P&L
    """
    var_95: float
    var_99: float
    es_95: float
    es_99: float
    num_paths: int
    mean_pnl: float
    std_pnl: float
    pnl_distribution: np.ndarray = field(repr=False)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "var_95": self.var_95,
            "var_99": self.var_99,
            "es_95": self.es_95,
            "es_99": self.es_99,
            "num_paths": self.num_paths,
            "mean_pnl": self.mean_pnl,
            "std_pnl": self.std_pnl
        }


class MonteCarloVaR:
    """
    Monte Carlo VaR engine.
    
    Simulates correlated rate changes using multivariate normal
    distribution calibrated to historical data.
    """
    
    # Standard tenors
    RISK_FACTOR_TENORS = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    
    def __init__(
        self,
        base_curve: Curve,
        historical_data: pd.DataFrame,
        pricer_func: Callable[[Curve], float],
        tenors: Optional[List[str]] = None
    ):
        """
        Initialize Monte Carlo engine.
        
        Args:
            base_curve: Current yield curve
            historical_data: DataFrame with historical rates
            pricer_func: Function that prices portfolio given curve
            tenors: Risk factor tenors
        """
        self.base_curve = base_curve
        self.historical_data = historical_data
        self.pricer_func = pricer_func
        self.tenors = tenors or self.RISK_FACTOR_TENORS
        
        # Calibrate model
        self._calibrate()
    
    def _calibrate(self) -> None:
        """Calibrate covariance matrix from historical data."""
        df = self.historical_data.copy()
        if "rate" not in df.columns and "date" in df.columns:
            value_cols = [c for c in df.columns if c.lower() != "date"]
            df = df.melt(id_vars="date", value_vars=value_cols, var_name="tenor", value_name="rate")
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            pivot = df.pivot_table(index='date', columns='tenor', values='rate')
        else:
            pivot = df
        
        # Filter tenors
        self.available_tenors = [t for t in self.tenors if t in pivot.columns]
        pivot = pivot[self.available_tenors].dropna()
        
        # Compute daily rate changes in basis points
        changes = pivot.diff() * 10000
        changes = changes.dropna()
        
        # Estimate mean and covariance
        self.mean_changes = changes.mean().values  # Should be ~0
        self.cov_matrix = changes.cov().values
        
        # Ensure covariance is positive semi-definite
        self._ensure_psd()
        
        self.num_factors = len(self.available_tenors)
    
    def _ensure_psd(self) -> None:
        """Ensure covariance matrix is positive semi-definite."""
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)
        
        # Clip negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        # Reconstruct
        self.cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Compute Cholesky for simulation
        self.cholesky = np.linalg.cholesky(self.cov_matrix)
    
    def simulate_scenarios(
        self,
        num_paths: int = 10000,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate simulated rate change scenarios.
        
        Args:
            num_paths: Number of simulation paths
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (num_paths, num_factors) with rate changes in bp
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate standard normal samples
        z = np.random.standard_normal((num_paths, self.num_factors))
        
        # Transform using Cholesky decomposition
        # X = L @ Z gives correlated samples
        scenarios = z @ self.cholesky.T
        
        # Add mean (should be ~0)
        scenarios += self.mean_changes
        
        return scenarios
    
    def _create_shocked_curve(self, rate_changes: np.ndarray) -> Curve:
        """Create curve with simulated rate changes applied."""
        bump_profile = {
            tenor: rate_changes[i]
            for i, tenor in enumerate(self.available_tenors)
        }
        
        from ..risk.bumping import BumpEngine
        engine = BumpEngine(self.base_curve)
        return engine.custom_bump(bump_profile)
    
    def run_simulation(
        self,
        num_paths: int = 10000,
        seed: Optional[int] = 42
    ) -> MonteCarloResult:
        """
        Run Monte Carlo VaR simulation.
        
        Args:
            num_paths: Number of simulation paths
            seed: Random seed
            
        Returns:
            MonteCarloResult
        """
        # Generate scenarios
        scenarios = self.simulate_scenarios(num_paths, seed)
        
        # Base PV
        base_pv = self.pricer_func(self.base_curve)
        
        # Price under each scenario
        pnl_distribution = []
        
        for i in range(num_paths):
            try:
                shocked_curve = self._create_shocked_curve(scenarios[i])
                shocked_pv = self.pricer_func(shocked_curve)
                pnl = shocked_pv - base_pv
                pnl_distribution.append(pnl)
            except:
                # Skip problematic scenarios
                continue
        
        pnl_array = np.array(pnl_distribution)
        
        # Compute statistics
        var_95 = -np.percentile(pnl_array, 5)
        var_99 = -np.percentile(pnl_array, 1)
        
        # ES
        losses_95 = pnl_array[pnl_array <= -var_95]
        losses_99 = pnl_array[pnl_array <= -var_99]
        
        es_95 = -np.mean(losses_95) if len(losses_95) > 0 else var_95
        es_99 = -np.mean(losses_99) if len(losses_99) > 0 else var_99
        
        return MonteCarloResult(
            var_95=var_95,
            var_99=var_99,
            es_95=es_95,
            es_99=es_99,
            num_paths=len(pnl_array),
            mean_pnl=np.mean(pnl_array),
            std_pnl=np.std(pnl_array),
            pnl_distribution=pnl_array
        )
    
    def run_delta_normal_var(self) -> Tuple[float, float]:
        """
        Quick delta-normal VaR (parametric closed-form).
        
        Uses linear approximation: P&L ≈ -KRD' @ ΔR
        Var[P&L] = KRD' @ Σ @ KRD
        
        Returns:
            Tuple of (VaR_95, VaR_99)
        """
        from ..risk.keyrate import KeyRateEngine
        
        # Get key-rate DV01 vector
        kr_engine = KeyRateEngine(self.base_curve, self.available_tenors)
        kr_dv01 = kr_engine.compute_key_rate_dv01(self.pricer_func)
        
        # DV01 vector (convert to proper array)
        dv01_vector = np.array([kr_dv01.dv01s.get(t, 0) for t in self.available_tenors])
        
        # Portfolio variance = DV01' @ Cov @ DV01
        portfolio_variance = dv01_vector @ self.cov_matrix @ dv01_vector
        portfolio_std = np.sqrt(portfolio_variance)
        
        # VaR using normal quantiles
        z_95 = 1.645
        z_99 = 2.326
        
        var_95 = z_95 * portfolio_std
        var_99 = z_99 * portfolio_std
        
        return var_95, var_99


def compute_mc_var(
    base_curve: Curve,
    historical_data: pd.DataFrame,
    pricer_func: Callable[[Curve], float],
    confidence: float = 0.95,
    num_paths: int = 10000,
    seed: int = 42
) -> float:
    """
    Convenience function to compute Monte Carlo VaR.
    
    Args:
        base_curve: Current curve
        historical_data: Historical rates for calibration
        pricer_func: Portfolio pricer
        confidence: Confidence level
        num_paths: Number of simulation paths
        seed: Random seed
        
    Returns:
        VaR (positive number)
    """
    mc = MonteCarloVaR(base_curve, historical_data, pricer_func)
    result = mc.run_simulation(num_paths, seed)
    
    if confidence == 0.95:
        return result.var_95
    elif confidence == 0.99:
        return result.var_99
    else:
        percentile = (1 - confidence) * 100
        return -np.percentile(result.pnl_distribution, percentile)


def compute_delta_normal_var(
    base_curve: Curve,
    dv01: float,
    historical_data: pd.DataFrame,
    confidence: float = 0.95
) -> float:
    """
    Quick delta-normal VaR using parallel DV01.
    
    Args:
        base_curve: Current curve
        dv01: Portfolio DV01
        historical_data: Historical rates
        confidence: Confidence level
        
    Returns:
        VaR (positive number)
    """
    # Process historical data
    df = historical_data.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        pivot = df.pivot_table(index='date', columns='tenor', values='rate')
    else:
        pivot = df
    
    # Compute average rate changes across tenors (parallel)
    changes = pivot.diff() * 10000
    parallel_changes = changes.mean(axis=1).dropna()
    
    # Volatility of parallel rate changes
    vol_bp = parallel_changes.std()
    
    # Portfolio P&L volatility
    pnl_vol = abs(dv01) * vol_bp
    
    # VaR
    if confidence == 0.95:
        z = 1.645
    elif confidence == 0.99:
        z = 2.326
    else:
        from scipy.stats import norm
        z = norm.ppf(confidence)
    
    return z * pnl_vol


__all__ = [
    "MonteCarloVaR",
    "MonteCarloResult",
    "compute_mc_var",
    "compute_delta_normal_var",
]
