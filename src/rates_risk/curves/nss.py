"""
Nelson-Siegel-Svensson (NSS) parametric yield curve model.

The NSS model provides a parsimonious representation of the yield curve
using 6 parameters (4 for original Nelson-Siegel):

    y(τ) = β₀ + β₁ * [(1-e^(-τ/λ₁))/(τ/λ₁)]
             + β₂ * [(1-e^(-τ/λ₁))/(τ/λ₁) - e^(-τ/λ₁)]
             + β₃ * [(1-e^(-τ/λ₂))/(τ/λ₂) - e^(-τ/λ₂)]

Parameters:
    β₀: Long-term level (asymptotic rate)
    β₁: Short-term component (slope)
    β₂: Medium-term hump (curvature 1)
    β₃: Second hump (curvature 2) - Svensson extension
    λ₁: Decay for first hump
    λ₂: Decay for second hump

This is commonly used for:
- Treasury par yield curves
- Sovereign bond curve fitting
- Central bank yield curve publications
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize, differential_evolution

from ..conventions import DayCount, year_fraction
from .curve import Curve


@dataclass
class NSSParameters:
    """Nelson-Siegel-Svensson parameters."""
    beta0: float  # Long-term level
    beta1: float  # Short-term component
    beta2: float  # Medium-term hump 1
    beta3: float  # Medium-term hump 2 (Svensson)
    lambda1: float  # Decay 1
    lambda2: float  # Decay 2
    
    @classmethod
    def default_initial_guess(cls) -> "NSSParameters":
        """Reasonable initial guess for USD curve."""
        return cls(
            beta0=0.04,   # ~4% long-term level
            beta1=-0.02,  # Upward sloping
            beta2=0.01,   # Slight hump
            beta3=0.01,   # Additional curvature
            lambda1=1.5,  # Medium-term decay
            lambda2=3.0   # Longer-term decay
        )
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for optimization."""
        return np.array([
            self.beta0, self.beta1, self.beta2, 
            self.beta3, self.lambda1, self.lambda2
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "NSSParameters":
        """Create from numpy array."""
        return cls(
            beta0=arr[0], beta1=arr[1], beta2=arr[2],
            beta3=arr[3], lambda1=arr[4], lambda2=arr[5]
        )


class NelsonSiegelSvensson:
    """
    Nelson-Siegel-Svensson yield curve model.
    
    Fits the NSS model to par yields or zero rates and provides
    methods to query the fitted curve.
    
    Attributes:
        anchor_date: Valuation date
        params: Fitted NSS parameters
        is_fitted: Whether model has been fitted
        day_count: Day count for yield calculations
    """
    
    def __init__(
        self,
        anchor_date: date,
        day_count: DayCount = DayCount.ACT_ACT
    ):
        self.anchor_date = anchor_date
        self.day_count = day_count
        self.params: Optional[NSSParameters] = None
        self.is_fitted = False
        self._fit_residuals: Optional[np.ndarray] = None
    
    @staticmethod
    def _nss_yield(tau: float, params: NSSParameters) -> float:
        """
        Compute NSS yield at maturity tau.
        
        Args:
            tau: Time to maturity in years
            params: NSS parameters
            
        Returns:
            Yield at maturity tau
        """
        if tau <= 0:
            # At t=0, return short rate (limit as tau -> 0)
            return params.beta0 + params.beta1
        
        # Factor loadings
        exp1 = np.exp(-tau / params.lambda1)
        exp2 = np.exp(-tau / params.lambda2)
        
        # Nelson-Siegel term
        ns_factor1 = (1 - exp1) / (tau / params.lambda1)
        ns_factor2 = ns_factor1 - exp1
        
        # Svensson extension
        sv_factor = (1 - exp2) / (tau / params.lambda2) - exp2
        
        y = (params.beta0 + 
             params.beta1 * ns_factor1 + 
             params.beta2 * ns_factor2 +
             params.beta3 * sv_factor)
        
        return y
    
    def yield_at(self, tau: Union[float, date]) -> float:
        """
        Get yield at maturity.
        
        Args:
            tau: Time to maturity in years, or maturity date
            
        Returns:
            Continuously compounded yield
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted - call fit() first")
        
        if isinstance(tau, date):
            tau = year_fraction(self.anchor_date, tau, self.day_count)
        
        return self._nss_yield(tau, self.params)
    
    def discount_factor(self, tau: Union[float, date]) -> float:
        """
        Get discount factor at maturity.
        
        Args:
            tau: Time to maturity in years, or maturity date
            
        Returns:
            Discount factor
        """
        if isinstance(tau, date):
            tau = year_fraction(self.anchor_date, tau, self.day_count)
        
        if tau <= 0:
            return 1.0
        
        y = self.yield_at(tau)
        return np.exp(-y * tau)
    
    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Get forward rate between t1 and t2.
        
        Args:
            t1: Start time (years)
            t2: End time (years)
            
        Returns:
            Simple forward rate
        """
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")
        
        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        
        return (df1 / df2 - 1) / (t2 - t1)
    
    def instantaneous_forward(self, tau: float) -> float:
        """
        Get instantaneous forward rate.
        
        The instantaneous forward rate is:
        f(τ) = y(τ) + τ * dy/dτ
        
        Args:
            tau: Time in years
            
        Returns:
            Instantaneous forward rate
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        if tau <= 0:
            return self.params.beta0 + self.params.beta1
        
        p = self.params
        exp1 = np.exp(-tau / p.lambda1)
        exp2 = np.exp(-tau / p.lambda2)
        
        # f(τ) = β₀ + β₁*exp(-τ/λ₁) + β₂*(τ/λ₁)*exp(-τ/λ₁) + β₃*(τ/λ₂)*exp(-τ/λ₂)
        f = (p.beta0 + 
             p.beta1 * exp1 + 
             p.beta2 * (tau / p.lambda1) * exp1 +
             p.beta3 * (tau / p.lambda2) * exp2)
        
        return f
    
    def fit(
        self,
        maturities: List[float],
        yields: List[float],
        weights: Optional[List[float]] = None,
        initial_guess: Optional[NSSParameters] = None,
        method: str = "L-BFGS-B"
    ) -> Tuple[float, np.ndarray]:
        """
        Fit NSS model to observed yields.
        
        Args:
            maturities: List of maturities in years
            yields: List of observed yields (decimal)
            weights: Optional weights for each observation
            initial_guess: Starting parameters
            method: Optimization method
            
        Returns:
            Tuple of (final_error, residuals)
        """
        if len(maturities) != len(yields):
            raise ValueError("Maturities and yields must have same length")
        
        if len(maturities) < 4:
            raise ValueError("Need at least 4 points to fit NSS model")
        
        tau = np.array(maturities, dtype=np.float64)
        y_obs = np.array(yields, dtype=np.float64)
        w = np.array(weights, dtype=np.float64) if weights else np.ones_like(tau)
        
        # Normalize weights
        w = w / w.sum()
        
        # Initial guess
        if initial_guess is None:
            initial_guess = NSSParameters.default_initial_guess()
            # Adjust level based on observed data
            initial_guess.beta0 = np.mean(y_obs)
        
        x0 = initial_guess.to_array()
        
        # Parameter bounds
        bounds = [
            (-0.5, 0.5),    # beta0: reasonable rate range
            (-0.5, 0.5),    # beta1
            (-0.5, 0.5),    # beta2
            (-0.5, 0.5),    # beta3
            (0.1, 10.0),    # lambda1: positive, reasonable range
            (0.1, 20.0),    # lambda2: positive, can be larger
        ]
        
        # Objective function
        def objective(params):
            p = NSSParameters.from_array(params)
            y_pred = np.array([self._nss_yield(t, p) for t in tau])
            residuals = y_pred - y_obs
            return np.sum(w * residuals**2)
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        # Try global optimization if local fails
        if not result.success or result.fun > 1e-6:
            try:
                result_de = differential_evolution(
                    objective,
                    bounds,
                    maxiter=500,
                    tol=1e-10,
                    seed=42
                )
                if result_de.fun < result.fun:
                    result = result_de
            except:
                pass
        
        # Store fitted parameters
        self.params = NSSParameters.from_array(result.x)
        self.is_fitted = True
        
        # Compute residuals
        y_pred = np.array([self._nss_yield(t, self.params) for t in tau])
        self._fit_residuals = y_pred - y_obs
        
        return result.fun, self._fit_residuals
    
    def fit_from_par_yields(
        self,
        tenors: List[str],
        par_yields: List[float],
        weights: Optional[List[float]] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Fit NSS model from par yield data.
        
        Args:
            tenors: List of tenor strings (e.g., ["3M", "1Y", "5Y"])
            par_yields: List of par yields
            weights: Optional weights
            
        Returns:
            Tuple of (final_error, residuals)
        """
        from ..dates import DateUtils
        
        maturities = [DateUtils.tenor_to_years(t) for t in tenors]
        return self.fit(maturities, par_yields, weights)
    
    def to_curve(
        self,
        tenors: List[float] = None,
        interpolation_method: str = "cubic_spline"
    ) -> Curve:
        """
        Convert fitted NSS model to a Curve object.
        
        Args:
            tenors: Tenors to sample (default: standard set)
            interpolation_method: Interpolation for output curve
            
        Returns:
            Curve object
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        if tenors is None:
            tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        
        curve = Curve(
            anchor_date=self.anchor_date,
            currency="USD",
            day_count=self.day_count,
            interpolation_method=interpolation_method
        )
        
        for tau in tenors:
            df = self.discount_factor(tau)
            curve.add_node(tau, df)
        
        curve.build()
        return curve
    
    def get_curve_data(
        self,
        min_tenor: float = 0.0,
        max_tenor: float = 30.0,
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get curve data for plotting.
        
        Returns:
            Tuple of (tenors, yields, forward_rates)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        tenors = np.linspace(max(0.01, min_tenor), max_tenor, num_points)
        yields = np.array([self.yield_at(t) for t in tenors])
        forwards = np.array([self.instantaneous_forward(t) for t in tenors])
        
        return tenors, yields, forwards
    
    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"NelsonSiegelSvensson(anchor={self.anchor_date}, fitted=False)"
        p = self.params
        return (f"NelsonSiegelSvensson(anchor={self.anchor_date}, "
                f"β₀={p.beta0:.4f}, β₁={p.beta1:.4f}, β₂={p.beta2:.4f}, "
                f"β₃={p.beta3:.4f}, λ₁={p.lambda1:.2f}, λ₂={p.lambda2:.2f})")


def fit_treasury_curve(
    anchor_date: date,
    tenors: List[str],
    par_yields: List[float],
    output_tenors: List[float] = None
) -> Curve:
    """
    Convenience function to fit Treasury curve using NSS.
    
    Args:
        anchor_date: Valuation date
        tenors: List of tenor strings
        par_yields: List of par yields (decimal)
        output_tenors: Tenors for output curve
        
    Returns:
        Fitted Curve object
    """
    nss = NelsonSiegelSvensson(anchor_date)
    nss.fit_from_par_yields(tenors, par_yields)
    return nss.to_curve(output_tenors)


__all__ = [
    "NelsonSiegelSvensson",
    "NSSParameters",
    "fit_treasury_curve",
]
