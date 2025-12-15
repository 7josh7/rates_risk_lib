"""
Interpolation methods for yield curves.

Provides:
- LinearInterpolator: Simple linear interpolation (debugging baseline)
- CubicSplineInterpolator: Cubic spline on zero rates or log discount factors
- LogLinearInterpolator: Log-linear interpolation on discount factors

All interpolators work with year fractions as x-coordinates and
either zero rates or log discount factors as y-coordinates.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np


class Interpolator(ABC):
    """Abstract base class for curve interpolation."""
    
    @abstractmethod
    def fit(self, times: np.ndarray, values: np.ndarray) -> None:
        """
        Fit the interpolator to data points.
        
        Args:
            times: Array of year fractions (must be sorted ascending)
            values: Array of values (zero rates or log discount factors)
        """
        pass
    
    @abstractmethod
    def interpolate(self, t: float) -> float:
        """
        Interpolate at a single point.
        
        Args:
            t: Year fraction
            
        Returns:
            Interpolated value
        """
        pass
    
    def __call__(self, t: float) -> float:
        """Convenience method to call interpolate."""
        return self.interpolate(t)
    
    @abstractmethod
    def derivative(self, t: float) -> float:
        """
        Return the first derivative at point t.
        
        Useful for computing instantaneous forward rates.
        """
        pass


class LinearInterpolator(Interpolator):
    """
    Linear interpolation.
    
    Simple linear interpolation between knot points.
    Extrapolates flat beyond boundaries.
    """
    
    def __init__(self):
        self.times: Optional[np.ndarray] = None
        self.values: Optional[np.ndarray] = None
    
    def fit(self, times: np.ndarray, values: np.ndarray) -> None:
        """Fit linear interpolator."""
        if len(times) != len(values):
            raise ValueError("Times and values must have same length")
        if len(times) < 2:
            raise ValueError("Need at least 2 points for interpolation")
        
        # Sort by time
        idx = np.argsort(times)
        self.times = np.array(times)[idx]
        self.values = np.array(values)[idx]
    
    def interpolate(self, t: float) -> float:
        """Linear interpolation with flat extrapolation."""
        if self.times is None:
            raise RuntimeError("Interpolator not fitted")
        
        if t <= self.times[0]:
            return float(self.values[0])
        if t >= self.times[-1]:
            return float(self.values[-1])
        
        # Find bracket
        idx = np.searchsorted(self.times, t, side='right') - 1
        idx = max(0, min(idx, len(self.times) - 2))
        
        t0, t1 = self.times[idx], self.times[idx + 1]
        v0, v1 = self.values[idx], self.values[idx + 1]
        
        # Linear interpolation
        w = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
        return float(v0 + w * (v1 - v0))
    
    def derivative(self, t: float) -> float:
        """Derivative of linear interpolation (piecewise constant)."""
        if self.times is None:
            raise RuntimeError("Interpolator not fitted")
        
        if t <= self.times[0] or t >= self.times[-1]:
            return 0.0
        
        idx = np.searchsorted(self.times, t, side='right') - 1
        idx = max(0, min(idx, len(self.times) - 2))
        
        t0, t1 = self.times[idx], self.times[idx + 1]
        v0, v1 = self.values[idx], self.values[idx + 1]
        
        return float((v1 - v0) / (t1 - t0)) if t1 != t0 else 0.0


class CubicSplineInterpolator(Interpolator):
    """
    Cubic spline interpolation.
    
    Uses natural cubic splines (second derivative = 0 at boundaries).
    Provides smooth first and second derivatives.
    """
    
    def __init__(self):
        self.times: Optional[np.ndarray] = None
        self.values: Optional[np.ndarray] = None
        self.coefficients: Optional[np.ndarray] = None  # Shape: (n-1, 4) for [a, b, c, d]
    
    def fit(self, times: np.ndarray, values: np.ndarray) -> None:
        """
        Fit natural cubic spline.
        
        Solves tridiagonal system for second derivatives,
        then computes polynomial coefficients for each interval.
        """
        if len(times) != len(values):
            raise ValueError("Times and values must have same length")
        
        n = len(times)
        if n < 2:
            raise ValueError("Need at least 2 points for interpolation")
        
        # Sort by time
        idx = np.argsort(times)
        self.times = np.array(times, dtype=np.float64)[idx]
        self.values = np.array(values, dtype=np.float64)[idx]
        
        if n == 2:
            # Degenerate to linear
            h = self.times[1] - self.times[0]
            slope = (self.values[1] - self.values[0]) / h if h > 0 else 0
            self.coefficients = np.array([[self.values[0], slope, 0, 0]])
            return
        
        # Compute spline coefficients using natural boundary conditions
        h = np.diff(self.times)
        
        # Build tridiagonal system for second derivatives
        # Natural spline: M[0] = M[n-1] = 0
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        A[0, 0] = 1.0
        A[n-1, n-1] = 1.0
        
        for i in range(1, n-1):
            A[i, i-1] = h[i-1]
            A[i, i] = 2 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            b[i] = 6 * ((self.values[i+1] - self.values[i]) / h[i] - 
                       (self.values[i] - self.values[i-1]) / h[i-1])
        
        # Solve for second derivatives M
        M = np.linalg.solve(A, b)
        
        # Compute polynomial coefficients for each interval
        # S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
        self.coefficients = np.zeros((n-1, 4))
        
        for i in range(n-1):
            self.coefficients[i, 0] = self.values[i]  # a
            self.coefficients[i, 1] = (self.values[i+1] - self.values[i]) / h[i] - h[i] * (M[i+1] + 2*M[i]) / 6  # b
            self.coefficients[i, 2] = M[i] / 2  # c
            self.coefficients[i, 3] = (M[i+1] - M[i]) / (6 * h[i])  # d
    
    def interpolate(self, t: float) -> float:
        """Evaluate cubic spline at point t."""
        if self.times is None or self.coefficients is None:
            raise RuntimeError("Interpolator not fitted")
        
        # Flat extrapolation
        if t <= self.times[0]:
            return float(self.values[0])
        if t >= self.times[-1]:
            return float(self.values[-1])
        
        # Find interval
        idx = np.searchsorted(self.times, t, side='right') - 1
        idx = max(0, min(idx, len(self.coefficients) - 1))
        
        dx = t - self.times[idx]
        a, b, c, d = self.coefficients[idx]
        
        return float(a + b*dx + c*dx**2 + d*dx**3)
    
    def derivative(self, t: float) -> float:
        """First derivative of cubic spline at point t."""
        if self.times is None or self.coefficients is None:
            raise RuntimeError("Interpolator not fitted")
        
        if t <= self.times[0] or t >= self.times[-1]:
            return 0.0
        
        idx = np.searchsorted(self.times, t, side='right') - 1
        idx = max(0, min(idx, len(self.coefficients) - 1))
        
        dx = t - self.times[idx]
        _, b, c, d = self.coefficients[idx]
        
        return float(b + 2*c*dx + 3*d*dx**2)
    
    def second_derivative(self, t: float) -> float:
        """Second derivative of cubic spline at point t."""
        if self.times is None or self.coefficients is None:
            raise RuntimeError("Interpolator not fitted")
        
        if t <= self.times[0] or t >= self.times[-1]:
            return 0.0
        
        idx = np.searchsorted(self.times, t, side='right') - 1
        idx = max(0, min(idx, len(self.coefficients) - 1))
        
        dx = t - self.times[idx]
        _, _, c, d = self.coefficients[idx]
        
        return float(2*c + 6*d*dx)


class LogLinearInterpolator(Interpolator):
    """
    Log-linear interpolation on discount factors.
    
    Interpolates linearly in log(discount factor) space,
    which corresponds to piecewise constant forward rates.
    """
    
    def __init__(self):
        self.times: Optional[np.ndarray] = None
        self.log_df: Optional[np.ndarray] = None
    
    def fit(self, times: np.ndarray, discount_factors: np.ndarray) -> None:
        """
        Fit log-linear interpolator.
        
        Args:
            times: Year fractions
            discount_factors: Discount factors (not log!)
        """
        if len(times) != len(discount_factors):
            raise ValueError("Times and discount factors must have same length")
        if len(times) < 2:
            raise ValueError("Need at least 2 points for interpolation")
        
        if np.any(discount_factors <= 0):
            raise ValueError("Discount factors must be positive")
        
        idx = np.argsort(times)
        self.times = np.array(times, dtype=np.float64)[idx]
        self.log_df = np.log(np.array(discount_factors, dtype=np.float64)[idx])
    
    def interpolate(self, t: float) -> float:
        """
        Interpolate log discount factor.
        
        Returns the log of the discount factor at time t.
        """
        if self.times is None or self.log_df is None:
            raise RuntimeError("Interpolator not fitted")
        
        if t <= self.times[0]:
            return float(self.log_df[0])
        if t >= self.times[-1]:
            # Linear extrapolation in log space
            slope = (self.log_df[-1] - self.log_df[-2]) / (self.times[-1] - self.times[-2])
            return float(self.log_df[-1] + slope * (t - self.times[-1]))
        
        idx = np.searchsorted(self.times, t, side='right') - 1
        idx = max(0, min(idx, len(self.times) - 2))
        
        t0, t1 = self.times[idx], self.times[idx + 1]
        v0, v1 = self.log_df[idx], self.log_df[idx + 1]
        
        w = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
        return float(v0 + w * (v1 - v0))
    
    def get_discount_factor(self, t: float) -> float:
        """Get discount factor at time t."""
        return np.exp(self.interpolate(t))
    
    def derivative(self, t: float) -> float:
        """Derivative of log discount factor (negative of instantaneous forward rate)."""
        if self.times is None or self.log_df is None:
            raise RuntimeError("Interpolator not fitted")
        
        if t <= self.times[0] or t >= self.times[-1]:
            if t >= self.times[-1]:
                return float((self.log_df[-1] - self.log_df[-2]) / 
                           (self.times[-1] - self.times[-2]))
            return 0.0
        
        idx = np.searchsorted(self.times, t, side='right') - 1
        idx = max(0, min(idx, len(self.times) - 2))
        
        t0, t1 = self.times[idx], self.times[idx + 1]
        v0, v1 = self.log_df[idx], self.log_df[idx + 1]
        
        return float((v1 - v0) / (t1 - t0)) if t1 != t0 else 0.0


def create_interpolator(method: str) -> Interpolator:
    """
    Factory function to create an interpolator by name.
    
    Args:
        method: One of "linear", "cubic_spline", "log_linear"
        
    Returns:
        Interpolator instance
    """
    method = method.lower().replace("-", "_").replace(" ", "_")
    
    if method in ("linear", "lin"):
        return LinearInterpolator()
    elif method in ("cubic_spline", "cubic", "spline"):
        return CubicSplineInterpolator()
    elif method in ("log_linear", "loglinear"):
        return LogLinearInterpolator()
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


__all__ = [
    "Interpolator",
    "LinearInterpolator",
    "CubicSplineInterpolator",
    "LogLinearInterpolator",
    "create_interpolator",
]
