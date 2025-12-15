"""
Base option pricing models.

Implements:
- Bachelier (normal) model for rates options
- Black'76 model for forward/futures options
- Shifted Black for negative rates

These are the "base models" that take implied vol as input.
SABR provides the implied vol, then these models price.
"""

from typing import Dict, Tuple
import numpy as np
from scipy.stats import norm


# Standard normal CDF and PDF
N = norm.cdf
n = norm.pdf


def bachelier_call(
    F: float,
    K: float,
    T: float,
    sigma_n: float,
    df: float = 1.0
) -> float:
    """
    Bachelier (normal) model call option price.
    
    Assumes forward follows arithmetic Brownian motion:
    dF = sigma_n * dW
    
    Args:
        F: Forward rate
        K: Strike
        T: Time to expiry (years)
        sigma_n: Normal volatility
        df: Discount factor to payment
        
    Returns:
        Call option price
    """
    if T <= 0:
        return max(F - K, 0) * df
    
    if sigma_n <= 0:
        return max(F - K, 0) * df
    
    sqrt_t = np.sqrt(T)
    d = (F - K) / (sigma_n * sqrt_t)
    
    price = df * ((F - K) * N(d) + sigma_n * sqrt_t * n(d))
    
    return price


def bachelier_put(
    F: float,
    K: float,
    T: float,
    sigma_n: float,
    df: float = 1.0
) -> float:
    """
    Bachelier (normal) model put option price.
    
    Args:
        F: Forward rate
        K: Strike
        T: Time to expiry
        sigma_n: Normal volatility
        df: Discount factor
        
    Returns:
        Put option price
    """
    if T <= 0:
        return max(K - F, 0) * df
    
    if sigma_n <= 0:
        return max(K - F, 0) * df
    
    sqrt_t = np.sqrt(T)
    d = (F - K) / (sigma_n * sqrt_t)
    
    price = df * ((K - F) * N(-d) + sigma_n * sqrt_t * n(d))
    
    return price


def black76_call(
    F: float,
    K: float,
    T: float,
    sigma_b: float,
    df: float = 1.0
) -> float:
    """
    Black'76 model call option price.
    
    Assumes forward follows geometric Brownian motion:
    dF = sigma_b * F * dW
    
    Args:
        F: Forward rate
        K: Strike
        T: Time to expiry
        sigma_b: Black (lognormal) volatility
        df: Discount factor
        
    Returns:
        Call option price
    """
    if T <= 0:
        return max(F - K, 0) * df
    
    if F <= 0 or K <= 0:
        raise ValueError("Forward and strike must be positive for Black model")
    
    if sigma_b <= 0:
        return max(F - K, 0) * df
    
    sqrt_t = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma_b**2 * T) / (sigma_b * sqrt_t)
    d2 = d1 - sigma_b * sqrt_t
    
    price = df * (F * N(d1) - K * N(d2))
    
    return price


def black76_put(
    F: float,
    K: float,
    T: float,
    sigma_b: float,
    df: float = 1.0
) -> float:
    """
    Black'76 model put option price.
    
    Args:
        F: Forward rate
        K: Strike
        T: Time to expiry
        sigma_b: Black volatility
        df: Discount factor
        
    Returns:
        Put option price
    """
    if T <= 0:
        return max(K - F, 0) * df
    
    if F <= 0 or K <= 0:
        raise ValueError("Forward and strike must be positive for Black model")
    
    if sigma_b <= 0:
        return max(K - F, 0) * df
    
    sqrt_t = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma_b**2 * T) / (sigma_b * sqrt_t)
    d2 = d1 - sigma_b * sqrt_t
    
    price = df * (K * N(-d2) - F * N(-d1))
    
    return price


def shifted_black_call(
    F: float,
    K: float,
    T: float,
    sigma_b: float,
    shift: float,
    df: float = 1.0
) -> float:
    """
    Shifted Black'76 model call option price.
    
    Allows pricing when forward can be negative:
    d(F + shift) = sigma_b * (F + shift) * dW
    
    Args:
        F: Forward rate (can be negative)
        K: Strike (can be negative)
        T: Time to expiry
        sigma_b: Black volatility
        shift: Shift parameter (positive)
        df: Discount factor
        
    Returns:
        Call option price
    """
    F_shifted = F + shift
    K_shifted = K + shift
    
    if F_shifted <= 0 or K_shifted <= 0:
        raise ValueError(f"Shifted forward ({F_shifted}) and strike ({K_shifted}) must be positive")
    
    return black76_call(F_shifted, K_shifted, T, sigma_b, df)


def shifted_black_put(
    F: float,
    K: float,
    T: float,
    sigma_b: float,
    shift: float,
    df: float = 1.0
) -> float:
    """
    Shifted Black'76 model put option price.
    
    Args:
        F: Forward rate
        K: Strike
        T: Time to expiry
        sigma_b: Black volatility
        shift: Shift parameter
        df: Discount factor
        
    Returns:
        Put option price
    """
    F_shifted = F + shift
    K_shifted = K + shift
    
    if F_shifted <= 0 or K_shifted <= 0:
        raise ValueError(f"Shifted forward ({F_shifted}) and strike ({K_shifted}) must be positive")
    
    return black76_put(F_shifted, K_shifted, T, sigma_b, df)


def bachelier_greeks(
    F: float,
    K: float,
    T: float,
    sigma_n: float,
    df: float = 1.0,
    is_call: bool = True
) -> Dict[str, float]:
    """
    Compute Greeks for Bachelier model.
    
    Args:
        F: Forward rate
        K: Strike
        T: Time to expiry
        sigma_n: Normal volatility
        df: Discount factor
        is_call: True for call, False for put
        
    Returns:
        Dict with delta, gamma, vega, theta
    """
    if T <= 0 or sigma_n <= 0:
        return {
            'delta': df if (F > K and is_call) or (F < K and not is_call) else 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0
        }
    
    sqrt_t = np.sqrt(T)
    d = (F - K) / (sigma_n * sqrt_t)
    
    # Delta
    if is_call:
        delta = df * N(d)
    else:
        delta = -df * N(-d)
    
    # Gamma (same for call and put)
    gamma = df * n(d) / (sigma_n * sqrt_t)
    
    # Vega (sensitivity to normal vol)
    vega = df * sqrt_t * n(d)
    
    # Theta (time decay)
    theta = -df * sigma_n * n(d) / (2 * sqrt_t)
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta
    }


def black76_greeks(
    F: float,
    K: float,
    T: float,
    sigma_b: float,
    df: float = 1.0,
    is_call: bool = True
) -> Dict[str, float]:
    """
    Compute Greeks for Black'76 model.
    
    Args:
        F: Forward rate
        K: Strike
        T: Time to expiry
        sigma_b: Black volatility
        df: Discount factor
        is_call: True for call, False for put
        
    Returns:
        Dict with delta, gamma, vega, theta
    """
    if T <= 0 or sigma_b <= 0 or F <= 0 or K <= 0:
        return {
            'delta': df if (F > K and is_call) or (F < K and not is_call) else 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0
        }
    
    sqrt_t = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma_b**2 * T) / (sigma_b * sqrt_t)
    d2 = d1 - sigma_b * sqrt_t
    
    # Delta
    if is_call:
        delta = df * N(d1)
    else:
        delta = -df * N(-d1)
    
    # Gamma (same for call and put)
    gamma = df * n(d1) / (F * sigma_b * sqrt_t)
    
    # Vega (sensitivity to Black vol)
    vega = df * F * sqrt_t * n(d1)
    
    # Theta (time decay)
    theta = -df * F * sigma_b * n(d1) / (2 * sqrt_t)
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta
    }


def implied_vol_bachelier(
    price: float,
    F: float,
    K: float,
    T: float,
    df: float = 1.0,
    is_call: bool = True,
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    """
    Compute implied normal volatility from option price.
    
    Uses Newton-Raphson iteration.
    
    Args:
        price: Option price
        F: Forward rate
        K: Strike
        T: Time to expiry
        df: Discount factor
        is_call: True for call, False for put
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Implied normal volatility
    """
    if T <= 0:
        raise ValueError("Cannot compute implied vol for expired option")
    
    # Initial guess
    intrinsic = max(F - K, 0) if is_call else max(K - F, 0)
    if price <= intrinsic * df:
        return 0.0
    
    sigma = abs(F - K) / np.sqrt(T) if abs(F - K) > 0 else 0.01
    
    pricer = bachelier_call if is_call else bachelier_put
    
    for _ in range(max_iter):
        p = pricer(F, K, T, sigma, df)
        vega = bachelier_greeks(F, K, T, sigma, df, is_call)['vega']
        
        if abs(vega) < 1e-15:
            break
        
        diff = p - price
        if abs(diff) < tol:
            break
        
        sigma = sigma - diff / vega
        sigma = max(sigma, 1e-10)
    
    return sigma


def implied_vol_black(
    price: float,
    F: float,
    K: float,
    T: float,
    df: float = 1.0,
    is_call: bool = True,
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    """
    Compute implied Black volatility from option price.
    
    Uses Newton-Raphson iteration.
    
    Args:
        price: Option price
        F: Forward rate
        K: Strike
        T: Time to expiry
        df: Discount factor
        is_call: True for call, False for put
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Implied Black volatility
    """
    if T <= 0:
        raise ValueError("Cannot compute implied vol for expired option")
    
    if F <= 0 or K <= 0:
        raise ValueError("Forward and strike must be positive")
    
    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2 * np.pi / T) * price / (df * F)
    sigma = max(sigma, 0.01)
    
    pricer = black76_call if is_call else black76_put
    
    for _ in range(max_iter):
        p = pricer(F, K, T, sigma, df)
        vega = black76_greeks(F, K, T, sigma, df, is_call)['vega']
        
        if abs(vega) < 1e-15:
            break
        
        diff = p - price
        if abs(diff) < tol:
            break
        
        sigma = sigma - diff / vega
        sigma = max(sigma, 1e-10)
    
    return sigma
