"""
Volatility quote handling and loading.

Provides utilities for:
- Loading vol quotes from CSV
- Parsing strike conventions (ATM, +/-25bp, etc.)
- Converting between vol types
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np


@dataclass
class VolQuote:
    """
    A single volatility quote.
    
    Attributes:
        quote_date: Date of the quote
        expiry: Expiry tenor (e.g., "1Y")
        underlying_tenor: Underlying swap tenor (e.g., "5Y")
        strike: Strike value or convention (e.g., "ATM", "+25bp", 0.05)
        vol: Implied volatility
        vol_type: "NORMAL" or "LOGNORMAL"
        shift: Shift for shifted lognormal (default 0)
    """
    quote_date: date
    expiry: str
    underlying_tenor: str
    strike: Union[str, float]
    vol: float
    vol_type: str = "NORMAL"
    shift: float = 0.0
    
    def strike_value(self, forward: float) -> float:
        """
        Convert strike convention to absolute value.
        
        Args:
            forward: Forward rate for ATM reference
            
        Returns:
            Absolute strike value
        """
        if isinstance(self.strike, (int, float)):
            return float(self.strike)
        
        strike_str = str(self.strike).upper().strip()
        
        if strike_str == "ATM":
            return forward
        elif strike_str.endswith("BP"):
            # Parse +/-XXbp format
            bp_str = strike_str.replace("BP", "")
            bp_value = float(bp_str) / 10000.0
            return forward + bp_value
        else:
            # Try to parse as number
            try:
                return float(strike_str)
            except ValueError:
                raise ValueError(f"Unknown strike format: {self.strike}")


def load_vol_quotes(
    filepath: str,
    quote_date: Optional[date] = None
) -> List[VolQuote]:
    """
    Load volatility quotes from CSV file.
    
    Expected CSV format:
    date, expiry, underlying_tenor, strike, vol, vol_type, shift
    
    Args:
        filepath: Path to CSV file
        quote_date: Optional filter for specific date
        
    Returns:
        List of VolQuote objects
    """
    df = pd.read_csv(filepath)
    
    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Parse dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Filter by date if specified
    if quote_date is not None and 'date' in df.columns:
        df = df[df['date'] == quote_date]
    
    # Set defaults
    if 'vol_type' not in df.columns:
        df['vol_type'] = 'NORMAL'
    if 'shift' not in df.columns:
        df['shift'] = 0.0
    
    quotes = []
    for _, row in df.iterrows():
        quote = VolQuote(
            quote_date=row.get('date', date.today()),
            expiry=str(row['expiry']).strip(),
            underlying_tenor=str(row['underlying_tenor']).strip(),
            strike=row['strike'],
            vol=float(row['vol']),
            vol_type=str(row['vol_type']).upper().strip(),
            shift=float(row.get('shift', 0.0))
        )
        quotes.append(quote)
    
    return quotes


def quotes_to_dataframe(
    quotes: List[VolQuote],
    forward: float
) -> pd.DataFrame:
    """
    Convert quotes to DataFrame with absolute strikes.
    
    Args:
        quotes: List of VolQuote objects
        forward: Forward rate for strike conversion
        
    Returns:
        DataFrame with columns [strike, vol, vol_type, shift]
    """
    data = []
    for q in quotes:
        data.append({
            'strike': q.strike_value(forward),
            'vol': q.vol,
            'vol_type': q.vol_type,
            'shift': q.shift
        })
    
    return pd.DataFrame(data)


def convert_normal_to_black(
    sigma_n: float,
    F: float,
    K: float,
    T: float,
    shift: float = 0.0
) -> float:
    """
    Convert normal vol to Black vol (approximate).
    
    Uses the ATM-centered approximation:
    sigma_B ≈ sigma_N / F * (1 + corrections)
    
    Args:
        sigma_n: Normal (Bachelier) implied vol
        F: Forward rate
        K: Strike
        T: Time to expiry
        shift: Shift for negative rates
        
    Returns:
        Black implied volatility
    """
    F_s = F + shift
    K_s = K + shift
    
    if F_s <= 0 or K_s <= 0:
        raise ValueError("Shifted forward and strike must be positive for Black vol")
    
    # Simple approximation
    if abs(F_s - K_s) < 1e-10:
        # ATM
        return sigma_n / F_s
    else:
        # OTM: use geometric mean approximation
        fk_sqrt = np.sqrt(F_s * K_s)
        log_fk = np.log(F_s / K_s)
        return sigma_n / fk_sqrt * (1 + log_fk**2 / 24)


def convert_black_to_normal(
    sigma_b: float,
    F: float,
    K: float,
    T: float,
    shift: float = 0.0
) -> float:
    """
    Convert Black vol to normal vol (approximate).
    
    Args:
        sigma_b: Black implied vol
        F: Forward rate
        K: Strike
        T: Time to expiry
        shift: Shift for negative rates
        
    Returns:
        Normal (Bachelier) implied volatility
    """
    F_s = F + shift
    K_s = K + shift
    
    if abs(F_s - K_s) < 1e-10:
        # ATM
        return sigma_b * F_s
    else:
        # OTM
        fk_sqrt = np.sqrt(F_s * K_s)
        log_fk = np.log(F_s / K_s)
        return sigma_b * fk_sqrt * (1 - log_fk**2 / 24)


def generate_strike_ladder(
    forward: float,
    strikes_bp: List[int] = None
) -> List[float]:
    """
    Generate standard strike ladder around forward.
    
    Args:
        forward: ATM forward rate
        strikes_bp: Strike offsets in basis points (default: standard set)
        
    Returns:
        List of absolute strike values
    """
    if strikes_bp is None:
        strikes_bp = [-100, -50, -25, 0, 25, 50, 100]
    
    return [forward + bp / 10000.0 for bp in strikes_bp]
