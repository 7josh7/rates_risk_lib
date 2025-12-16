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

from typing import TYPE_CHECKING
from ..options.swaption import SwaptionPricer
from ..options.caplet import CapletPricer
from ..dates import DateUtils
from .sabr_surface import make_bucket_key

if TYPE_CHECKING:
    from ..market_state import CurveState


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


def _resolve_columns(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column present from a candidate list."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _compute_forward(
    instrument: str,
    expiry: str,
    tenor: str,
    curve_state
) -> Dict[str, float]:
    """
    Compute forward and maturity for a quote row.
    """
    instrument = instrument.upper()
    if instrument == "SWAPTION":
        expiry_years = DateUtils.tenor_to_years(expiry)
        swap_years = DateUtils.tenor_to_years(tenor)
        pricer = SwaptionPricer(curve_state.discount_curve, curve_state.projection_curve)
        fwd, annuity = pricer.forward_swap_rate(expiry_years, swap_years)
        return {"F0": fwd, "T": expiry_years, "annuity": annuity}

    # Default to caplet conventions
    start = DateUtils.tenor_to_years(expiry)
    accrual = DateUtils.tenor_to_years(tenor)
    end = start + accrual
    pricer = CapletPricer(curve_state.discount_curve, curve_state.projection_curve)
    fwd = pricer.forward_rate(start, end)
    return {"F0": fwd, "T": start, "delta_t": accrual}


def _normalize_strike(
    forward: float,
    row: pd.Series,
    quote_type: str
) -> Optional[float]:
    """
    Convert quote notation (ATM, BPS, absolute) into an absolute strike.
    """
    quote_type = str(quote_type).upper()

    # Direct strike numeric value
    if isinstance(row.get("strike"), (int, float)) and quote_type not in {"BPS", "ATM"}:
        return float(row["strike"])

    # ATM or explicit ATMF label
    if quote_type in {"ATM", "ATMF"}:
        return forward

    # Basis point relative strikes
    if quote_type in {"BPS", "ATM+BPS", "ATM_BP", "ATM+BPS", "ATM-BP"}:
        try:
            bp = float(row.get("strike", 0.0))
            return forward + bp / 10000.0
        except (TypeError, ValueError):
            return None

    # String like "+25BP" or "-50BP"
    strike_val = row.get("strike")
    if isinstance(strike_val, str) and strike_val.upper().endswith("BP"):
        try:
            bp_val = float(strike_val[:-2])
            return forward + bp_val / 10000.0
        except ValueError:
            return None

    return None


def normalize_vol_quotes(
    raw_quotes: Union[pd.DataFrame, str],
    curve_state,
    instrument_hint: str = "SWAPTION"
) -> pd.DataFrame:
    """
    Convert heterogeneous vol quotes into a canonical DataFrame.

    Canonical columns:
        instrument, expiry, tenor, bucket_key, T, F0, K,
        sigma_mkt, vol_type, shift, quote_type, quote_value, source

    Unsupported or malformed rows are dropped; callers should inspect length.
    """
    if isinstance(raw_quotes, str):
        df_raw = pd.read_csv(raw_quotes)
    else:
        df_raw = raw_quotes.copy()

    df_raw.columns = [c.strip().lower() for c in df_raw.columns]

    expiry_col = _resolve_columns(df_raw, ["expiry"])
    tenor_col = _resolve_columns(df_raw, ["tenor", "underlying_tenor", "index_tenor"])
    vol_col = _resolve_columns(df_raw, ["ivol", "vol"])
    strike_type_col = _resolve_columns(df_raw, ["quote_type", "strike_type"])
    strike_val_col = _resolve_columns(df_raw, ["quote_value", "strike"])
    instrument_col = _resolve_columns(df_raw, ["instrument", "instrument_type"])

    if expiry_col is None or tenor_col is None or vol_col is None:
        raise ValueError("Vol quotes must include expiry, tenor and vol columns.")

    records = []
    for _, row in df_raw.iterrows():
        instrument = str(row.get(instrument_col, instrument_hint)).upper() if instrument_col else instrument_hint.upper()
        expiry = str(row[expiry_col]).strip()
        tenor = str(row[tenor_col]).strip()
        vol_type = str(row.get("vol_type", "NORMAL")).upper()
        shift = float(row.get("shift", 0.0))

        forward_info = _compute_forward(instrument, expiry, tenor, curve_state)
        F0 = forward_info["F0"]
        T = forward_info["T"]

        quote_type = str(row.get(strike_type_col, "ATM")).upper() if strike_type_col else "ATM"
        row_local = row.copy()
        if strike_val_col:
            row_local["strike"] = row[strike_val_col]
        K = _normalize_strike(F0, row_local, quote_type)

        if K is None:
            # Skip malformed rows but annotate decision for the caller
            continue

        sigma_mkt = float(row[vol_col])
        bucket = make_bucket_key(expiry, tenor)

        records.append(
            {
                "instrument": instrument,
                "expiry": expiry,
                "tenor": tenor,
                "bucket_key": bucket,
                "T": T,
                "F0": F0,
                "K": K,
                "sigma_mkt": sigma_mkt,
                "vol_type": vol_type,
                "shift": shift,
                "quote_type": quote_type,
                "quote_value": row_local.get("strike", 0.0),
                "source": row.get("source"),
            }
        )

    return pd.DataFrame(records)


__all__ = [
    "VolQuote",
    "load_vol_quotes",
    "quotes_to_dataframe",
    "convert_normal_to_black",
    "convert_black_to_normal",
    "generate_strike_ladder",
    "normalize_vol_quotes",
]
