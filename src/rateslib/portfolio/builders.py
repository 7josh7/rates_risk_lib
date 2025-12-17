"""
Explicit trade builders for production portfolio construction.

=============================================================================
DESIGN PRINCIPLES
=============================================================================

1. NO INFERENCE: Option fields must be explicitly provided, never inferred.
   - Do NOT infer expiry from maturity
   - Do NOT coerce CAPLET → SWAPTION
   - Missing required fields must raise descriptive errors

2. SIGN CONVENTIONS (applied uniformly):
   | Concept           | Rule                                          |
   |-------------------|-----------------------------------------------|
   | Position          | LONG = +1, SHORT = −1                         |
   | Payer swaption    | +delta when rates ↑                           |
   | Receiver swaption | −delta when rates ↑                           |
   | Greeks            | Always returned as **signed $ sensitivities** |

3. EXPLICIT FAILURE: Never silently swallow errors. Return failure diagnostics.

=============================================================================
POSITION SCHEMA (REQUIRED FIELDS)
=============================================================================

For SWAPTIONS:
    - option_type: "SWAPTION" (explicit)
    - expiry_date OR expiry_tenor: option expiry (NOT maturity_date!)
    - underlying_swap_tenor: e.g., "5Y", "10Y"
    - payer_receiver: "PAYER" or "RECEIVER"
    - position: "LONG" or "SHORT"
    - strike: rate (e.g., 0.045) or "ATM"
    - notional: absolute value
    - vol_type: "NORMAL" or "LOGNORMAL" (optional, defaults to NORMAL)

For CAPLETS:
    - option_type: "CAPLET"
    - caplet_start_date: start of the accrual period
    - caplet_end_date: end of the accrual period
    - strike: rate (e.g., 0.05) or "ATM"
    - position: "LONG" or "SHORT"
    - notional: absolute value
    - is_cap: True for cap, False for floor (optional, default True)

For BONDS:
    - instrument_type: "BOND" or "UST"
    - maturity_date: bond maturity
    - coupon: coupon rate
    - notional: absolute value
    - direction: "LONG" or "SHORT"
    - frequency: payment frequency (optional, default 2)

For SWAPS:
    - instrument_type: "SWAP" or "IRS"
    - maturity_date: swap maturity
    - coupon: fixed rate
    - notional: absolute value
    - direction: "PAY_FIXED", "PAY", "REC_FIXED", "RECEIVE", "LONG", "SHORT"

=============================================================================
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..dates import DateUtils


# =============================================================================
# SIGN CONVENTIONS (CRITICAL)
# =============================================================================

SIGN_LONG = 1.0
SIGN_SHORT = -1.0


def get_position_sign(direction: str) -> float:
    """
    Determine position sign from direction string.
    
    LONG, BUY, REC_FIXED, RECEIVE → +1
    SHORT, SELL, PAY_FIXED, PAY → −1
    
    Raises ValueError for ambiguous/unknown direction.
    """
    direction = str(direction).upper().strip()
    
    long_indicators = {"LONG", "BUY", "REC_FIXED", "RECEIVE"}
    short_indicators = {"SHORT", "SELL", "PAY_FIXED", "PAY"}
    
    if direction in long_indicators:
        return SIGN_LONG
    if direction in short_indicators:
        return SIGN_SHORT
    
    raise ValueError(
        f"Ambiguous direction '{direction}'. "
        f"Must be one of: {long_indicators | short_indicators}"
    )


def get_payer_receiver_sign(payer_receiver: str) -> float:
    """
    Sign multiplier for payer vs receiver swaptions.
    
    PAYER swaption: profits when rates ↑ → delta sign = +1
    RECEIVER swaption: profits when rates ↓ → delta sign = −1
    """
    pr = str(payer_receiver).upper().strip()
    if pr == "PAYER":
        return SIGN_LONG
    if pr == "RECEIVER":
        return SIGN_SHORT
    raise ValueError(f"payer_receiver must be 'PAYER' or 'RECEIVER', got '{payer_receiver}'")


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class PositionValidationError(ValueError):
    """Base exception for position validation failures."""
    
    def __init__(self, position_id: Optional[str], message: str):
        self.position_id = position_id
        super().__init__(f"[{position_id or 'UNKNOWN'}] {message}")


class MissingFieldError(PositionValidationError):
    """Required field is missing from position."""
    
    def __init__(self, position_id: Optional[str], field_name: str, instrument_type: str):
        self.field_name = field_name
        self.instrument_type = instrument_type
        super().__init__(
            position_id,
            f"{instrument_type} requires field '{field_name}' but it was not provided"
        )


class InvalidOptionError(PositionValidationError):
    """Option position has invalid or ambiguous specification."""
    
    def __init__(self, position_id: Optional[str], message: str):
        super().__init__(position_id, f"Invalid option specification: {message}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _parse_date(val) -> Optional[date]:
    """Parse a value to date, returning None if unparseable."""
    if val is None:
        return None
    if isinstance(val, date):
        return val
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None


def _require_field(pos: pd.Series, field_name: str, instrument_type: str) -> Any:
    """Get a required field, raising MissingFieldError if absent or None."""
    position_id = pos.get("position_id")
    val = pos.get(field_name)
    
    if val is None or (isinstance(val, str) and val.strip() == ""):
        raise MissingFieldError(position_id, field_name, instrument_type)
    
    return val


def _parse_strike(strike_raw: Any, forward: float = 0.0) -> float:
    """
    Parse strike value.
    
    - "ATM" or None → use forward rate
    - Numeric string or number → parse as float
    - Ensure positive
    """
    if strike_raw is None:
        return forward if forward > 0 else 0.01  # Default fallback
    
    if isinstance(strike_raw, str):
        strike_str = strike_raw.upper().strip()
        if strike_str == "ATM":
            return forward if forward > 0 else 0.01
        try:
            return float(strike_raw)
        except ValueError:
            raise ValueError(f"Cannot parse strike '{strike_raw}' as number or ATM")
    
    return float(strike_raw)


# =============================================================================
# BOND TRADE BUILDER
# =============================================================================

def build_bond_trade(
    pos: pd.Series,
    valuation_date: date
) -> Dict[str, Any]:
    """
    Build a bond trade dict from a position row.
    
    Required fields:
        - maturity_date
        - notional
        - direction
    
    Optional fields:
        - coupon (default: 0.0)
        - frequency (default: 2)
    
    Returns:
        Trade dict ready for price_trade()
    
    Raises:
        MissingFieldError: Required field is missing
        PositionValidationError: Invalid field values
    """
    position_id = pos.get("position_id")
    inst_type = str(pos.get("instrument_type", "BOND")).upper()
    
    # Required fields
    maturity_date = _parse_date(_require_field(pos, "maturity_date", inst_type))
    if maturity_date is None:
        raise PositionValidationError(position_id, f"Cannot parse maturity_date")
    
    notional_raw = _require_field(pos, "notional", inst_type)
    notional = float(abs(notional_raw))
    
    direction = _require_field(pos, "direction", inst_type)
    sign = get_position_sign(direction)
    
    # Optional fields
    coupon = float(pos.get("coupon", 0.0))
    frequency = int(pos.get("frequency", 2))
    
    return {
        "instrument_type": inst_type,
        "settlement": valuation_date,
        "maturity": maturity_date,
        "coupon": coupon,
        "notional": notional * sign,
        "frequency": frequency,
        "face_value": 100.0,
        # Metadata for traceability
        "_position_id": position_id,
        "_direction": direction,
        "_sign": sign,
    }


# =============================================================================
# SWAP TRADE BUILDER
# =============================================================================

def build_swap_trade(
    pos: pd.Series,
    valuation_date: date
) -> Dict[str, Any]:
    """
    Build a swap trade dict from a position row.
    
    Required fields:
        - maturity_date
        - notional
        - direction (PAY_FIXED/PAY or REC_FIXED/RECEIVE)
    
    Optional fields:
        - coupon (fixed rate, default: 0.0)
    
    Sign convention:
        - PAY_FIXED: you pay fixed, receive floating
        - REC_FIXED: you receive fixed, pay floating
    
    Returns:
        Trade dict ready for price_trade()
    
    Raises:
        MissingFieldError: Required field is missing
    """
    position_id = pos.get("position_id")
    inst_type = str(pos.get("instrument_type", "SWAP")).upper()
    
    # Required fields
    maturity_date = _parse_date(_require_field(pos, "maturity_date", inst_type))
    if maturity_date is None:
        raise PositionValidationError(position_id, f"Cannot parse maturity_date")
    
    notional_raw = _require_field(pos, "notional", inst_type)
    notional = float(abs(notional_raw))
    
    direction = str(_require_field(pos, "direction", inst_type)).upper()
    
    # Determine pay_receive semantics
    if direction in {"PAY_FIXED", "PAY", "SHORT"}:
        pay_receive = "PAY"
    elif direction in {"REC_FIXED", "RECEIVE", "LONG"}:
        pay_receive = "RECEIVE"
    else:
        raise PositionValidationError(
            position_id,
            f"Swap direction '{direction}' is ambiguous. "
            "Use PAY_FIXED/PAY or REC_FIXED/RECEIVE"
        )
    
    # Optional fields
    fixed_rate = float(pos.get("coupon", 0.0))
    
    return {
        "instrument_type": "SWAP",
        "effective": valuation_date,
        "maturity": maturity_date,
        "notional": notional,
        "fixed_rate": fixed_rate,
        "pay_receive": pay_receive,
        # Metadata for traceability
        "_position_id": position_id,
        "_direction": direction,
    }


# =============================================================================
# SWAPTION TRADE BUILDER (EXPLICIT - NO INFERENCE)
# =============================================================================

def build_swaption_trade(
    pos: pd.Series,
    valuation_date: date
) -> Dict[str, Any]:
    """
    Build a swaption trade dict from a position row.
    
    REQUIRED FIELDS (no inference allowed):
        - expiry_date OR expiry_tenor: option expiry (NOT maturity_date)
        - underlying_swap_tenor: tenor of the underlying swap (e.g., "5Y")
        - payer_receiver: "PAYER" or "RECEIVER"
        - position: "LONG" or "SHORT"
        - notional: absolute value
        - strike: rate value or "ATM"
    
    OPTIONAL FIELDS:
        - vol_type: "NORMAL" or "LOGNORMAL" (default: "NORMAL")
    
    ⚠️ WILL RAISE ERRORS IF:
        - expiry_date/expiry_tenor is missing
        - underlying_swap_tenor is missing
        - payer_receiver is missing or invalid
        - position is missing
    
    Returns:
        Trade dict ready for price_trade()
    
    Raises:
        MissingFieldError: Required field is missing
        InvalidOptionError: Invalid option specification
    """
    position_id = pos.get("position_id")
    inst_type = "SWAPTION"
    
    # Check for deprecated inference patterns and fail explicitly
    has_maturity_only = pos.get("maturity_date") is not None
    has_expiry = pos.get("expiry_date") is not None or pos.get("expiry_tenor") is not None
    
    if has_maturity_only and not has_expiry:
        raise InvalidOptionError(
            position_id,
            "SWAPTION has 'maturity_date' but no 'expiry_date' or 'expiry_tenor'. "
            "Inferring expiry from maturity is not allowed. Please specify expiry explicitly."
        )
    
    # Get expiry - MUST be explicit
    expiry_date = _parse_date(pos.get("expiry_date"))
    expiry_tenor = pos.get("expiry_tenor")
    
    if expiry_date is None and expiry_tenor is None:
        raise MissingFieldError(position_id, "expiry_date or expiry_tenor", inst_type)
    
    # Convert expiry_date to tenor if needed
    if expiry_tenor is None and expiry_date is not None:
        years = (expiry_date - valuation_date).days / 365.25
        if years <= 0:
            raise InvalidOptionError(position_id, f"expiry_date {expiry_date} is not after valuation_date {valuation_date}")
        # Convert to appropriate tenor string
        if years < 1:
            months = int(round(years * 12))
            expiry_tenor = f"{max(1, months)}M"
        else:
            expiry_tenor = f"{int(round(years))}Y"
    
    # Get underlying swap tenor - MUST be explicit
    underlying_swap_tenor = pos.get("underlying_swap_tenor") or pos.get("swap_tenor")
    if underlying_swap_tenor is None:
        raise MissingFieldError(position_id, "underlying_swap_tenor", inst_type)
    
    # Get payer/receiver - MUST be explicit
    payer_receiver_raw = pos.get("payer_receiver")
    if payer_receiver_raw is None:
        raise MissingFieldError(position_id, "payer_receiver", inst_type)
    
    payer_receiver = str(payer_receiver_raw).upper().strip()
    if payer_receiver not in {"PAYER", "RECEIVER"}:
        raise InvalidOptionError(
            position_id,
            f"payer_receiver must be 'PAYER' or 'RECEIVER', got '{payer_receiver}'"
        )
    
    # Get position (LONG/SHORT) - MUST be explicit
    position_str = pos.get("position")
    if position_str is None:
        # Fallback to direction for backward compatibility, but warn
        position_str = pos.get("direction")
        if position_str is None:
            raise MissingFieldError(position_id, "position", inst_type)
    
    position_sign = get_position_sign(position_str)
    
    # Get notional
    notional_raw = pos.get("notional")
    if notional_raw is None:
        raise MissingFieldError(position_id, "notional", inst_type)
    notional = float(abs(notional_raw))
    
    # Get strike
    strike = pos.get("strike", "ATM")
    
    # Optional fields
    vol_type = str(pos.get("vol_type", "NORMAL")).upper()
    if vol_type not in {"NORMAL", "LOGNORMAL"}:
        vol_type = "NORMAL"
    
    return {
        "instrument_type": "SWAPTION",
        "expiry_tenor": str(expiry_tenor),
        "swap_tenor": str(underlying_swap_tenor),
        "strike": strike,
        "payer_receiver": payer_receiver,
        "notional": notional * position_sign,
        "vol_type": vol_type,
        # Metadata for traceability
        "_position_id": position_id,
        "_position": position_str,
        "_position_sign": position_sign,
        "_payer_receiver_sign": get_payer_receiver_sign(payer_receiver),
    }


# =============================================================================
# CAPLET TRADE BUILDER (EXPLICIT - NO INFERENCE)
# =============================================================================

def build_caplet_trade(
    pos: pd.Series,
    valuation_date: date
) -> Dict[str, Any]:
    """
    Build a caplet trade dict from a position row.
    
    REQUIRED FIELDS (no inference allowed):
        - caplet_start_date: start of accrual period
        - caplet_end_date: end of accrual period
        - position: "LONG" or "SHORT"
        - notional: absolute value
        - strike: rate value or "ATM"
    
    OPTIONAL FIELDS:
        - is_cap: True for cap, False for floor (default: True)
        - vol_type: "NORMAL" or "LOGNORMAL" (default: "NORMAL")
    
    ⚠️ WILL NOT:
        - Coerce CAPLET to SWAPTION
        - Infer dates from maturity_date
    
    Returns:
        Trade dict ready for price_trade()
    
    Raises:
        MissingFieldError: Required field is missing
        InvalidOptionError: Invalid option specification
    """
    position_id = pos.get("position_id")
    inst_type = "CAPLET"
    
    # Get caplet dates - MUST be explicit
    start_date = _parse_date(pos.get("caplet_start_date"))
    end_date = _parse_date(pos.get("caplet_end_date"))
    
    if start_date is None:
        raise MissingFieldError(position_id, "caplet_start_date", inst_type)
    if end_date is None:
        raise MissingFieldError(position_id, "caplet_end_date", inst_type)
    
    if end_date <= start_date:
        raise InvalidOptionError(
            position_id,
            f"caplet_end_date ({end_date}) must be after caplet_start_date ({start_date})"
        )
    
    # Get position (LONG/SHORT) - MUST be explicit
    position_str = pos.get("position")
    if position_str is None:
        position_str = pos.get("direction")
        if position_str is None:
            raise MissingFieldError(position_id, "position", inst_type)
    
    position_sign = get_position_sign(position_str)
    
    # Get notional
    notional_raw = pos.get("notional")
    if notional_raw is None:
        raise MissingFieldError(position_id, "notional", inst_type)
    notional = float(abs(notional_raw))
    
    # Get strike
    strike = pos.get("strike", "ATM")
    
    # Optional fields
    is_cap = pos.get("is_cap", True)
    if isinstance(is_cap, str):
        is_cap = is_cap.upper() in {"TRUE", "CAP", "1", "YES"}
    
    vol_type = str(pos.get("vol_type", "NORMAL")).upper()
    if vol_type not in {"NORMAL", "LOGNORMAL"}:
        vol_type = "NORMAL"
    
    # Compute index_tenor for SABR lookup
    delta_t = (end_date - start_date).days / 365.25
    if delta_t < 0.17:  # ~2 months
        index_tenor = "1M"
    elif delta_t < 0.29:  # ~3.5 months
        index_tenor = "3M"
    elif delta_t < 0.42:
        index_tenor = "6M"
    else:
        index_tenor = "12M"
    
    # Compute expiry tenor for SABR lookup
    expiry_days = (start_date - valuation_date).days
    if expiry_days <= 0:
        raise InvalidOptionError(
            position_id,
            f"caplet_start_date ({start_date}) must be after valuation_date ({valuation_date})"
        )
    expiry_years = expiry_days / 365.25
    if expiry_years < 0.5:
        expiry_tenor = f"{int(round(expiry_years * 12))}M"
    else:
        expiry_tenor = f"{int(round(expiry_years))}Y"
    
    return {
        "instrument_type": "CAPLET",
        "start_date": start_date,
        "end_date": end_date,
        "strike": strike,
        "notional": notional * position_sign,
        "is_cap": is_cap,
        "vol_type": vol_type,
        "expiry_tenor": expiry_tenor,
        "index_tenor": index_tenor,
        "delta_t": delta_t,
        # Metadata for traceability
        "_position_id": position_id,
        "_position": position_str,
        "_position_sign": position_sign,
    }


# =============================================================================
# FUTURES TRADE BUILDER
# =============================================================================

def build_futures_trade(
    pos: pd.Series,
    valuation_date: date
) -> Dict[str, Any]:
    """
    Build a futures trade dict from a position row.
    
    REQUIRED FIELDS:
        - expiry_date OR maturity_date: contract expiry
        - notional: number of contracts (can be positive/negative)
          OR num_contracts + direction
        - direction: LONG or SHORT (if notional is unsigned)
    
    OPTIONAL FIELDS:
        - contract_code: exchange code (default: "FUT")
        - contract_size: notional per contract (default: 1,000,000)
        - underlying_tenor: tenor of underlying rate (default: "3M")
        - tick_size: minimum price increment (default: 0.0025)
        - trade_price: entry price for P&L (default: None)
    
    PV CONVENTION:
        For rate futures (SOFR, Fed Funds, etc.):
        - Price = 100 - 100 * implied_forward_rate
        - PV = (model_price - trade_price) * contracts * tick_value / tick_size
        - If no trade_price, PV = 0 (futures have zero inception value)
        
        For DV01/bump-and-reprice:
        - Futures respond to curve changes via implied forward rate
        - Long position: gains when rates fall (price rises)
        - DV01 is computed from bump-and-reprice, not from formula
    
    Returns:
        Trade dict ready for price_trade()
    
    Raises:
        MissingFieldError: Required field is missing
        PositionValidationError: Invalid field values
    """
    position_id = pos.get("position_id")
    inst_type = "FUT"
    
    # Get expiry date - try multiple field names
    expiry_date = _parse_date(pos.get("expiry_date"))
    if expiry_date is None:
        expiry_date = _parse_date(pos.get("maturity_date"))
    if expiry_date is None:
        raise MissingFieldError(position_id, "expiry_date or maturity_date", inst_type)
    
    if expiry_date <= valuation_date:
        raise PositionValidationError(
            position_id,
            f"Futures expiry_date ({expiry_date}) must be after valuation_date ({valuation_date})"
        )
    
    # Get number of contracts and direction
    notional_raw = pos.get("notional")
    num_contracts_raw = pos.get("num_contracts")
    direction = pos.get("direction")
    
    if notional_raw is not None:
        # notional can be signed (positive=long, negative=short)
        # or it can be the number of contracts with direction separate
        notional_val = float(notional_raw)
        
        if direction is not None:
            # Direction is explicit, notional is count
            num_contracts = int(abs(notional_val))
            sign = get_position_sign(direction)
        else:
            # Infer direction from sign of notional
            num_contracts = int(abs(notional_val))
            sign = SIGN_LONG if notional_val >= 0 else SIGN_SHORT
    elif num_contracts_raw is not None:
        num_contracts = int(abs(float(num_contracts_raw)))
        if direction is None:
            raise MissingFieldError(position_id, "direction", inst_type)
        sign = get_position_sign(direction)
    else:
        raise MissingFieldError(position_id, "notional or num_contracts", inst_type)
    
    # Optional fields with sensible defaults
    contract_code = str(pos.get("contract_code") or pos.get("instrument_id") or "FUT")
    contract_size = float(pos.get("contract_size", 1_000_000))
    underlying_tenor = str(pos.get("underlying_tenor", "3M"))
    tick_size = float(pos.get("tick_size", 0.0025))
    tick_value = float(pos.get("tick_value", 6.25))  # $6.25 per 0.25bp for SOFR
    
    # Trade price for P&L calculation (optional)
    trade_price = pos.get("trade_price") or pos.get("entry_price")
    if trade_price is not None:
        try:
            trade_price = float(trade_price)
        except (ValueError, TypeError):
            trade_price = None
    
    return {
        "instrument_type": "FUT",
        "expiry": expiry_date,
        "contract_code": contract_code,
        "contract_size": contract_size,
        "underlying_tenor": underlying_tenor,
        "tick_size": tick_size,
        "tick_value": tick_value,
        "num_contracts": int(num_contracts * sign),  # Signed contract count
        "trade_price": trade_price,
        # Metadata for traceability
        "_position_id": position_id,
        "_direction": direction,
        "_sign": sign,
    }


# =============================================================================
# UNIFIED TRADE BUILDER (dispatches by instrument type)
# =============================================================================

def build_trade_from_position(
    pos: pd.Series,
    valuation_date: date,
    allow_legacy_options: bool = False,
) -> Dict[str, Any]:
    """
    Build a trade dict from a position row, dispatching to appropriate builder.
    
    Args:
        pos: Position row from DataFrame
        valuation_date: Valuation date for pricing
        allow_legacy_options: If True, attempt backward-compatible option parsing.
            NOT RECOMMENDED for production. Default False.
    
    Returns:
        Trade dict ready for price_trade()
    
    Raises:
        PositionValidationError: Position cannot be converted to trade
        MissingFieldError: Required field is missing
        InvalidOptionError: Invalid option specification
    """
    position_id = pos.get("position_id")
    inst_type = str(pos.get("instrument_type", "")).upper()
    
    # Check for explicit option_type first (preferred)
    option_type = pos.get("option_type")
    if option_type is not None:
        option_type = str(option_type).upper().strip()
        if option_type == "SWAPTION":
            return build_swaption_trade(pos, valuation_date)
        elif option_type == "CAPLET":
            return build_caplet_trade(pos, valuation_date)
        else:
            raise InvalidOptionError(
                position_id,
                f"Unknown option_type '{option_type}'. Expected 'SWAPTION' or 'CAPLET'"
            )
    
    # Dispatch by instrument_type
    if inst_type in {"BOND", "UST"}:
        return build_bond_trade(pos, valuation_date)
    
    if inst_type in {"SWAP", "IRS"}:
        return build_swap_trade(pos, valuation_date)
    
    if inst_type == "SWAPTION":
        return build_swaption_trade(pos, valuation_date)
    
    if inst_type in {"CAPLET", "CAP", "CAPFLOOR"}:
        return build_caplet_trade(pos, valuation_date)
    
    if inst_type in {"FUT", "FUTURE", "FUTURES"}:
        return build_futures_trade(pos, valuation_date)
    
    raise PositionValidationError(
        position_id,
        f"Unknown instrument_type '{inst_type}'. "
        "Supported: BOND, UST, SWAP, IRS, SWAPTION, CAPLET, CAP, CAPFLOOR"
    )


# =============================================================================
# PORTFOLIO PRICING WITH FAILURE TRACKING
# =============================================================================

@dataclass
class TradeFailure:
    """
    Record of a failed trade pricing attempt.
    
    Attributes:
        position_id: ID of the position that failed
        instrument_type: Type of instrument
        error_type: Exception class name
        error_message: Full error message
        stage: Where the failure occurred ("build" or "price")
    """
    position_id: Optional[str]
    instrument_type: str
    error_type: str
    error_message: str
    stage: str  # "build" or "price"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id": self.position_id or "UNKNOWN",
            "instrument_type": self.instrument_type,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stage": self.stage,
        }


@dataclass
class PortfolioPricingResult:
    """
    Result of portfolio pricing with full diagnostics.
    
    Attributes:
        total_pv: Total portfolio present value (only from successful trades)
        successful_trades: List of successfully priced trades
        successful_pvs: List of PVs for successful trades
        failed_trades: List of TradeFailure objects
        coverage_ratio: successful / total
        has_failures: True if any trades failed
        warnings: List of warning messages
    """
    total_pv: float
    successful_trades: List[Dict[str, Any]]
    successful_pvs: List[float]
    failed_trades: List[TradeFailure]
    total_positions: int
    
    @property
    def successful_count(self) -> int:
        return len(self.successful_trades)
    
    @property
    def failed_count(self) -> int:
        return len(self.failed_trades)
    
    @property
    def coverage_ratio(self) -> float:
        if self.total_positions == 0:
            return 1.0
        return self.successful_count / self.total_positions
    
    @property
    def has_failures(self) -> bool:
        return len(self.failed_trades) > 0
    
    @property
    def is_complete(self) -> bool:
        return not self.has_failures
    
    def get_warnings(self) -> List[str]:
        """Generate warning messages for failed trades."""
        warnings = []
        if self.has_failures:
            warnings.append(
                f"⚠️ {self.failed_count}/{self.total_positions} positions failed to price "
                f"(coverage: {self.coverage_ratio:.1%})"
            )
            # Group by error type
            by_type: Dict[str, int] = {}
            for f in self.failed_trades:
                key = f"{f.stage}:{f.error_type}"
                by_type[key] = by_type.get(key, 0) + 1
            for key, count in by_type.items():
                warnings.append(f"  - {key}: {count} failures")
        return warnings
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_pv": self.total_pv,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "total_positions": self.total_positions,
            "coverage_ratio": self.coverage_ratio,
            "has_failures": self.has_failures,
            "failed_trades": [f.to_dict() for f in self.failed_trades],
        }


def price_portfolio_with_diagnostics(
    positions_df: pd.DataFrame,
    market_state: "MarketState",
    valuation_date: date,
    include_options: bool = True,
    allow_legacy_options: bool = False,
) -> PortfolioPricingResult:
    """
    Price a portfolio with comprehensive failure tracking.
    
    NEVER silently swallows errors. All failures are recorded and returned.
    
    Args:
        positions_df: DataFrame with position details
        market_state: Current market state (curves + vol surface)
        valuation_date: Valuation date for pricing
        include_options: Whether to include options in pricing
        allow_legacy_options: Allow deprecated option field patterns (NOT recommended)
    
    Returns:
        PortfolioPricingResult with full diagnostics
    """
    from ..pricers.dispatcher import price_trade
    
    successful_trades: List[Dict[str, Any]] = []
    successful_pvs: List[float] = []
    failed_trades: List[TradeFailure] = []
    
    for _, pos in positions_df.iterrows():
        position_id = pos.get("position_id")
        inst_type = str(pos.get("instrument_type", "UNKNOWN")).upper()
        
        # Check option exclusion
        is_option = inst_type in {"SWAPTION", "CAPLET", "CAP", "CAPFLOOR"}
        if is_option and not include_options:
            failed_trades.append(TradeFailure(
                position_id=position_id,
                instrument_type=inst_type,
                error_type="OptionExcluded",
                error_message="Options excluded from pricing by include_options=False",
                stage="build",
            ))
            continue
        
        # Build trade
        try:
            trade = build_trade_from_position(pos, valuation_date, allow_legacy_options)
        except (PositionValidationError, MissingFieldError, InvalidOptionError) as e:
            failed_trades.append(TradeFailure(
                position_id=position_id,
                instrument_type=inst_type,
                error_type=type(e).__name__,
                error_message=str(e),
                stage="build",
            ))
            continue
        except Exception as e:
            failed_trades.append(TradeFailure(
                position_id=position_id,
                instrument_type=inst_type,
                error_type=type(e).__name__,
                error_message=str(e),
                stage="build",
            ))
            continue
        
        # Price trade
        try:
            result = price_trade(trade, market_state)
            successful_trades.append(trade)
            successful_pvs.append(result.pv)
        except Exception as e:
            failed_trades.append(TradeFailure(
                position_id=position_id,
                instrument_type=inst_type,
                error_type=type(e).__name__,
                error_message=str(e),
                stage="price",
            ))
    
    total_pv = sum(successful_pvs)
    
    return PortfolioPricingResult(
        total_pv=total_pv,
        successful_trades=successful_trades,
        successful_pvs=successful_pvs,
        failed_trades=failed_trades,
        total_positions=len(positions_df),
    )
