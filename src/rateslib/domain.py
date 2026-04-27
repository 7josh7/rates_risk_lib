"""
Typed domain objects for trade normalization and pricing policy.

This module introduces an explicit boundary between external, dictionary-style
 inputs and the internal pricing/risk engines. Existing APIs can continue to
 pass dictionaries, but dispatchers can normalize those payloads into typed
 trade objects before pricing.
"""

from dataclasses import dataclass, field, fields
from datetime import date
from typing import Any, Dict, Mapping, Optional, Union


class TradeNormalizationError(ValueError):
    """Raised when a trade payload cannot be normalized into a typed trade."""


@dataclass(frozen=True)
class PricingPolicy:
    """
    Controls how pricing workflows handle model fallbacks.

    Attributes:
        sabr_bucket_fallback:
            "allow" -> use nearest bucket silently
            "warn"  -> use nearest bucket and surface a warning
            "error" -> require an exact bucket match
        allow_zero_option_vol:
            If False, option pricing raises when no SABR bucket or explicit
            volatility is available.
        record_audit:
            Include audit metadata in PricerOutput.
    """

    sabr_bucket_fallback: str = "allow"
    allow_zero_option_vol: bool = True
    record_audit: bool = True

    def __post_init__(self) -> None:
        valid_modes = {"allow", "warn", "error"}
        mode = str(self.sabr_bucket_fallback).lower()
        if mode not in valid_modes:
            raise ValueError(
                "sabr_bucket_fallback must be one of "
                f"{sorted(valid_modes)}, got {self.sabr_bucket_fallback!r}"
            )
        object.__setattr__(self, "sabr_bucket_fallback", mode)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the policy."""
        return {
            "sabr_bucket_fallback": self.sabr_bucket_fallback,
            "allow_zero_option_vol": self.allow_zero_option_vol,
            "record_audit": self.record_audit,
        }


@dataclass(frozen=True)
class TradeBase:
    """Base class for typed trade inputs."""

    instrument_type: str
    trade_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def position_id(self) -> Optional[str]:
        """Prefer the explicit trade id, then a position id in metadata."""
        if self.trade_id:
            return self.trade_id
        return self.metadata.get("_position_id")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the typed trade back to the legacy dictionary schema."""
        result: Dict[str, Any] = {}
        for f in fields(self):
            result[f.name] = getattr(self, f.name)
        for key, value in self.metadata.items():
            if key not in result:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Compatibility helper for legacy dict-style access patterns."""
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class BondTrade(TradeBase):
    settlement: date = field(default_factory=date.today)
    maturity: date = field(default_factory=date.today)
    coupon: float = 0.0
    notional: float = 1.0
    frequency: int = 2
    face_value: float = 100.0


@dataclass(frozen=True)
class SwapTrade(TradeBase):
    effective: date = field(default_factory=date.today)
    maturity: date = field(default_factory=date.today)
    notional: float = 1.0
    fixed_rate: float = 0.0
    pay_receive: str = "PAY"


@dataclass(frozen=True)
class FuturesTrade(TradeBase):
    expiry: date = field(default_factory=date.today)
    contract_code: str = "FUT"
    contract_size: float = 1_000_000.0
    tick_size: float = 0.0025
    tick_value: float = 6.25
    underlying_tenor: str = "3M"
    num_contracts: int = 1
    trade_price: Optional[float] = None


@dataclass(frozen=True)
class SwaptionTrade(TradeBase):
    expiry_tenor: str = "1Y"
    swap_tenor: str = "5Y"
    strike: Any = "ATM"
    payer_receiver: str = "PAYER"
    notional: float = 1.0
    vol_type: str = "NORMAL"
    vol: Optional[float] = None
    shift: float = 0.0


@dataclass(frozen=True)
class CapletTrade(TradeBase):
    start_date: date = field(default_factory=date.today)
    end_date: date = field(default_factory=date.today)
    strike: Any = "ATM"
    notional: float = 1.0
    is_cap: bool = True
    vol_type: str = "NORMAL"
    expiry_tenor: str = "0D"
    index_tenor: str = "3M"
    delta_t: Optional[float] = None
    vol: Optional[float] = None
    shift: float = 0.0


TypedTrade = Union[BondTrade, SwapTrade, FuturesTrade, SwaptionTrade, CapletTrade]
TradeLike = Union[TypedTrade, Mapping[str, Any], Any]


def _coerce_metadata(trade: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep non-core metadata so audit trails survive normalization."""
    core_keys = {
        "instrument_type",
        "trade_id",
        "settlement",
        "effective",
        "maturity",
        "coupon",
        "notional",
        "frequency",
        "face_value",
        "fixed_rate",
        "pay_receive",
        "expiry",
        "contract_code",
        "contract_size",
        "tick_size",
        "tick_value",
        "underlying_tenor",
        "num_contracts",
        "trade_price",
        "expiry_tenor",
        "swap_tenor",
        "strike",
        "payer_receiver",
        "vol_type",
        "vol",
        "shift",
        "start_date",
        "end_date",
        "is_cap",
        "index_tenor",
        "delta_t",
    }
    return {key: value for key, value in trade.items() if key not in core_keys}


def _coerce_trade_mapping(trade: TradeLike) -> Mapping[str, Any]:
    """Accept dict-like objects and objects that implement to_dict()."""
    if isinstance(trade, Mapping):
        return trade
    if hasattr(trade, "to_dict"):
        candidate = trade.to_dict()
        if isinstance(candidate, Mapping):
            return candidate
    raise TradeNormalizationError(
        f"Unsupported trade payload type: {type(trade).__name__}"
    )


def _require(trade: Mapping[str, Any], field_name: str) -> Any:
    if field_name not in trade:
        raise TradeNormalizationError(
            f"Trade is missing required field {field_name!r}"
        )
    return trade[field_name]


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().upper() in {"TRUE", "1", "YES", "Y", "CAP"}
    return bool(value)


def normalize_trade(trade: TradeLike) -> TypedTrade:
    """
    Normalize dict-style or typed trades into an explicit trade object.

    Existing callers can continue to pass dictionaries, while internal code can
    rely on the returned object shape.
    """
    if isinstance(trade, TradeBase):
        return trade

    mapping = _coerce_trade_mapping(trade)
    inst = str(mapping.get("instrument_type", "")).upper()
    trade_id = mapping.get("trade_id") or mapping.get("_position_id")
    metadata = _coerce_metadata(mapping)

    if inst in {"BOND", "UST"}:
        return BondTrade(
            instrument_type=inst,
            trade_id=trade_id,
            metadata=metadata,
            settlement=_require(mapping, "settlement"),
            maturity=_require(mapping, "maturity"),
            coupon=float(mapping.get("coupon", 0.0)),
            notional=float(mapping.get("notional", 1.0)),
            frequency=int(mapping.get("frequency", 2)),
            face_value=float(mapping.get("face_value", 100.0)),
        )

    if inst in {"SWAP", "IRS"}:
        return SwapTrade(
            instrument_type=inst,
            trade_id=trade_id,
            metadata=metadata,
            effective=_require(mapping, "effective"),
            maturity=_require(mapping, "maturity"),
            notional=float(mapping.get("notional", 1.0)),
            fixed_rate=float(mapping.get("fixed_rate", 0.0)),
            pay_receive=str(mapping.get("pay_receive", "PAY")).upper(),
        )

    if inst in {"FUT", "FUTURE", "FUTURES"}:
        trade_price = mapping.get("trade_price")
        if trade_price is not None:
            trade_price = float(trade_price)
        return FuturesTrade(
            instrument_type=inst,
            trade_id=trade_id,
            metadata=metadata,
            expiry=_require(mapping, "expiry"),
            contract_code=str(mapping.get("contract_code", "FUT")),
            contract_size=float(mapping.get("contract_size", 1_000_000)),
            tick_size=float(mapping.get("tick_size", 0.0025)),
            tick_value=float(mapping.get("tick_value", 6.25)),
            underlying_tenor=str(mapping.get("underlying_tenor", "3M")),
            num_contracts=int(mapping.get("num_contracts", 1)),
            trade_price=trade_price,
        )

    if inst == "SWAPTION":
        vol = mapping.get("vol")
        if vol is not None:
            vol = float(vol)
        return SwaptionTrade(
            instrument_type=inst,
            trade_id=trade_id,
            metadata=metadata,
            expiry_tenor=str(_require(mapping, "expiry_tenor")),
            swap_tenor=str(_require(mapping, "swap_tenor")),
            strike=mapping.get("strike", "ATM"),
            payer_receiver=str(mapping.get("payer_receiver", "PAYER")).upper(),
            notional=float(mapping.get("notional", 1.0)),
            vol_type=str(mapping.get("vol_type", "NORMAL")).upper(),
            vol=vol,
            shift=float(mapping.get("shift", 0.0)),
        )

    if inst in {"CAPLET", "CAP", "CAPFLOOR"}:
        vol = mapping.get("vol")
        if vol is not None:
            vol = float(vol)
        delta_t = mapping.get("delta_t")
        if delta_t is not None:
            delta_t = float(delta_t)
        return CapletTrade(
            instrument_type=inst,
            trade_id=trade_id,
            metadata=metadata,
            start_date=_require(mapping, "start_date"),
            end_date=_require(mapping, "end_date"),
            strike=mapping.get("strike", "ATM"),
            notional=float(mapping.get("notional", 1.0)),
            is_cap=_coerce_bool(mapping.get("is_cap", True), default=True),
            vol_type=str(mapping.get("vol_type", "NORMAL")).upper(),
            expiry_tenor=str(mapping.get("expiry_tenor", "0D")),
            index_tenor=str(mapping.get("index_tenor", "3M")),
            delta_t=delta_t,
            vol=vol,
            shift=float(mapping.get("shift", 0.0)),
        )

    raise TradeNormalizationError(
        f"Unsupported instrument_type for normalization: {inst!r}"
    )


__all__ = [
    "PricingPolicy",
    "TradeNormalizationError",
    "TradeBase",
    "BondTrade",
    "SwapTrade",
    "FuturesTrade",
    "SwaptionTrade",
    "CapletTrade",
    "TypedTrade",
    "TradeLike",
    "normalize_trade",
]
