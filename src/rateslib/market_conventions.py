"""
Named market-convention templates for product and currency-aware pricing.

This module provides a lightweight registry of standard market templates so
callers can resolve realistic conventions from a trade's instrument type,
currency, or an explicit market-convention identifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .calendars import UnitedStatesHolidayCalendar, WeekendOnlyCalendar
from .conventions import (
    BusinessDayConvention,
    CompoundingConvention,
    Conventions,
    DayCount,
)


class MarketConventionError(ValueError):
    """Raised when a requested market-convention template cannot be resolved."""


@dataclass(frozen=True)
class ConventionSpec:
    """Serializable convention specification used by the template registry."""

    day_count: DayCount
    business_day: BusinessDayConvention
    compounding: CompoundingConvention = CompoundingConvention.CONTINUOUS
    payment_frequency: int = 1
    settlement_days: int = 2
    holiday_calendar: str = "WEEKEND_ONLY"
    end_of_month: bool = False
    stub: str = "short_front"

    def materialize(self) -> Conventions:
        """Convert the specification into a fresh Conventions object."""
        return Conventions(
            day_count=self.day_count,
            business_day=self.business_day,
            compounding=self.compounding,
            payment_frequency=self.payment_frequency,
            settlement_days=self.settlement_days,
            holiday_calendar=_calendar_from_name(self.holiday_calendar),
            end_of_month=self.end_of_month,
            stub=self.stub,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day_count": self.day_count.value,
            "business_day": self.business_day.value,
            "compounding": self.compounding.value,
            "payment_frequency": self.payment_frequency,
            "settlement_days": self.settlement_days,
            "holiday_calendar": self.holiday_calendar,
            "end_of_month": self.end_of_month,
            "stub": self.stub,
        }


@dataclass(frozen=True)
class MarketConventionTemplate:
    """Template describing conventions for a product family in one currency."""

    template_id: str
    currency: str
    instrument_family: str
    description: str
    bond: Optional[ConventionSpec] = None
    fixed_leg: Optional[ConventionSpec] = None
    float_leg: Optional[ConventionSpec] = None
    swaption_fixed_frequency: Optional[int] = None
    swaption_float_frequency: Optional[int] = None
    default_underlying_tenor: Optional[str] = None
    tags: Tuple[str, ...] = ()

    def resolve(
        self,
        instrument_type: str,
        currency: Optional[str],
        source: str,
    ) -> "ResolvedMarketConvention":
        return ResolvedMarketConvention(
            instrument_type=instrument_type,
            currency=currency or self.currency,
            source=source,
            template=self,
            bond_conventions=self.bond.materialize() if self.bond else None,
            fixed_leg_conventions=self.fixed_leg.materialize() if self.fixed_leg else None,
            float_leg_conventions=self.float_leg.materialize() if self.float_leg else None,
            swaption_fixed_frequency=self.swaption_fixed_frequency,
            swaption_float_frequency=self.swaption_float_frequency,
            default_underlying_tenor=self.default_underlying_tenor,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "currency": self.currency,
            "instrument_family": self.instrument_family,
            "description": self.description,
            "bond": self.bond.to_dict() if self.bond else None,
            "fixed_leg": self.fixed_leg.to_dict() if self.fixed_leg else None,
            "float_leg": self.float_leg.to_dict() if self.float_leg else None,
            "swaption_fixed_frequency": self.swaption_fixed_frequency,
            "swaption_float_frequency": self.swaption_float_frequency,
            "default_underlying_tenor": self.default_underlying_tenor,
            "tags": list(self.tags),
        }


@dataclass
class ResolvedMarketConvention:
    """Materialized convention resolution for a specific trade."""

    instrument_type: str
    currency: Optional[str]
    source: str
    template: Optional[MarketConventionTemplate] = None
    bond_conventions: Optional[Conventions] = None
    fixed_leg_conventions: Optional[Conventions] = None
    float_leg_conventions: Optional[Conventions] = None
    swaption_fixed_frequency: Optional[int] = None
    swaption_float_frequency: Optional[int] = None
    default_underlying_tenor: Optional[str] = None

    @property
    def template_id(self) -> Optional[str]:
        if self.template is None:
            return None
        return self.template.template_id

    def primary_conventions(self) -> Optional[Conventions]:
        """Return the convention block most relevant for settlement logic."""
        if self.bond_conventions is not None:
            return self.bond_conventions
        if self.fixed_leg_conventions is not None:
            return self.fixed_leg_conventions
        return self.float_leg_conventions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "currency": self.currency,
            "source": self.source,
            "instrument_type": self.instrument_type,
            "bond_conventions": _conventions_to_dict(self.bond_conventions),
            "fixed_leg_conventions": _conventions_to_dict(self.fixed_leg_conventions),
            "float_leg_conventions": _conventions_to_dict(self.float_leg_conventions),
            "swaption_fixed_frequency": self.swaption_fixed_frequency,
            "swaption_float_frequency": self.swaption_float_frequency,
            "default_underlying_tenor": self.default_underlying_tenor,
        }


def _calendar_from_name(name: str):
    calendar_name = str(name or "WEEKEND_ONLY").upper()
    if calendar_name == "US_FED":
        return UnitedStatesHolidayCalendar()
    if calendar_name == "WEEKEND_ONLY":
        return WeekendOnlyCalendar()
    raise MarketConventionError(f"Unknown holiday calendar {name!r}")


def _conventions_to_dict(conventions: Optional[Conventions]) -> Optional[Dict[str, Any]]:
    if conventions is None:
        return None
    holiday_calendar = conventions.holiday_calendar
    holiday_name = None
    if holiday_calendar is not None:
        holiday_name = type(holiday_calendar).__name__
    return {
        "day_count": conventions.day_count.value,
        "business_day": conventions.business_day.value,
        "compounding": conventions.compounding.value,
        "payment_frequency": conventions.payment_frequency,
        "settlement_days": conventions.settlement_days,
        "holiday_calendar": holiday_name,
        "end_of_month": conventions.end_of_month,
        "stub": conventions.stub,
    }


def _instrument_family(instrument_type: str) -> str:
    inst = str(instrument_type or "").upper()
    if inst in {"BOND", "UST"}:
        return "BOND"
    if inst in {"SWAP", "IRS"}:
        return "SWAP"
    if inst == "SWAPTION":
        return "SWAPTION"
    if inst in {"CAPLET", "CAP", "CAPFLOOR"}:
        return "CAPLET"
    if inst in {"FUT", "FUTURE", "FUTURES"}:
        return "FUTURE"
    return inst


def _coerce_mapping(trade: Any) -> Mapping[str, Any]:
    if isinstance(trade, Mapping):
        return trade
    if hasattr(trade, "to_dict"):
        candidate = trade.to_dict()
        if isinstance(candidate, Mapping):
            return candidate
    raise MarketConventionError(
        f"Market-convention resolution requires a mapping-like trade, got {type(trade).__name__}"
    )


USD_UST = MarketConventionTemplate(
    template_id="USD_UST",
    currency="USD",
    instrument_family="BOND",
    description="USD Treasury-style coupon bond conventions.",
    bond=ConventionSpec(
        day_count=DayCount.ACT_ACT,
        business_day=BusinessDayConvention.FOLLOWING,
        compounding=CompoundingConvention.SEMI_ANNUAL,
        payment_frequency=2,
        settlement_days=1,
        holiday_calendar="US_FED",
    ),
    tags=("treasury", "government", "usd"),
)

USD_VANILLA_SWAP = MarketConventionTemplate(
    template_id="USD_VANILLA_SWAP",
    currency="USD",
    instrument_family="SWAP",
    description="USD vanilla fixed-float swap conventions used by the sample library.",
    fixed_leg=ConventionSpec(
        day_count=DayCount.ACT_360,
        business_day=BusinessDayConvention.MODIFIED_FOLLOWING,
        compounding=CompoundingConvention.SEMI_ANNUAL,
        payment_frequency=2,
        settlement_days=2,
        holiday_calendar="US_FED",
    ),
    float_leg=ConventionSpec(
        day_count=DayCount.ACT_360,
        business_day=BusinessDayConvention.MODIFIED_FOLLOWING,
        compounding=CompoundingConvention.QUARTERLY,
        payment_frequency=4,
        settlement_days=2,
        holiday_calendar="US_FED",
    ),
    tags=("swap", "irs", "usd"),
)

USD_VANILLA_SWAPTION = MarketConventionTemplate(
    template_id="USD_VANILLA_SWAPTION",
    currency="USD",
    instrument_family="SWAPTION",
    description="USD swaption conventions aligned to the vanilla swap template.",
    fixed_leg=USD_VANILLA_SWAP.fixed_leg,
    float_leg=USD_VANILLA_SWAP.float_leg,
    swaption_fixed_frequency=2,
    swaption_float_frequency=4,
    tags=("swaption", "usd", "sabr"),
)

USD_CAPLET = MarketConventionTemplate(
    template_id="USD_CAPLET",
    currency="USD",
    instrument_family="CAPLET",
    description="USD caplet/floorlet conventions using money-market accrual and US holidays.",
    float_leg=ConventionSpec(
        day_count=DayCount.ACT_360,
        business_day=BusinessDayConvention.MODIFIED_FOLLOWING,
        compounding=CompoundingConvention.QUARTERLY,
        payment_frequency=4,
        settlement_days=2,
        holiday_calendar="US_FED",
    ),
    default_underlying_tenor="3M",
    tags=("caplet", "floorlet", "usd"),
)

USD_FUTURE = MarketConventionTemplate(
    template_id="USD_FUTURE",
    currency="USD",
    instrument_family="FUTURE",
    description="USD short-rate future template with 3M underlying tenor.",
    default_underlying_tenor="3M",
    tags=("future", "futures", "usd"),
)


STANDARD_MARKET_TEMPLATES: Dict[str, MarketConventionTemplate] = {
    template.template_id.upper(): template
    for template in (
        USD_UST,
        USD_VANILLA_SWAP,
        USD_VANILLA_SWAPTION,
        USD_CAPLET,
        USD_FUTURE,
    )
}

DEFAULT_TEMPLATE_BY_CURRENCY_AND_FAMILY: Dict[Tuple[str, str], str] = {
    ("USD", "BOND"): "USD_UST",
    ("USD", "SWAP"): "USD_VANILLA_SWAP",
    ("USD", "SWAPTION"): "USD_VANILLA_SWAPTION",
    ("USD", "CAPLET"): "USD_CAPLET",
    ("USD", "FUTURE"): "USD_FUTURE",
}


def available_market_conventions() -> Dict[str, MarketConventionTemplate]:
    """Return a copy of the registered standard templates."""
    return dict(STANDARD_MARKET_TEMPLATES)


def get_market_convention(template_id: str) -> MarketConventionTemplate:
    """Return a named market-convention template."""
    key = str(template_id or "").upper().strip()
    if key not in STANDARD_MARKET_TEMPLATES:
        raise MarketConventionError(
            f"Unknown market convention {template_id!r}. "
            f"Available templates: {sorted(STANDARD_MARKET_TEMPLATES)}"
        )
    return STANDARD_MARKET_TEMPLATES[key]


def resolve_market_convention_for_trade(
    trade: Any,
    curve_currency: Optional[str] = None,
) -> ResolvedMarketConvention:
    """
    Resolve market conventions for a trade.

    Resolution order:
    1. explicit ``market_convention`` or ``convention_template`` on the trade
    2. default template by explicit trade currency and instrument family
    3. default template by curve currency and instrument family
    4. fall back to legacy pricer defaults
    """
    mapping = _coerce_mapping(trade)
    instrument_type = str(mapping.get("instrument_type", "")).upper()
    family = _instrument_family(instrument_type)

    explicit_template_id = (
        mapping.get("market_convention")
        or mapping.get("convention_template")
        or mapping.get("pricing_template")
    )
    if explicit_template_id:
        template = get_market_convention(str(explicit_template_id))
        return template.resolve(
            instrument_type=instrument_type,
            currency=template.currency,
            source="explicit_template",
        )

    trade_currency_raw = mapping.get("currency")
    trade_currency = None
    if trade_currency_raw:
        trade_currency = str(trade_currency_raw).upper().strip()
    if trade_currency:
        template_id = DEFAULT_TEMPLATE_BY_CURRENCY_AND_FAMILY.get((trade_currency, family))
        if template_id is not None:
            template = get_market_convention(template_id)
            return template.resolve(
                instrument_type=instrument_type,
                currency=trade_currency,
                source="trade_currency_default",
            )

    curve_currency_norm = None
    if curve_currency:
        curve_currency_norm = str(curve_currency).upper().strip()
        template_id = DEFAULT_TEMPLATE_BY_CURRENCY_AND_FAMILY.get((curve_currency_norm, family))
        if template_id is not None:
            template = get_market_convention(template_id)
            return template.resolve(
                instrument_type=instrument_type,
                currency=curve_currency_norm,
                source="curve_currency_default",
            )

    return ResolvedMarketConvention(
        instrument_type=instrument_type,
        currency=trade_currency or curve_currency_norm,
        source="legacy_defaults",
    )


__all__ = [
    "ConventionSpec",
    "MarketConventionTemplate",
    "ResolvedMarketConvention",
    "MarketConventionError",
    "STANDARD_MARKET_TEMPLATES",
    "DEFAULT_TEMPLATE_BY_CURRENCY_AND_FAMILY",
    "available_market_conventions",
    "get_market_convention",
    "resolve_market_convention_for_trade",
]
