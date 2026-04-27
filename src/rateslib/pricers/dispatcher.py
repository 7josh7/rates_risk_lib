"""
Unified trade-level pricing and risk dispatch.

Supports linear instruments (bond, swap, futures) and options (caplet,
swaption) using MarketState = (CurveState, SabrSurfaceState).

The dispatcher accepts either legacy trade dictionaries or the typed trade
objects defined in ``rateslib.domain``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..dates import DateUtils
from ..domain import CapletTrade, SwaptionTrade, TradeLike, normalize_trade
from ..market_conventions import resolve_market_convention_for_trade
from ..market_state import MarketState
from ..options.caplet import CapletPricer
from ..options.sabr_risk import SabrOptionRisk
from ..options.swaption import SwaptionPricer
from ..vol.sabr_surface import SabrLookupResult, make_bucket_key
from .bonds import BondPricer
from .futures import FuturesContract, FuturesPricer
from .swaps import SwapPricer


@dataclass
class PricerOutput:
    """Container for pricing outputs to keep return type consistent."""

    instrument_type: str
    pv: float
    details: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)
    trade_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "instrument_type": self.instrument_type,
            "pv": self.pv,
            **self.details,
        }
        if self.warnings:
            result["warnings"] = list(self.warnings)
        if self.audit:
            result["audit"] = dict(self.audit)
        if self.trade_id is not None:
            result["trade_id"] = self.trade_id
        return result


def _build_audit(
    trade,
    market_state: MarketState,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not market_state.pricing_policy.record_audit:
        return {}

    audit = {
        "trade_id": trade.trade_id,
        "position_id": trade.position_id,
        "market_asof": market_state.asof,
        "pricing_policy": market_state.pricing_policy.to_dict(),
    }
    if extra:
        audit.update(extra)
    return audit


def _build_output(
    trade,
    market_state: MarketState,
    pv: float,
    details: Dict[str, Any],
    warnings: Optional[List[str]] = None,
    audit_extra: Optional[Dict[str, Any]] = None,
) -> PricerOutput:
    return PricerOutput(
        instrument_type=str(trade.instrument_type).upper(),
        pv=pv,
        details=details,
        warnings=list(warnings or []),
        audit=_build_audit(trade, market_state, extra=audit_extra),
        trade_id=trade.trade_id or trade.position_id,
    )


def _merge_audit_extras(*extras: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for extra in extras:
        if extra:
            merged.update(extra)
    return merged


def _resolve_strike(strike_raw: Any, forward: float) -> float:
    if strike_raw is None:
        strike = forward
    elif isinstance(strike_raw, str) and strike_raw.upper() == "ATM":
        strike = forward
    else:
        try:
            strike = float(strike_raw)
        except Exception:
            strike = forward
    if strike <= 0:
        return max(forward, 1e-6)
    return strike


def _lookup_to_audit(lookup: Optional[SabrLookupResult]) -> Dict[str, Any]:
    if lookup is None:
        return {"sabr_lookup": None}
    return {
        "sabr_lookup": {
            "requested_bucket": lookup.requested_bucket,
            "used_bucket": lookup.used_bucket,
            "used_fallback": lookup.used_fallback,
            "reason": lookup.reason,
        }
    }


def _lookup_warnings(
    trade,
    lookup: Optional[SabrLookupResult],
    market_state: MarketState,
) -> List[str]:
    if lookup is None or not lookup.used_fallback:
        return []
    if market_state.pricing_policy.sabr_bucket_fallback != "warn":
        return []
    return [
        (
            f"{trade.instrument_type} used SABR fallback bucket "
            f"{lookup.used_bucket} for requested {lookup.requested_bucket}."
        )
    ]


def _resolve_option_vol(
    trade,
    market_state: MarketState,
    fallback_label: str,
) -> Tuple[float, List[str]]:
    warnings: List[str] = []
    explicit_vol = trade.get("vol")
    if explicit_vol is not None:
        return float(explicit_vol), warnings

    if not market_state.pricing_policy.allow_zero_option_vol:
        raise ValueError(
            f"No SABR parameters or explicit volatility available for {fallback_label}."
        )

    warnings.append(
        f"{fallback_label} used zero-volatility fallback because no model or explicit vol was available."
    )
    return 0.0, warnings


def price_trade(trade: TradeLike, market_state: MarketState) -> PricerOutput:
    """
    Price a trade using the supplied market state.

    Accepted payloads:
        - legacy dict-style trade objects
        - typed trades from ``rateslib.domain``
    """
    normalized_trade = normalize_trade(trade)
    trade_data = normalized_trade.to_dict()
    inst = str(normalized_trade.instrument_type).upper()
    curve_state = market_state.curve
    resolved_market = resolve_market_convention_for_trade(
        trade_data,
        curve_currency=getattr(curve_state.discount_curve, "currency", None),
    )
    market_audit = {"market_convention": resolved_market.to_dict()}

    if inst in {"BOND", "UST"}:
        pricer = BondPricer(
            curve_state.discount_curve,
            conventions=resolved_market.bond_conventions,
        )
        settlement = trade_data["settlement"]
        maturity = trade_data["maturity"]
        coupon = float(trade_data.get("coupon", 0.0))
        frequency = int(trade_data.get("frequency", 2))
        face_value = float(trade_data.get("face_value", 100.0))
        notional = float(trade_data.get("notional", face_value))

        dirty, clean, accrued = pricer.price(
            settlement=settlement,
            maturity=maturity,
            coupon_rate=coupon,
            face_value=face_value,
            frequency=frequency,
        )
        pv = dirty / face_value * notional
        return _build_output(
            normalized_trade,
            market_state,
            pv,
            {"dirty": dirty, "clean": clean, "accrued": accrued},
            audit_extra=market_audit,
        )

    if inst in {"SWAP", "IRS"}:
        pricer = SwapPricer(
            curve_state.discount_curve,
            curve_state.projection_curve,
            fixed_conventions=resolved_market.fixed_leg_conventions,
            float_conventions=resolved_market.float_leg_conventions,
        )
        pv = pricer.present_value(
            effective=trade_data["effective"],
            maturity=trade_data["maturity"],
            notional=float(trade_data["notional"]),
            fixed_rate=float(trade_data["fixed_rate"]),
            pay_receive=str(trade_data.get("pay_receive", "PAY")).upper(),
        )
        dv01 = pricer.dv01(
            effective=trade_data["effective"],
            maturity=trade_data["maturity"],
            notional=float(trade_data["notional"]),
            fixed_rate=float(trade_data["fixed_rate"]),
            pay_receive=str(trade_data.get("pay_receive", "PAY")).upper(),
        )
        return _build_output(
            normalized_trade,
            market_state,
            pv,
            {"dv01": dv01},
            audit_extra=market_audit,
        )

    if inst in {"FUT", "FUTURE", "FUTURES"}:
        contract = FuturesContract(
            contract_code=trade_data.get("contract_code", "FUT"),
            expiry=trade_data["expiry"],
            contract_size=float(trade_data.get("contract_size", 1_000_000)),
            tick_size=float(trade_data.get("tick_size", 0.0025)),
            tick_value=float(trade_data.get("tick_value", 6.25)),
            underlying_tenor=str(trade_data.get("underlying_tenor", "3M")),
        )

        pricer = FuturesPricer(curve_state.discount_curve)
        theoretical_price = pricer.theoretical_price(contract)
        num_contracts = int(trade_data.get("num_contracts", 1))
        dv01 = pricer.dv01(contract, num_contracts)

        trade_price = trade_data.get("trade_price")
        tick_size = contract.tick_size
        tick_value = float(trade_data.get("tick_value", 6.25))

        if trade_price is not None:
            _, pv, _ = pricer.position_pv(contract, num_contracts, float(trade_price))
            warnings = []
        else:
            pv = 0.0
            warnings = [
                "Futures trade has no trade_price; PV is reported as zero mark-to-market."
            ]

        return _build_output(
            normalized_trade,
            market_state,
            pv,
            {
                "theoretical_price": theoretical_price,
                "implied_rate": pricer.implied_rate(contract.expiry, contract.underlying_tenor),
                "dv01": dv01,
                "num_contracts": num_contracts,
                "trade_price": trade_price,
            },
            warnings=warnings,
            audit_extra=market_audit,
        )

    if inst == "SWAPTION":
        trade_obj = normalized_trade
        if not isinstance(trade_obj, SwaptionTrade):
            raise TypeError("Normalized SWAPTION trade has unexpected type")

        pricer = SwaptionPricer(
            curve_state.discount_curve,
            curve_state.projection_curve,
            fixed_freq=resolved_market.swaption_fixed_frequency or 2,
            float_freq=resolved_market.swaption_float_frequency or 4,
        )
        forward, annuity = pricer.forward_swap_rate(
            expiry=DateUtils.tenor_to_years(trade_obj.expiry_tenor),
            tenor=DateUtils.tenor_to_years(trade_obj.swap_tenor),
        )
        strike = _resolve_strike(trade_obj.strike, forward)
        lookup = market_state.resolve_sabr_lookup(
            trade_obj.expiry_tenor,
            trade_obj.swap_tenor,
            allow_fallback=True,
        )
        warnings = _lookup_warnings(trade_obj, lookup, market_state)
        audit_extra = _lookup_to_audit(lookup)

        if lookup is not None and lookup.params is not None:
            result = pricer.price_with_sabr(
                expiry_tenor=trade_obj.expiry_tenor,
                swap_tenor=trade_obj.swap_tenor,
                K=strike,
                sabr_params=lookup.params.to_sabr_params(),
                vol_type=trade_obj.vol_type,
                payer_receiver=trade_obj.payer_receiver,
                notional=trade_obj.notional,
            )
            return _build_output(
                trade_obj,
                market_state,
                result.price,
                {
                    "forward": result.forward_swap_rate,
                    "annuity": result.annuity,
                    "implied_vol": result.implied_vol,
                    "vol_type": trade_obj.vol_type,
                    "source_bucket": lookup.used_bucket or make_bucket_key(trade_obj.expiry_tenor, trade_obj.swap_tenor),
                    "strike": strike,
                },
                warnings=warnings,
                audit_extra=_merge_audit_extras(market_audit, audit_extra),
            )

        vol, vol_warnings = _resolve_option_vol(
            trade_obj,
            market_state,
            fallback_label="SWAPTION",
        )
        warnings.extend(vol_warnings)
        pv = pricer.price(
            S=forward,
            K=strike,
            T=DateUtils.tenor_to_years(trade_obj.expiry_tenor),
            annuity=annuity,
            vol=vol,
            vol_type=trade_obj.vol_type,
            payer_receiver=trade_obj.payer_receiver,
            notional=trade_obj.notional,
            shift=trade_obj.shift,
        )
        return _build_output(
            trade_obj,
            market_state,
            pv,
            {
                "forward": forward,
                "annuity": annuity,
                "implied_vol": vol,
                "vol_type": trade_obj.vol_type,
                "strike": strike,
            },
            warnings=warnings,
            audit_extra=_merge_audit_extras(market_audit, audit_extra),
        )

    if inst in {"CAPLET", "CAP", "CAPFLOOR"}:
        trade_obj = normalized_trade
        if not isinstance(trade_obj, CapletTrade):
            raise TypeError("Normalized CAPLET trade has unexpected type")

        pricer = CapletPricer(curve_state.discount_curve, curve_state.projection_curve)
        lookup = market_state.resolve_sabr_lookup(
            trade_obj.expiry_tenor,
            trade_obj.index_tenor,
            allow_fallback=True,
        )
        warnings = _lookup_warnings(trade_obj, lookup, market_state)
        audit_extra = _lookup_to_audit(lookup)

        if lookup is not None and lookup.params is not None:
            from ..conventions import year_fraction

            anchor = curve_state.discount_curve.anchor_date
            day_count = curve_state.discount_curve.day_count
            t_start = year_fraction(anchor, trade_obj.start_date, day_count)
            t_end = year_fraction(anchor, trade_obj.end_date, day_count)
            forward = pricer.forward_rate(t_start, t_end)
            strike = _resolve_strike(trade_obj.strike, forward)

            result = pricer.price_with_sabr(
                start_date=trade_obj.start_date,
                end_date=trade_obj.end_date,
                K=strike,
                sabr_params=lookup.params.to_sabr_params(),
                vol_type=trade_obj.vol_type,
                notional=trade_obj.notional,
                is_cap=trade_obj.is_cap,
            )
            return _build_output(
                trade_obj,
                market_state,
                result.price,
                {
                    "forward": result.forward,
                    "discount_factor": result.discount_factor,
                    "implied_vol": result.implied_vol,
                    "vol_type": trade_obj.vol_type,
                    "strike": strike,
                },
                warnings=warnings,
                audit_extra=_merge_audit_extras(market_audit, audit_extra),
            )

        try:
            expiry_years = DateUtils.tenor_to_years(trade_obj.expiry_tenor)
        except Exception:
            expiry_years = 0.0
        try:
            index_years = DateUtils.tenor_to_years(trade_obj.index_tenor)
        except Exception:
            index_years = 0.25

        delta_t = trade_obj.delta_t if trade_obj.delta_t is not None else index_years
        vol, vol_warnings = _resolve_option_vol(
            trade_obj,
            market_state,
            fallback_label="CAPLET",
        )
        warnings.extend(vol_warnings)
        forward = pricer.forward_rate(start=expiry_years, end=expiry_years + delta_t)
        discount_factor = curve_state.discount_curve.discount_factor(expiry_years + delta_t)
        strike = _resolve_strike(trade_obj.strike, forward)
        pv = pricer.price(
            F=forward,
            K=strike,
            T=expiry_years,
            df=discount_factor,
            vol=vol,
            vol_type=trade_obj.vol_type,
            notional=trade_obj.notional,
            delta_t=delta_t or 0.25,
            is_cap=trade_obj.is_cap,
            shift=trade_obj.shift,
        )
        return _build_output(
            trade_obj,
            market_state,
            pv,
            {
                "forward": forward,
                "discount_factor": discount_factor,
                "implied_vol": vol,
                "vol_type": trade_obj.vol_type,
                "strike": strike,
            },
            warnings=warnings,
            audit_extra=_merge_audit_extras(market_audit, audit_extra),
        )

    raise ValueError(f"Unsupported instrument_type: {inst}")


def risk_trade(trade: TradeLike, market_state: MarketState, method: str = "bump") -> Dict[str, Any]:
    """
    Compute risk for a trade using SABR-aware Greeks where applicable.

    Currently returns SABR parameter sensitivities for options and DV01 for
    swaps/bonds.
    """
    normalized_trade = normalize_trade(trade)
    trade_data = normalized_trade.to_dict()
    inst = str(normalized_trade.instrument_type).upper()
    curve_state = market_state.curve
    resolved_market = resolve_market_convention_for_trade(
        trade_data,
        curve_currency=getattr(curve_state.discount_curve, "currency", None),
    )

    if inst in {"SWAP", "IRS"}:
        pricer = SwapPricer(
            curve_state.discount_curve,
            curve_state.projection_curve,
            fixed_conventions=resolved_market.fixed_leg_conventions,
            float_conventions=resolved_market.float_leg_conventions,
        )
        dv01 = pricer.dv01(
            effective=trade_data["effective"],
            maturity=trade_data["maturity"],
            notional=float(trade_data["notional"]),
            fixed_rate=float(trade_data["fixed_rate"]),
            pay_receive=str(trade_data.get("pay_receive", "PAY")).upper(),
        )
        return {"dv01": dv01}

    if inst in {"BOND", "UST"}:
        pricer = BondPricer(
            curve_state.discount_curve,
            conventions=resolved_market.bond_conventions,
        )
        dv01 = pricer.compute_dv01(
            settlement=trade_data["settlement"],
            maturity=trade_data["maturity"],
            coupon_rate=float(trade_data.get("coupon", 0.0)),
            face_value=float(trade_data.get("face_value", 100.0)),
            frequency=int(trade_data.get("frequency", 2)),
            notional=float(trade_data.get("notional", 1_000_000)),
        )
        return {"dv01": dv01}

    if inst == "SWAPTION":
        trade_obj = normalized_trade
        if not isinstance(trade_obj, SwaptionTrade):
            raise TypeError("Normalized SWAPTION trade has unexpected type")

        lookup = market_state.resolve_sabr_lookup(
            trade_obj.expiry_tenor,
            trade_obj.swap_tenor,
            allow_fallback=True,
        )
        if lookup is None or lookup.params is None:
            if market_state.pricing_policy.sabr_bucket_fallback == "error":
                raise ValueError(
                    f"No SABR parameters available for {trade_obj.expiry_tenor} x {trade_obj.swap_tenor}."
                )
            return {}

        pricer = SwaptionPricer(
            curve_state.discount_curve,
            curve_state.projection_curve,
            fixed_freq=resolved_market.swaption_fixed_frequency or 2,
            float_freq=resolved_market.swaption_float_frequency or 4,
        )
        forward, annuity = pricer.forward_swap_rate(
            expiry=DateUtils.tenor_to_years(trade_obj.expiry_tenor),
            tenor=DateUtils.tenor_to_years(trade_obj.swap_tenor),
        )
        strike = _resolve_strike(trade_obj.strike, forward)
        risk_engine = SabrOptionRisk(vol_type=trade_obj.vol_type)

        from ..vol.sabr import SabrModel

        t_expiry = DateUtils.tenor_to_years(trade_obj.expiry_tenor)
        model = SabrModel()
        if trade_obj.vol_type == "NORMAL":
            implied_vol = model.implied_vol_normal(
                forward,
                strike,
                t_expiry,
                lookup.params.to_sabr_params(),
            )
        else:
            implied_vol = model.implied_vol_black(
                forward,
                strike,
                t_expiry,
                lookup.params.to_sabr_params(),
            )
        sens = risk_engine.parameter_sensitivities(
            F=forward,
            K=strike,
            T=t_expiry,
            sabr_params=lookup.params.to_sabr_params(),
            annuity=annuity,
            is_call=(trade_obj.payer_receiver == "PAYER"),
            notional=trade_obj.notional,
        )
        greeks = pricer.greeks(
            S=forward,
            K=strike,
            T=t_expiry,
            annuity=annuity,
            vol=implied_vol,
            vol_type=trade_obj.vol_type,
            payer_receiver=trade_obj.payer_receiver,
            notional=trade_obj.notional,
        )
        return {"sabr_sensitivities": sens, "forward": forward, "annuity": annuity, "greeks": greeks}

    if inst in {"CAPLET", "CAP", "CAPFLOOR"}:
        trade_obj = normalized_trade
        if not isinstance(trade_obj, CapletTrade):
            raise TypeError("Normalized CAPLET trade has unexpected type")

        lookup = market_state.resolve_sabr_lookup(
            trade_obj.expiry_tenor,
            trade_obj.index_tenor,
            allow_fallback=True,
        )
        if lookup is None or lookup.params is None:
            if market_state.pricing_policy.sabr_bucket_fallback == "error":
                raise ValueError(
                    f"No SABR parameters available for {trade_obj.expiry_tenor} x {trade_obj.index_tenor}."
                )
            return {}

        pricer = CapletPricer(curve_state.discount_curve, curve_state.projection_curve)
        try:
            start = DateUtils.tenor_to_years(trade_obj.expiry_tenor)
        except Exception:
            start = 0.0
        try:
            index_years = DateUtils.tenor_to_years(trade_obj.index_tenor)
        except Exception:
            index_years = 0.25
        delta_t = trade_obj.delta_t if trade_obj.delta_t is not None else index_years
        end = start + delta_t
        forward = pricer.forward_rate(start, end) if end > start else 0.0
        annuity = curve_state.discount_curve.discount_factor(end) if end >= 0 else 1.0
        strike = _resolve_strike(trade_obj.strike, forward)
        risk_engine = SabrOptionRisk(vol_type=trade_obj.vol_type)

        from ..vol.sabr import SabrModel

        model = SabrModel()
        if trade_obj.vol_type == "NORMAL":
            implied_vol = model.implied_vol_normal(
                forward,
                strike,
                start,
                lookup.params.to_sabr_params(),
            )
        else:
            implied_vol = model.implied_vol_black(
                forward,
                strike,
                start,
                lookup.params.to_sabr_params(),
            )
        sens = risk_engine.parameter_sensitivities(
            F=forward,
            K=strike,
            T=start,
            sabr_params=lookup.params.to_sabr_params(),
            annuity=annuity,
            is_call=trade_obj.is_cap,
            notional=trade_obj.notional,
        )
        greeks = pricer.greeks(
            F=forward,
            K=strike,
            T=start,
            df=annuity,
            vol=implied_vol,
            vol_type=trade_obj.vol_type,
            notional=trade_obj.notional,
            delta_t=delta_t or 0.25,
            is_cap=trade_obj.is_cap,
        )
        return {"sabr_sensitivities": sens, "forward": forward, "annuity": annuity, "greeks": greeks}

    return {}


__all__ = ["price_trade", "risk_trade", "PricerOutput"]
