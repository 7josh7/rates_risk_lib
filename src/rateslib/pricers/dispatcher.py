"""
Unified trade-level pricing and risk dispatch.

Supports linear instruments (bond, swap, futures) and options (caplet, swaption)
using MarketState = (CurveState, SabrSurfaceState).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import date

from ..market_state import MarketState
from .bonds import BondPricer
from .swaps import SwapPricer
from .futures import FuturesPricer, FuturesContract
from ..options.caplet import CapletPricer
from ..options.swaption import SwaptionPricer
from ..options.sabr_risk import SabrOptionRisk
from ..vol.sabr_surface import make_bucket_key
from ..dates import DateUtils


@dataclass
class PricerOutput:
    """Container for pricing outputs to keep return type consistent."""

    instrument_type: str
    pv: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"instrument_type": self.instrument_type, "pv": self.pv, **self.details}


def price_trade(trade: Dict[str, Any], market_state: MarketState) -> PricerOutput:
    """
    Price a trade dict using the MarketState.

    Expected trade keys (subset used per instrument):
        instrument_type: BOND/UST, SWAP/IRS, FUT, SWAPTION, CAPLET
        notional: currency or contract units
        direction/pay_receive: direction flags
        Dates/tenors/strike/vol as appropriate per instrument
    """
    inst = str(trade.get("instrument_type", "")).upper()
    curve_state = market_state.curve

    if inst in {"BOND", "UST"}:
        pricer = BondPricer(curve_state.discount_curve)
        settlement = trade["settlement"]
        maturity = trade["maturity"]
        coupon = float(trade.get("coupon", 0.0))
        frequency = int(trade.get("frequency", 2))
        face_value = float(trade.get("face_value", 100.0))
        notional = float(trade.get("notional", face_value))

        dirty, clean, accrued = pricer.price(
            settlement=settlement,
            maturity=maturity,
            coupon_rate=coupon,
            face_value=face_value,
            frequency=frequency,
        )
        pv = dirty / face_value * notional
        return PricerOutput(
            instrument_type=inst,
            pv=pv,
            details={"dirty": dirty, "clean": clean, "accrued": accrued},
        )

    if inst in {"SWAP", "IRS"}:
        pricer = SwapPricer(curve_state.discount_curve, curve_state.projection_curve)
        pv = pricer.present_value(
            effective=trade["effective"],
            maturity=trade["maturity"],
            notional=float(trade["notional"]),
            fixed_rate=float(trade["fixed_rate"]),
            pay_receive=str(trade.get("pay_receive", "PAY")).upper(),
        )
        dv01 = pricer.dv01(
            effective=trade["effective"],
            maturity=trade["maturity"],
            notional=float(trade["notional"]),
            fixed_rate=float(trade["fixed_rate"]),
            pay_receive=str(trade.get("pay_receive", "PAY")).upper(),
        )
        return PricerOutput(
            instrument_type=inst,
            pv=pv,
            details={"dv01": dv01},
        )

    if inst in {"FUT", "FUTURE", "FUTURES"}:
        contract = FuturesContract(
            contract_code=trade.get("contract_code", "FUT"),
            expiry=trade["expiry"],
            contract_size=float(trade.get("contract_size", 1_000_000)),
            tick_size=float(trade.get("tick_size", 0.0025)),
            underlying_tenor=str(trade.get("underlying_tenor", "3M")),
        )
        pricer = FuturesPricer(curve_state.discount_curve)
        price = pricer.theoretical_price(contract)
        dv01 = pricer.dv01(contract, int(trade.get("num_contracts", 1)))
        return PricerOutput(
            instrument_type=inst,
            pv=0.0,  # Futures PV is margin-based; report theoretical price instead
            details={"theoretical_price": price, "dv01": dv01},
        )

    if inst == "SWAPTION":
        expiry_tenor = trade["expiry_tenor"]
        swap_tenor = trade["swap_tenor"]
        strike_raw = trade.get("strike")
        payer_receiver = str(trade.get("payer_receiver", "PAYER")).upper()
        notional = float(trade.get("notional", 1.0))
        vol_type = str(trade.get("vol_type", "NORMAL")).upper()

        pricer = SwaptionPricer(curve_state.discount_curve, curve_state.projection_curve)
        forward, annuity = pricer.forward_swap_rate(
            expiry=DateUtils.tenor_to_years(expiry_tenor),
            tenor=DateUtils.tenor_to_years(swap_tenor),
        )
        # Determine strike, defaulting to ATM if missing/invalid
        if strike_raw is None or (isinstance(strike_raw, str) and strike_raw.upper() == "ATM"):
            strike = forward
        else:
            try:
                strike = float(strike_raw)
            except Exception:
                strike = forward
        if strike <= 0:
            strike = max(forward, 1e-6)

        sabr_params = market_state.get_sabr_params(expiry_tenor, swap_tenor)

        if sabr_params:
            result = pricer.price_with_sabr(
                expiry_tenor=expiry_tenor,
                swap_tenor=swap_tenor,
                K=strike,
                sabr_params=sabr_params.to_sabr_params(),
                vol_type=vol_type,
                payer_receiver=payer_receiver,
                notional=notional,
            )
            pv = result.price
            details = {
                "forward": result.forward_swap_rate,
                "annuity": result.annuity,
                "implied_vol": result.implied_vol,
                "vol_type": vol_type,
                "source_bucket": make_bucket_key(expiry_tenor, swap_tenor),
                "strike": strike,
            }
            return PricerOutput(inst, pv, details)

        # Fall back to flat vol pricing
        vol = float(trade.get("vol", 0.0))
        pv = pricer.price(
            S=forward,
            K=strike,
            T=DateUtils.tenor_to_years(expiry_tenor),
            annuity=annuity,
            vol=vol,
            vol_type=vol_type,
            payer_receiver=payer_receiver,
            notional=notional,
            shift=float(trade.get("shift", 0.0)),
        )
        return PricerOutput(
            instrument_type=inst,
            pv=pv,
            details={"forward": forward, "annuity": annuity, "implied_vol": vol, "vol_type": vol_type, "strike": strike},
        )

    if inst in {"CAPLET", "CAP", "CAPFLOOR"}:
        start_date = trade["start_date"]
        end_date = trade["end_date"]
        strike_raw = trade.get("strike")
        notional = float(trade.get("notional", 1.0))
        vol_type = str(trade.get("vol_type", "NORMAL")).upper()

        pricer = CapletPricer(curve_state.discount_curve, curve_state.projection_curve)
        sabr_params = market_state.get_sabr_params(trade.get("expiry_tenor", ""), trade.get("index_tenor", ""), allow_fallback=True)

        if sabr_params:
            # Compute forward to determine strike
            from ..conventions import year_fraction
            anchor = curve_state.discount_curve.anchor_date
            day_count = curve_state.discount_curve.day_count
            T_start = year_fraction(anchor, start_date, day_count)
            T_end = year_fraction(anchor, end_date, day_count)
            F = pricer.forward_rate(T_start, T_end)
            
            # Resolve strike
            if strike_raw is None or (isinstance(strike_raw, str) and strike_raw.upper() == "ATM"):
                K = F
            else:
                try:
                    K = float(strike_raw)
                except Exception:
                    K = F
            if K <= 0:
                K = max(F, 1e-6)
            
            result = pricer.price_with_sabr(
                start_date=start_date,
                end_date=end_date,
                K=K,
                sabr_params=sabr_params.to_sabr_params(),
                vol_type=vol_type,
                notional=notional,
                is_cap=trade.get("is_cap", True),
            )
            return PricerOutput(
                instrument_type=inst,
                pv=result.price,
                details={
                    "forward": result.forward,
                    "discount_factor": result.discount_factor,
                    "implied_vol": result.implied_vol,
                    "vol_type": vol_type,
                },
            )

        expiry_tenor = trade.get("expiry_tenor", "0D")
        index_tenor = trade.get("index_tenor", "3M")
        try:
            expiry_years = DateUtils.tenor_to_years(expiry_tenor)
        except Exception:
            expiry_years = 0.0
        try:
            index_years = DateUtils.tenor_to_years(index_tenor)
        except Exception:
            index_years = 0.25

        delta_t = trade.get("delta_t", index_years)
        vol = float(trade.get("vol", 0.0))
        F = pricer.forward_rate(start=expiry_years, end=expiry_years + delta_t)
        df = curve_state.discount_curve.discount_factor(expiry_years + delta_t)
        if strike_raw is None or (isinstance(strike_raw, str) and strike_raw.upper() == "ATM"):
            strike = F
        else:
            try:
                strike = float(strike_raw)
            except Exception:
                strike = F
        if strike <= 0:
            strike = max(F, 1e-6)
        pv = pricer.price(
            F=F,
            K=strike,
            T=expiry_years,
            df=df,
            vol=vol,
            vol_type=vol_type,
            notional=notional,
            delta_t=delta_t or 0.25,
            is_cap=trade.get("is_cap", True),
            shift=float(trade.get("shift", 0.0)),
        )
        return PricerOutput(
            instrument_type=inst,
            pv=pv,
            details={"forward": F, "discount_factor": df, "implied_vol": vol, "vol_type": vol_type},
        )

    raise ValueError(f"Unsupported instrument_type: {inst}")


def risk_trade(trade: Dict[str, Any], market_state: MarketState, method: str = "bump") -> Dict[str, Any]:
    """
    Compute risk for a trade using SABR-aware Greeks where applicable.

    Currently returns SABR parameter sensitivities for options and DV01 for swaps/bonds.
    """
    inst = str(trade.get("instrument_type", "")).upper()
    curve_state = market_state.curve

    if inst in {"SWAP", "IRS"}:
        pricer = SwapPricer(curve_state.discount_curve, curve_state.projection_curve)
        dv01 = pricer.dv01(
            effective=trade["effective"],
            maturity=trade["maturity"],
            notional=float(trade["notional"]),
            fixed_rate=float(trade["fixed_rate"]),
            pay_receive=str(trade.get("pay_receive", "PAY")).upper(),
        )
        return {"dv01": dv01}

    if inst in {"BOND", "UST"}:
        pricer = BondPricer(curve_state.discount_curve)
        dv01 = pricer.compute_dv01(
            settlement=trade["settlement"],
            maturity=trade["maturity"],
            coupon_rate=float(trade.get("coupon", 0.0)),
            face_value=float(trade.get("face_value", 100.0)),
            frequency=int(trade.get("frequency", 2)),
            notional=float(trade.get("notional", 1_000_000)),
        )
        return {"dv01": dv01}

    if inst == "SWAPTION":
        expiry_tenor = trade["expiry_tenor"]
        swap_tenor = trade["swap_tenor"]
        strike_raw = trade.get("strike")
        notional = float(trade.get("notional", 1.0))
        vol_type = str(trade.get("vol_type", "NORMAL")).upper()

        sabr_params = market_state.get_sabr_params(expiry_tenor, swap_tenor, allow_fallback=True)
        if sabr_params is None:
            return {}

        pricer = SwaptionPricer(curve_state.discount_curve, curve_state.projection_curve)
        forward, annuity = pricer.forward_swap_rate(
            expiry=DateUtils.tenor_to_years(expiry_tenor),
            tenor=DateUtils.tenor_to_years(swap_tenor),
        )
        # Resolve strike with ATM default
        if strike_raw is None or (isinstance(strike_raw, str) and str(strike_raw).upper() == "ATM"):
            strike = forward
        else:
            try:
                strike = float(strike_raw)
            except Exception:
                strike = forward
        if strike <= 0:
            strike = max(forward, 1e-6)
        risk_engine = SabrOptionRisk(vol_type=vol_type)
        # Compute SABR vol for greeks aggregation
        from ..vol.sabr import SabrModel
        T = DateUtils.tenor_to_years(expiry_tenor)
        model = SabrModel()
        if vol_type == "NORMAL":
            implied_vol = model.implied_vol_normal(forward, strike, T, sabr_params.to_sabr_params())
        else:
            implied_vol = model.implied_vol_black(forward, strike, T, sabr_params.to_sabr_params())
        sens = risk_engine.parameter_sensitivities(
            F=forward,
            K=strike,
            T=T,
            sabr_params=sabr_params.to_sabr_params(),
            annuity=annuity,
            is_call=True,
            notional=notional,
        )
        greeks = pricer.greeks(
            S=forward,
            K=strike,
            T=T,
            annuity=annuity,
            vol=implied_vol,
            vol_type=vol_type,
            payer_receiver="PAYER",
            notional=notional,
        )
        return {"sabr_sensitivities": sens, "forward": forward, "annuity": annuity, "greeks": greeks}

    if inst in {"CAPLET", "CAP", "CAPFLOOR"}:
        expiry_tenor = trade.get("expiry_tenor", "0D")
        index_tenor = trade.get("index_tenor", "3M")
        strike_raw = trade.get("strike")
        notional = float(trade.get("notional", 1.0))
        vol_type = str(trade.get("vol_type", "NORMAL")).upper()

        sabr_params = market_state.get_sabr_params(expiry_tenor, index_tenor, allow_fallback=True)
        if sabr_params is None:
            return {}

        pricer = CapletPricer(curve_state.discount_curve, curve_state.projection_curve)
        try:
            start = DateUtils.tenor_to_years(expiry_tenor)
        except Exception:
            start = 0.0
        try:
            index_years = DateUtils.tenor_to_years(index_tenor)
        except Exception:
            index_years = 0.25
        delta_t = trade.get("delta_t", index_years)
        end = start + delta_t
        forward = pricer.forward_rate(start, end) if end > start else 0.0
        annuity = curve_state.discount_curve.discount_factor(end) if end >= 0 else 1.0
        if strike_raw is None or (isinstance(strike_raw, str) and strike_raw.upper() == "ATM"):
            strike = forward
        else:
            try:
                strike = float(strike_raw)
            except Exception:
                strike = forward
        if strike <= 0:
            strike = max(forward, 1e-6)
        risk_engine = SabrOptionRisk(vol_type=vol_type)
        sens = risk_engine.parameter_sensitivities(
            F=forward,
            K=strike,
            T=start,
            sabr_params=sabr_params.to_sabr_params(),
            annuity=annuity,
            is_call=trade.get("is_cap", True),
            notional=notional,
        )
        return {"sabr_sensitivities": sens, "forward": forward, "annuity": annuity}

    return {}


__all__ = ["price_trade", "risk_trade", "PricerOutput"]
