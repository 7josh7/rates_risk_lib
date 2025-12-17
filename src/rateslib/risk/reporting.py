"""
Portfolio-level risk aggregation helpers used by reporting/UI layers.

Keeps computation out of the dashboard so the UI only orchestrates.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from ..pricers.dispatcher import price_trade, risk_trade
from ..market_state import MarketState, CurveState
from ..risk.bumping import BumpEngine
from ..dates import DateUtils


# Default key-rate tenors for bucketing
DEFAULT_KEYRATE_TENORS = ["2Y", "5Y", "10Y", "30Y"]
EXTENDED_KEYRATE_TENORS = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]


def _parse_date(val) -> Optional[date]:
    try:
        if isinstance(val, date):
            return val
        return pd.to_datetime(val).date()
    except Exception:
        return None


@dataclass
class CurveRiskMetrics:
    """
    Result of portfolio curve risk computation.
    
    Attributes:
        total_dv01: Portfolio-level DV01 ($ per 1bp parallel shift)
        keyrate_dv01: Dict of {tenor: dv01} for key-rate ladder
        worst_keyrate_dv01: Largest absolute key-rate DV01
        base_pv: Base portfolio present value
        tenors_used: List of tenors in the key-rate ladder
        instrument_coverage: Number of instruments successfully priced
        total_instruments: Total number of instruments in portfolio
        excluded_types: List of instrument types excluded (e.g., missing data)
    """
    total_dv01: float
    keyrate_dv01: Dict[str, float]
    worst_keyrate_dv01: float
    base_pv: float
    tenors_used: List[str]
    instrument_coverage: int
    total_instruments: int
    excluded_types: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_dv01": self.total_dv01,
            "keyrate_dv01": self.keyrate_dv01,
            "worst_keyrate_dv01": self.worst_keyrate_dv01,
            "base_pv": self.base_pv,
            "tenors_used": self.tenors_used,
            "instrument_coverage": self.instrument_coverage,
            "total_instruments": self.total_instruments,
            "excluded_types": self.excluded_types,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Return key-rate ladder as DataFrame for display."""
        return pd.DataFrame({
            "Tenor": self.tenors_used,
            "DV01": [self.keyrate_dv01.get(t, 0.0) for t in self.tenors_used]
        })


def _build_trade_from_position(pos: pd.Series, valuation_date: date) -> Optional[Dict[str, Any]]:
    """
    Build a trade dict from a position row.
    
    Returns None if the position cannot be converted.
    """
    inst = str(pos.get("instrument_type", "")).upper()
    notional = float(abs(pos.get("notional", 0.0)))
    direction = str(pos.get("direction", "")).upper()
    sign = -1.0 if direction in {"SHORT", "PAY_FIXED", "PAY"} else 1.0
    
    if inst in {"BOND", "UST"}:
        maturity = _parse_date(pos.get("maturity_date"))
        if not maturity:
            return None
        return {
            "instrument_type": inst,
            "settlement": valuation_date,
            "maturity": maturity,
            "coupon": float(pos.get("coupon", 0.0)),
            "notional": notional * sign,
            "frequency": int(pos.get("frequency", 2)),
        }
    
    if inst in {"SWAP", "IRS"}:
        maturity = _parse_date(pos.get("maturity_date"))
        if not maturity:
            return None
        pay_receive = "PAY" if "PAY" in direction else "RECEIVE"
        return {
            "instrument_type": "SWAP",
            "effective": valuation_date,
            "maturity": maturity,
            "notional": notional,
            "fixed_rate": float(pos.get("coupon", 0.0)),
            "pay_receive": pay_receive,
        }
    
    if inst in {"SWAPTION", "CAPLET", "CAP", "CAPFLOOR"}:
        maturity = _parse_date(pos.get("maturity_date"))
        expiry_tenor = pos.get("expiry_tenor") or pos.get("option_expiry")
        swap_tenor = pos.get("swap_tenor") or pos.get("tenor")
        if not expiry_tenor and maturity:
            years = max(0.25, round((maturity - valuation_date).days / 365.25))
            expiry_tenor = f"{int(years)}Y"
        if not swap_tenor:
            swap_tenor = "5Y"
        return {
            "instrument_type": "SWAPTION",
            "expiry_tenor": str(expiry_tenor),
            "swap_tenor": str(swap_tenor),
            "strike": pos.get("strike", "ATM") or "ATM",
            "payer_receiver": "PAYER" if direction in {"LONG", "BUY"} else "RECEIVER",
            "notional": notional * sign,
            "vol_type": "NORMAL",
        }
    
    return None


def compute_curve_risk_metrics(
    positions_df: pd.DataFrame,
    market_state: MarketState,
    valuation_date: date,
    keyrate_tenors: Optional[List[str]] = None,
    bump_bp: float = 1.0,
) -> CurveRiskMetrics:
    """
    Compute portfolio DV01 and key-rate DV01 using bump-and-reprice.
    
    This function replaces synthetic/approximated risk values with actual
    repricing under shocked curves.
    
    Args:
        positions_df: DataFrame with position details
        market_state: Current market state (curves + vol surface)
        valuation_date: Valuation date for pricing
        keyrate_tenors: List of tenors for key-rate ladder (default: 2Y,5Y,10Y,30Y)
        bump_bp: Bump size in basis points (default: 1.0)
    
    Returns:
        CurveRiskMetrics object with total DV01, key-rate ladder, etc.
    """
    keyrate_tenors = keyrate_tenors or DEFAULT_KEYRATE_TENORS
    
    # Build list of priceable trades
    trades = []
    excluded_types = set()
    
    for _, pos in positions_df.iterrows():
        trade = _build_trade_from_position(pos, valuation_date)
        if trade:
            trades.append(trade)
        else:
            excluded_types.add(str(pos.get("instrument_type", "UNKNOWN")))
    
    if not trades:
        return CurveRiskMetrics(
            total_dv01=0.0,
            keyrate_dv01={t: 0.0 for t in keyrate_tenors},
            worst_keyrate_dv01=0.0,
            base_pv=0.0,
            tenors_used=keyrate_tenors,
            instrument_coverage=0,
            total_instruments=len(positions_df),
            excluded_types=list(excluded_types),
        )
    
    # Compute base PV
    base_pv = 0.0
    for trade in trades:
        try:
            result = price_trade(trade, market_state)
            base_pv += result.pv
        except Exception:
            pass
    
    # Compute parallel DV01 using bump-and-reprice
    bump_engine = BumpEngine(market_state.curve.discount_curve)
    bumped_discount = bump_engine.parallel_bump(bump_bp)
    
    # Also bump projection curve if different
    if market_state.curve.projection_curve is market_state.curve.discount_curve:
        bumped_projection = bumped_discount
    else:
        proj_engine = BumpEngine(market_state.curve.projection_curve)
        bumped_projection = proj_engine.parallel_bump(bump_bp)
    
    bumped_curve_state = CurveState(
        discount_curve=bumped_discount,
        projection_curve=bumped_projection,
        metadata=market_state.curve.metadata,
    )
    bumped_market = MarketState(
        curve=bumped_curve_state,
        sabr_surface=market_state.sabr_surface,
        asof=market_state.asof,
    )
    
    # Price under bumped curves
    bumped_pv = 0.0
    for trade in trades:
        try:
            result = price_trade(trade, bumped_market)
            bumped_pv += result.pv
        except Exception:
            pass
    
    total_dv01 = -(bumped_pv - base_pv) / bump_bp
    
    # Compute key-rate DV01 ladder using localized bumps
    keyrate_dv01 = {}
    
    for tenor in keyrate_tenors:
        # Create localized bump at this tenor
        tenor_bumped_discount = bump_engine.tenor_bump(tenor, bump_bp)
        
        if market_state.curve.projection_curve is market_state.curve.discount_curve:
            tenor_bumped_projection = tenor_bumped_discount
        else:
            proj_engine = BumpEngine(market_state.curve.projection_curve)
            tenor_bumped_projection = proj_engine.tenor_bump(tenor, bump_bp)
        
        tenor_bumped_state = CurveState(
            discount_curve=tenor_bumped_discount,
            projection_curve=tenor_bumped_projection,
            metadata=market_state.curve.metadata,
        )
        tenor_bumped_market = MarketState(
            curve=tenor_bumped_state,
            sabr_surface=market_state.sabr_surface,
            asof=market_state.asof,
        )
        
        # Price under tenor-bumped curves
        tenor_bumped_pv = 0.0
        for trade in trades:
            try:
                result = price_trade(trade, tenor_bumped_market)
                tenor_bumped_pv += result.pv
            except Exception:
                pass
        
        kr_dv01 = -(tenor_bumped_pv - base_pv) / bump_bp
        keyrate_dv01[tenor] = kr_dv01
    
    # Find worst key-rate DV01
    worst_keyrate = max(abs(v) for v in keyrate_dv01.values()) if keyrate_dv01 else 0.0
    
    return CurveRiskMetrics(
        total_dv01=total_dv01,
        keyrate_dv01=keyrate_dv01,
        worst_keyrate_dv01=worst_keyrate,
        base_pv=base_pv,
        tenors_used=keyrate_tenors,
        instrument_coverage=len(trades),
        total_instruments=len(positions_df),
        excluded_types=list(excluded_types),
    )


def compute_limit_metrics(
    market_state: MarketState,
    positions_df: pd.DataFrame,
    valuation_date: date,
) -> (Dict[str, Optional[float]], Dict[str, bool]):
    """
    Compute metrics needed for limit evaluation using actual risk_trade outputs.

    Only metrics derivable from current positions/market_state are populated;
    others remain None so they show as Missing/Not Computed in the UI.
    """
    metrics: Dict[str, Optional[float]] = {
        "total_dv01": 0.0,
        "worst_keyrate_dv01": None,
        "option_delta": None,
        "option_gamma": None,
        "sabr_vega_atm": None,
        "sabr_vega_nu": None,
        "sabr_vega_rho": None,
        "var_95": None,
        "var_99": None,
        "es_975": None,
        "scenario_worst": None,
        "lvar_uplift": None,
        "sabr_rmse_max": None,
        "sabr_bucket_count": 0,
    }

    meta = {
        "has_option_positions": False,
        "computed_option_greeks": False,
        "has_var_results": False,
        "has_scenario_results": False,
        "has_liquidity_results": False,
        "has_keyrate_results": False,
    }

    # SABR diagnostics
    sabr_surface = getattr(market_state, "sabr_surface", None)
    if sabr_surface and getattr(sabr_surface, "params_by_bucket", None):
        metrics["sabr_bucket_count"] = len(sabr_surface.params_by_bucket)
        rmses = []
        for params in sabr_surface.params_by_bucket.values():
            diag = getattr(params, "diagnostics", {}) or {}
            if "rmse" in diag:
                rmses.append(diag["rmse"])
        if rmses:
            metrics["sabr_rmse_max"] = max(rmses)

    # Aggregate DV01 and option SABR sensitivities from positions
    opt_delta = 0.0
    opt_gamma = 0.0
    sabr_vega_atm = 0.0
    sabr_vega_nu = 0.0
    sabr_vega_rho = 0.0
    have_opt_delta = have_opt_gamma = False
    have_vegas = False

    for _, pos in positions_df.iterrows():
        inst = str(pos.get("instrument_type", "")).upper()
        notional = float(abs(pos.get("notional", 0.0)))
        direction = str(pos.get("direction", "")).upper()
        sign = -1.0 if direction in {"SHORT", "PAY_FIXED", "PAY"} else 1.0

        if inst in {"BOND", "UST"}:
            maturity = _parse_date(pos.get("maturity_date"))
            settlement = valuation_date
            if not maturity:
                continue
            trade = {
                "instrument_type": inst,
                "settlement": settlement,
                "maturity": maturity,
                "coupon": float(pos.get("coupon", 0.0)),
                "notional": notional * sign,
            }
        elif inst in {"SWAP", "IRS"}:
            maturity = _parse_date(pos.get("maturity_date"))
            if not maturity:
                continue
            pay_receive = "PAY" if "PAY" in direction else "RECEIVE"
            trade = {
                "instrument_type": "SWAP",
                "effective": valuation_date,
                "maturity": maturity,
                "notional": notional,
                "fixed_rate": float(pos.get("coupon", 0.0)),
                "pay_receive": pay_receive,
            }
        elif inst in {"SWAPTION", "CAPLET", "CAP", "CAPFLOOR"}:
            meta["has_option_positions"] = True
            # Attempt to build a minimal swaption trade using maturity tenor if available
            maturity = _parse_date(pos.get("maturity_date"))
            expiry_tenor = pos.get("expiry_tenor") or pos.get("option_expiry")
            swap_tenor = pos.get("swap_tenor") or pos.get("tenor")
            if not expiry_tenor and maturity:
                # derive tenor in years and round to nearest year label
                years = max(0.25, round((maturity - valuation_date).days / 365.25))
                expiry_tenor = f"{int(years)}Y"
            if not swap_tenor:
                swap_tenor = "5Y"
            trade = {
                "instrument_type": "SWAPTION",
                "expiry_tenor": str(expiry_tenor),
                "swap_tenor": str(swap_tenor),
                "strike": pos.get("strike", "ATM") or "ATM",
                "payer_receiver": "PAYER",
                "notional": notional * sign,
                "vol_type": "NORMAL",
            }
        else:
            continue

        try:
            res = risk_trade(trade, market_state)
        except Exception:
            continue

        if "dv01" in res:
            metrics["total_dv01"] += sign * float(res["dv01"])

        sens = res.get("sabr_sensitivities")
        if sens:
            # Use available sensitivities; defaults to 0.0 if missing
            have_vegas = True
            sabr_vega_atm += float(sens.get("dV_dsigma_atm", sens.get("vega", 0.0))) * sign
            sabr_vega_nu += float(sens.get("dV_dnu", 0.0)) * sign
            sabr_vega_rho += float(sens.get("dV_drho", 0.0)) * sign
        greeks = res.get("greeks")
        if greeks:
            opt_delta += float(greeks.get("delta", 0.0))
            opt_gamma += float(greeks.get("gamma", 0.0))
            have_opt_delta = True
            have_opt_gamma = True

    # Finalize optional aggregates
    metrics["total_dv01"] = metrics["total_dv01"] if metrics["total_dv01"] != 0 else 0.0
    if meta["has_option_positions"]:
        metrics["option_delta"] = opt_delta if have_opt_delta else None
        metrics["option_gamma"] = opt_gamma if have_opt_gamma else None
        metrics["sabr_vega_atm"] = sabr_vega_atm if have_vegas else None
        metrics["sabr_vega_nu"] = sabr_vega_nu if have_vegas else None
        metrics["sabr_vega_rho"] = sabr_vega_rho if have_vegas else None
        meta["computed_option_greeks"] = have_opt_delta or have_opt_gamma or have_vegas
    else:
        # Not applicable -> prefer explicit zeroes for limits
        metrics["option_delta"] = 0.0
        metrics["option_gamma"] = 0.0
        metrics["sabr_vega_atm"] = 0.0
        metrics["sabr_vega_nu"] = 0.0
        metrics["sabr_vega_rho"] = 0.0

    return metrics, meta


# ==============================================================================
# VaR Portfolio Pricing Functions
# ==============================================================================

@dataclass
class VaRCoverageInfo:
    """
    Information about VaR instrument coverage.
    
    Attributes:
        total_instruments: Total number of instruments in portfolio
        included_instruments: Number of instruments included in VaR
        excluded_instruments: Number of instruments excluded from VaR
        excluded_types: Set of instrument types that were excluded
        excluded_pv: PV of excluded instruments (options PV if not computed)
        included_pv: PV of included instruments
        coverage_ratio: included_pv / (included_pv + excluded_pv)
        is_linear_only: True if options are excluded
        warnings: List of warning messages
    """
    total_instruments: int
    included_instruments: int
    excluded_instruments: int
    excluded_types: List[str]
    excluded_pv: float
    included_pv: float
    coverage_ratio: float
    is_linear_only: bool
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_instruments": self.total_instruments,
            "included_instruments": self.included_instruments,
            "excluded_instruments": self.excluded_instruments,
            "excluded_types": self.excluded_types,
            "excluded_pv": self.excluded_pv,
            "included_pv": self.included_pv,
            "coverage_ratio": self.coverage_ratio,
            "is_linear_only": self.is_linear_only,
            "warnings": self.warnings,
        }


def build_var_portfolio_pricer(
    positions_df: pd.DataFrame,
    valuation_date: date,
    market_state: MarketState,
    include_options: bool = True,
) -> Tuple[callable, VaRCoverageInfo]:
    """
    Build a portfolio pricing function suitable for VaR computation.
    
    Returns a pricer function that takes a Curve and returns total PV,
    along with coverage information about what's included/excluded.
    
    Args:
        positions_df: DataFrame with position details
        valuation_date: Valuation date
        market_state: Market state (needed for options pricing with SABR)
        include_options: Whether to include options in VaR (default True)
    
    Returns:
        Tuple of (pricer_func, VaRCoverageInfo)
    """
    # Build trades from positions
    trades = []
    excluded_types = set()
    has_options = False
    option_pv = 0.0
    
    for _, pos in positions_df.iterrows():
        trade = _build_trade_from_position(pos, valuation_date)
        if trade:
            inst_type = trade["instrument_type"]
            is_option = inst_type in {"SWAPTION", "CAPLET", "CAP", "CAPFLOOR"}
            
            if is_option:
                has_options = True
                if not include_options:
                    excluded_types.add(inst_type)
                    # Try to price the option to get excluded PV
                    try:
                        from ..pricers.dispatcher import price_trade
                        result = price_trade(trade, market_state)
                        option_pv += result.pv
                    except Exception:
                        pass
                    continue
            
            trades.append(trade)
        else:
            excluded_types.add(str(pos.get("instrument_type", "UNKNOWN")))
    
    # Compute base PV for included instruments
    included_pv = 0.0
    for trade in trades:
        try:
            from ..pricers.dispatcher import price_trade
            result = price_trade(trade, market_state)
            included_pv += result.pv
        except Exception:
            pass
    
    total_pv = included_pv + option_pv
    coverage_ratio = included_pv / total_pv if total_pv != 0 else 1.0
    
    # Build warnings
    warnings = []
    is_linear_only = has_options and not include_options
    
    if is_linear_only:
        warnings.append(
            "⚠️ VaR/ES currently excludes options (linear-only). "
            f"Excluded options PV: ${option_pv:,.0f}"
        )
    
    if excluded_types:
        warnings.append(
            f"Excluded instrument types: {', '.join(sorted(excluded_types))}"
        )
    
    coverage_info = VaRCoverageInfo(
        total_instruments=len(positions_df),
        included_instruments=len(trades),
        excluded_instruments=len(positions_df) - len(trades),
        excluded_types=list(excluded_types),
        excluded_pv=option_pv,
        included_pv=included_pv,
        coverage_ratio=coverage_ratio,
        is_linear_only=is_linear_only,
        warnings=warnings,
    )
    
    # Build the pricer function
    def portfolio_pricer(curve):
        """Price the portfolio under a given curve (for VaR scenarios)."""
        from ..pricers.bonds import BondPricer
        from ..pricers.swaps import SwapPricer
        from ..pricers.dispatcher import price_trade
        from ..market_state import CurveState
        
        # Create market state with the new curve
        # For VaR, we typically only shock the curve, not vol
        new_curve_state = CurveState(
            discount_curve=curve,
            projection_curve=curve,  # Use same curve for projection
            metadata=market_state.curve.metadata if market_state.curve else {},
        )
        new_market_state = MarketState(
            curve=new_curve_state,
            sabr_surface=market_state.sabr_surface,  # Keep vol surface
            asof=market_state.asof,
        )
        
        pv_total = 0.0
        for trade in trades:
            try:
                inst = trade["instrument_type"]
                if inst in {"BOND", "UST"}:
                    pricer = BondPricer(curve)
                    dirty, _, _ = pricer.price(
                        settlement=trade["settlement"],
                        maturity=trade["maturity"],
                        coupon_rate=trade.get("coupon", 0.0),
                        face_value=100.0,
                        frequency=trade.get("frequency", 2),
                    )
                    pv_total += dirty / 100.0 * trade["notional"]
                elif inst in {"SWAP", "IRS"}:
                    pricer = SwapPricer(curve, curve)
                    direction = trade.get("pay_receive", "PAY")
                    pv_total += pricer.present_value(
                        effective=trade["effective"],
                        maturity=trade["maturity"],
                        notional=abs(trade["notional"]),
                        fixed_rate=trade.get("fixed_rate", 0.0),
                        pay_receive=direction,
                    )
                elif inst in {"SWAPTION", "CAPLET", "CAP", "CAPFLOOR"}:
                    # Options: use full pricer with updated market state
                    result = price_trade(trade, new_market_state)
                    pv_total += result.pv
                else:
                    # Fallback: try generic price_trade
                    result = price_trade(trade, new_market_state)
                    pv_total += result.pv
            except Exception:
                pass
        
        return pv_total
    
    return portfolio_pricer, coverage_info
