#!/usr/bin/env python3
"""
Debug utility to spot negative swaption implied vols and trace the cause.

The script rebuilds the same inputs used by the dashboard:
- Bootstraps the OIS curve from sample quotes
- Normalizes and calibrates the SABR surface from data/vol_quotes.csv
- Prices a swaption (default 10Y x 10Y ATM) and sweeps strike offsets

For any negative implied vol it reports:
    - Requested and resolved SABR bucket (including fallback usage)
    - Forward/strike/expiry inputs
    - Black vol, normal vol, and the Black->Normal conversion factor
      so it is clear whether the sign flip comes from the conversion term
      or the underlying Black vol.
"""

import sys
from pathlib import Path
from datetime import date
from typing import Iterable, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rateslib.curves.bootstrap import bootstrap_from_quotes
from rateslib.market_state import CurveState, MarketState
from rateslib.vol.quotes import normalize_vol_quotes
from rateslib.vol.calibration import build_sabr_surface
from rateslib.vol.sabr_surface import make_bucket_key
from rateslib.vol.sabr import SabrModel, hagan_black_vol, hagan_normal_vol
from rateslib.options.swaption import SwaptionPricer
from rateslib.dates import DateUtils


ROOT = Path(__file__).resolve().parent.parent
SAMPLE_OIS_PATH = ROOT / "data" / "sample_quotes" / "ois_quotes.csv"
VOL_QUOTES_PATH = ROOT / "data" / "vol_quotes.csv"


def build_curve_state(anchor: date = date(2024, 1, 15)) -> CurveState:
    """Bootstrap an OIS curve from sample quotes and wrap in CurveState."""
    ois_df = pd.read_csv(SAMPLE_OIS_PATH, comment="#")
    quotes = [
        {
            "instrument_type": row["instrument_type"],
            "tenor": row["tenor"],
            "quote": row["rate"],
            "day_count": row.get("day_count", "ACT/360"),
        }
        for _, row in ois_df.iterrows()
    ]
    curve = bootstrap_from_quotes(anchor, quotes)
    return CurveState(discount_curve=curve, projection_curve=curve)


def build_market_state(curve_state: CurveState) -> Tuple[MarketState, pd.DataFrame]:
    """Normalize vol quotes, calibrate SABR, and return MarketState plus normalized quotes."""
    vol_df = pd.read_csv(VOL_QUOTES_PATH)
    normalized = normalize_vol_quotes(vol_df, curve_state)
    sabr_surface = build_sabr_surface(normalized, curve_state, beta_policy=0.5)
    market_state = MarketState(curve=curve_state, sabr_surface=sabr_surface, asof=str(curve_state.discount_curve.anchor_date))
    return market_state, normalized


def _resolved_bucket(sabr_surface, requested: Tuple[str, str], params) -> Tuple[str, str]:
    """Determine which bucket was actually used (accounts for fallback)."""
    requested_key = make_bucket_key(*requested)
    if requested_key in sabr_surface.params_by_bucket:
        return requested_key

    fallback = params.diagnostics.get("fallback_from", [])
    if fallback:
        last = fallback[-1]
        used = tuple(last.get("used", requested_key))
        return used
    return requested_key


def diagnose_implied_vol(
    market_state: MarketState,
    expiry: str,
    swap_tenor: str,
    strike_offset_bp: float,
) -> Optional[Dict[str, Any]]:
    """
    Compute implied vol for a single (expiry, tenor, strike offset) and return diagnostics.
    """
    pricer = SwaptionPricer(
        market_state.curve.discount_curve, market_state.curve.projection_curve
    )
    model = SabrModel()

    T_expiry = DateUtils.tenor_to_years(expiry)
    T_swap = DateUtils.tenor_to_years(swap_tenor)
    forward, _ = pricer.forward_swap_rate(T_expiry, T_swap)

    strike = forward + strike_offset_bp / 10000.0
    if strike <= 0:
        strike = max(forward, 1e-6)

    params_bucket = market_state.get_sabr_params(expiry, swap_tenor, allow_fallback=True)
    if params_bucket is None:
        return None

    resolved_bucket = _resolved_bucket(
        market_state.sabr_surface, (expiry, swap_tenor), params_bucket
    )

    params = params_bucket.to_sabr_params()
    alpha = model.alpha_from_sigma_atm(forward, T_expiry, params)

    black_vol = hagan_black_vol(
        forward, strike, T_expiry, alpha, params.beta, params.rho, params.nu, params.shift
    )
    normal_vol = hagan_normal_vol(
        forward, strike, T_expiry, alpha, params.beta, params.rho, params.nu, params.shift
    )

    F_s = forward + params.shift
    K_s = strike + params.shift
    if F_s > 0 and K_s > 0:
        log_fk = np.log(F_s / K_s)
        conversion_factor = np.sqrt(F_s * K_s) * (1 - log_fk**2 / 24)
    else:
        log_fk = np.nan
        conversion_factor = np.nan

    reason = None
    if normal_vol < 0:
        if black_vol <= 0:
            reason = "black_vol_non_positive"
        elif conversion_factor < 0:
            reason = "conversion_factor_negative"
        else:
            reason = "unknown_sign_issue"

    return {
        "requested_bucket": make_bucket_key(expiry, swap_tenor),
        "used_bucket": resolved_bucket,
        "forward": forward,
        "strike": strike,
        "expiry_years": T_expiry,
        "strike_offset_bp": strike_offset_bp,
        "sigma_atm_param": params.sigma_atm,
        "rho": params.rho,
        "nu": params.nu,
        "beta": params.beta,
        "shift": params.shift,
        "black_vol": black_vol,
        "normal_vol": normal_vol,
        "conversion_factor": conversion_factor,
        "log_fk": log_fk,
        "cause": reason,
    }


def sweep_and_report(
    market_state: MarketState,
    normalized_quotes: pd.DataFrame,
    strike_offsets: Iterable[int],
):
    """
    Sweep through buckets and strike offsets, printing any negative implied vols.
    """
    buckets = set(
        normalized_quotes[["expiry", "tenor"]].drop_duplicates().itertuples(index=False, name=None)
    )
    # Ensure the dashboard scenario is covered even if missing from quotes
    buckets.add(("10Y", "10Y"))

    rows = []
    for expiry, tenor in sorted(buckets):
        for off in strike_offsets:
            diag = diagnose_implied_vol(market_state, expiry, tenor, off)
            if diag:
                rows.append(diag)

    df = pd.DataFrame(rows)
    negative = df[df["normal_vol"] < 0] if not df.empty else pd.DataFrame()

    print("\nChecked implied vols ({} combinations).".format(len(rows)))
    if negative.empty:
        print("No negative implied vols found in the sweep.")
        return

    cols = [
        "requested_bucket",
        "used_bucket",
        "strike_offset_bp",
        "forward",
        "strike",
        "normal_vol",
        "black_vol",
        "conversion_factor",
        "cause",
    ]
    print("\nNegative implied vols detected:")
    print(negative[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main():
    curve_state = build_curve_state()
    market_state, normalized = build_market_state(curve_state)

    print("Loaded curve with anchor date:", curve_state.discount_curve.anchor_date)
    print("Calibrated SABR buckets:", len(market_state.sabr_surface.params_by_bucket))

    # Targeted check: dashboard scenario
    print("\n--- Target scenario: 10Y x 10Y ATM ---")
    target = diagnose_implied_vol(market_state, "10Y", "10Y", strike_offset_bp=0)
    if target:
        for k in [
            "requested_bucket",
            "used_bucket",
            "forward",
            "strike",
            "normal_vol",
            "black_vol",
            "conversion_factor",
            "cause",
        ]:
            print(f"{k:>18}: {target[k]}")
    else:
        print("No SABR parameters available for 10Y x 10Y.")

    # Sweep around the strike slider range to catch sign flips
    sweep_and_report(market_state, normalized, strike_offsets=range(-200, 201, 50))


if __name__ == "__main__":
    main()
