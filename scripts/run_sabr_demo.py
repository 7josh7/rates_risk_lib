#!/usr/bin/env python3
"""
SABR Model Demo Script

Demonstrates the complete SABR workflow:
1. Load volatility quotes from CSV
2. Calibrate SABR model to market data
3. Price caplets and swaptions using SABR vol
4. Compute model-consistent Greeks with smile correction
5. Generate risk reports
"""

import sys
from pathlib import Path
from datetime import date

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rateslib import (
    # Curves
    Curve,
    OISBootstrapper,
    # Vol
    SabrParams,
    SabrModel,
    SabrCalibrator,
    VolQuote,
    load_vol_quotes,
    hagan_black_vol,
    hagan_normal_vol,
    # Options
    bachelier_call,
    bachelier_greeks,
    CapletPricer,
    SwaptionPricer,
    SabrOptionRisk,
)


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def demo_sabr_basics():
    """Demonstrate basic SABR model usage."""
    print_section("1. SABR Model Basics")
    
    # Create SABR parameters (sigma_ATM parameterization)
    sabr_params = SabrParams(
        sigma_atm=0.0050,  # 50 bps ATM normal vol
        beta=0.5,          # Stochastic backbone parameter
        rho=-0.20,         # Forward-vol correlation (typically negative)
        nu=0.40,           # Vol-of-vol
        shift=0.02         # 2% shift for negative rate handling
    )
    
    print(f"SABR Parameters:")
    print(f"  sigma_ATM  = {sabr_params.sigma_atm*10000:.1f} bps")
    print(f"  beta       = {sabr_params.beta:.2f}")
    print(f"  rho        = {sabr_params.rho:.2f}")
    print(f"  nu         = {sabr_params.nu:.2f}")
    print(f"  shift      = {sabr_params.shift*100:.1f}%")
    
    # Initialize model and compute alpha from sigma_ATM
    model = SabrModel()
    F = 0.04  # 4% forward rate
    T = 1.0   # 1Y expiry
    
    alpha = model.alpha_from_sigma_atm(F, T, sabr_params)
    print(f"\n  Implied alpha = {alpha:.6f}")
    
    # Compute smile across strikes
    print(f"\nSABR Smile (F = {F*100:.1f}%, T = {T:.0f}Y):")
    print("-" * 50)
    print(f"{'Strike':>10} {'Moneyness':>12} {'Normal Vol':>12} {'Black Vol':>12}")
    print("-" * 50)
    
    strikes = [F - 0.02, F - 0.01, F, F + 0.01, F + 0.02]
    
    for K in strikes:
        normal_vol = model.implied_vol_normal(F, K, T, sabr_params)
        black_vol = model.implied_vol_black(F, K, T, sabr_params)
        moneyness = (K - F) * 10000  # in bps
        print(f"{K*100:>10.2f}% {moneyness:>10.0f}bp {normal_vol*10000:>10.1f}bp {black_vol*100:>10.2f}%")
    
    return sabr_params


def demo_sabr_calibration():
    """Demonstrate SABR calibration from market quotes."""
    print_section("2. SABR Calibration")
    
    import pandas as pd
    
    # Market vol quotes (strike in bps from ATM)
    F = 0.04  # 4% forward
    T = 1.0   # 1Y expiry
    
    market_strikes = [F - 0.01, F - 0.005, F, F + 0.005, F + 0.01]
    market_vols = [0.0052, 0.0050, 0.0048, 0.0051, 0.0055]  # Normal vols
    
    print(f"Market Data (F = {F*100:.1f}%, T = {T:.0f}Y):")
    print("-" * 40)
    for K, vol in zip(market_strikes, market_vols):
        print(f"  K = {K*100:.2f}%  ->  sigma = {vol*10000:.1f} bps")
    
    # Create DataFrame for calibrator
    quotes_df = pd.DataFrame({
        'strike': market_strikes,
        'vol': market_vols
    })
    
    # Calibrate SABR
    calibrator = SabrCalibrator(beta=0.5)
    result = calibrator.fit(
        quotes_df,
        F=F,
        T=T,
        vol_type="NORMAL"
    )
    
    print(f"\nCalibration Result:")
    print(f"  sigma_ATM  = {result.params.sigma_atm*10000:.2f} bps")
    print(f"  beta       = {result.params.beta:.2f} (fixed)")
    print(f"  rho        = {result.params.rho:.4f}")
    print(f"  nu         = {result.params.nu:.4f}")
    print(f"  Fit Error  = {result.fit_error:.6f}")
    
    # Compare fit to market
    model = SabrModel()
    print(f"\nFit Quality:")
    print("-" * 50)
    print(f"{'Strike':>10} {'Market':>10} {'SABR':>10} {'Error':>10}")
    print("-" * 50)
    
    for K, mkt_vol in zip(market_strikes, market_vols):
        sabr_vol = model.implied_vol_normal(F, K, T, result.params)
        error = (sabr_vol - mkt_vol) * 10000
        print(f"{K*100:>10.2f}% {mkt_vol*10000:>10.1f}bp {sabr_vol*10000:>10.1f}bp {error:>+10.2f}bp")
    
    return result.params


def demo_option_pricing(sabr_params: SabrParams):
    """Demonstrate option pricing with SABR."""
    print_section("3. Option Pricing with SABR")
    
    from rateslib.curves import Curve
    
    # Build a simple discount curve manually
    discount_curve = Curve(anchor_date=date(2024, 1, 15))
    
    # Add nodes with discount factors: DF(t) = exp(-r*t)
    tenors_years = [1/12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    rates = [0.045, 0.046, 0.047, 0.048, 0.045, 0.042, 0.040]
    
    for t, r in zip(tenors_years, rates):
        df = np.exp(-r * t)
        discount_curve.add_node(t, df)
    
    discount_curve.build()
    
    print("Discount Curve constructed:")
    print(f"  1Y DF = {discount_curve.discount_factor(1.0):.6f}")
    print(f"  5Y DF = {discount_curve.discount_factor(5.0):.6f}")
    
    # Price a caplet
    print("\n--- Caplet Pricing ---")
    caplet_pricer = CapletPricer(
        discount_curve=discount_curve,
        projection_curve=discount_curve
    )
    
    F = 0.04  # Forward rate
    K = 0.04  # ATM strike
    T = 1.0   # 1Y expiry
    tau = 0.25  # 3M accrual
    notional = 10_000_000
    
    model = SabrModel()
    vol = model.implied_vol_normal(F, K, T, sabr_params)
    df = discount_curve.discount_factor(T + tau)
    
    price = bachelier_call(F, K, T, vol, df * tau) * notional
    
    print(f"  Forward     = {F*100:.2f}%")
    print(f"  Strike      = {K*100:.2f}%")
    print(f"  Expiry      = {T:.0f}Y")
    print(f"  SABR Vol    = {vol*10000:.1f} bps")
    print(f"  Notional    = ${notional:,.0f}")
    print(f"  Price       = ${price:,.2f}")
    
    # Price a swaption
    print("\n--- Swaption Pricing ---")
    swaption_pricer = SwaptionPricer(
        discount_curve=discount_curve,
        projection_curve=discount_curve
    )
    
    # Get forward swap rate
    S, annuity = swaption_pricer.forward_swap_rate(expiry=1.0, tenor=5.0)
    K_swap = S  # ATM
    vol_swap = model.implied_vol_normal(S, K_swap, 1.0, sabr_params)
    
    swaption_result = swaption_pricer.price(
        S=S, K=K_swap, T=1.0, annuity=annuity, vol=vol_swap,
        vol_type="NORMAL", payer_receiver="PAYER", notional=notional
    )
    
    print(f"  Swap Rate   = {S*100:.3f}%")
    print(f"  Strike      = {K_swap*100:.3f}%")
    print(f"  Annuity     = {annuity:.4f}")
    print(f"  SABR Vol    = {vol_swap*10000:.1f} bps")
    print(f"  Notional    = ${notional:,.0f}")
    print(f"  Price       = ${swaption_result:,.2f}")
    
    return discount_curve


def demo_sabr_risk(sabr_params: SabrParams):
    """Demonstrate SABR risk analytics."""
    print_section("4. SABR Risk Analytics")
    
    risk_engine = SabrOptionRisk(vol_type="NORMAL")
    
    F = 0.04
    K = 0.04
    T = 1.0
    annuity = 4.5  # Approximate annuity for 5Y swap
    notional = 10_000_000
    
    report = risk_engine.risk_report(
        F=F, K=K, T=T, sabr_params=sabr_params,
        annuity=annuity, is_call=True, notional=notional
    )
    
    print(f"Risk Report (ATM Payer Swaption, ${notional:,.0f} notional):")
    print("-" * 50)
    print(f"  Forward        = {report.forward*100:.2f}%")
    print(f"  Strike         = {report.strike*100:.2f}%")
    print(f"  Implied Vol    = {report.implied_vol*10000:.1f} bps")
    print()
    print(f"  Delta (base)   = ${report.delta_base:,.0f}")
    print(f"  Delta (SABR)   = ${report.delta_sabr:,.0f}")
    print(f"  Gamma          = ${report.gamma_base:,.0f}")
    print()
    print(f"  Vega (ATM)     = ${report.vega_atm:,.0f}")
    print(f"  Vanna          = ${report.vanna:,.0f}")
    print(f"  Volga          = ${report.volga:,.0f}")
    
    # Delta decomposition
    print(f"\n  Delta Decomposition:")
    print(f"    Sideways     = ${report.delta_sideways:,.0f}")
    print(f"    Backbone     = ${report.delta_backbone:,.0f}")
    
    # Smile risk ladder
    print(f"\n--- Smile Risk Ladder ---")
    strikes = [F - 0.01, F - 0.005, F, F + 0.005, F + 0.01]
    ladder = risk_engine.smile_risk_ladder(
        F=F, strikes=strikes, T=T, sabr_params=sabr_params,
        annuity=annuity, is_call=True, notional=notional
    )
    
    print(f"{'Strike':>10} {'Vol':>8} {'Delta_B':>12} {'Delta_S':>12} {'Vega':>10}")
    print("-" * 55)
    for row in ladder:
        print(f"{row['strike']*100:>10.2f}% {row['implied_vol']*10000:>6.0f}bp "
              f"{row['delta_base']:>12,.0f} {row['delta_sabr']:>12,.0f} "
              f"{row['vega']:>10,.0f}")


def demo_vol_quotes():
    """Demonstrate loading and using vol quotes."""
    print_section("5. Loading Vol Quotes from CSV")
    
    data_path = Path(__file__).parent.parent / "data" / "vol_quotes.csv"
    
    if not data_path.exists():
        print(f"  [Warning] Vol quotes file not found: {data_path}")
        return
    
    quotes = load_vol_quotes(str(data_path))
    print(f"  Loaded {len(quotes)} vol quotes")
    
    # Group by expiry/tenor
    from collections import defaultdict
    by_expiry = defaultdict(list)
    for q in quotes:
        key = f"{q.expiry}-{q.underlying_tenor}"
        by_expiry[key].append(q)
    
    print(f"  Expiry/Tenor combinations: {len(by_expiry)}")
    print()
    
    # Show sample quotes for 1Y5Y
    key = "1Y-5Y"
    if key in by_expiry:
        print(f"  Sample quotes for {key}:")
        for q in by_expiry[key]:
            strike_str = str(q.strike).upper()
            print(f"    Strike {strike_str:>15}: {q.vol*10000:.1f} bps ({q.vol_type})")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print(" SABR Model Demo - Rates Risk Library")
    print("="*60)
    
    # 1. Basic SABR usage
    sabr_params = demo_sabr_basics()
    
    # 2. Calibration
    calibrated_params = demo_sabr_calibration()
    
    # 3. Option pricing
    demo_option_pricing(calibrated_params)
    
    # 4. Risk analytics
    demo_sabr_risk(calibrated_params)
    
    # 5. Vol quotes I/O
    demo_vol_quotes()
    
    print("\n" + "="*60)
    print(" Demo Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
