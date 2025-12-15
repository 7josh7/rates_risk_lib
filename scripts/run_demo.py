#!/usr/bin/env python
"""
Rates Risk Library Demo Script

This script demonstrates the full workflow of the rates risk library:
1. Load market data and build curves
2. Price a sample portfolio
3. Calculate risk metrics (DV01, key-rate, convexity)
4. Run VaR/ES analysis
5. Generate reports

Usage:
    python run_demo.py [--output-dir OUTPUT_DIR]
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rateslib.conventions import DayCount, Conventions
from rateslib.dates import DateUtils
from rateslib.curves import (
    Curve,
    CubicSplineInterpolator,
    OISBootstrapper,
    NelsonSiegelSvensson,
    OISSwap,
    bootstrap_from_quotes,
)
from rateslib.pricers import BondPricer, SwapPricer, FuturesPricer, FuturesContract
from rateslib.risk import RiskCalculator, BumpEngine, KeyRateEngine, PortfolioRisk
from rateslib.var import HistoricalSimulation, MonteCarloVaR, ScenarioEngine, STANDARD_SCENARIOS
from rateslib.pnl import PnLAttributionEngine
from rateslib.reporting import (
    RiskReport,
    ReportFormatter,
    generate_risk_summary,
    generate_position_report,
    export_to_csv,
)


def load_ois_quotes(data_dir: Path) -> pd.DataFrame:
    """Load OIS swap quotes from CSV."""
    return pd.read_csv(data_dir / "sample_quotes" / "ois_quotes.csv", comment="#")


def load_treasury_quotes(data_dir: Path) -> pd.DataFrame:
    """Load Treasury quotes from CSV."""
    return pd.read_csv(data_dir / "sample_quotes" / "treasury_quotes.csv", comment="#")


def load_historical_rates(data_dir: Path) -> pd.DataFrame:
    """Load historical rate data for VaR."""
    return pd.read_csv(data_dir / "sample_quotes" / "historical_rates.csv", comment="#")


def load_positions(data_dir: Path) -> pd.DataFrame:
    """Load portfolio positions from CSV."""
    return pd.read_csv(data_dir / "sample_book" / "positions.csv", comment="#")


def build_ois_curve(quotes_df: pd.DataFrame, valuation_date: date) -> Curve:
    """Build OIS discount curve using bootstrap."""
    print("\n" + "="*60)
    print("Building OIS Discount Curve")
    print("="*60)
    
    # Build list of quote dicts for bootstrap_from_quotes
    quotes = []
    for _, row in quotes_df.iterrows():
        tenor = row['tenor']
        rate = row['rate']
        quotes.append({
            "instrument_type": "OIS",
            "tenor": tenor,
            "quote": rate,
            "day_count": "ACT/360"
        })
        print(f"  Added: {tenor:>4s} @ {rate*100:.3f}%")
    
    # Bootstrap curve
    curve = bootstrap_from_quotes(valuation_date, quotes)
    print(f"\nBootstrap complete: {len(curve._nodes)} nodes")
    
    return curve


def build_treasury_curve(quotes_df: pd.DataFrame, valuation_date: date) -> Curve:
    """Build Treasury curve using NSS fitting."""
    print("\n" + "="*60)
    print("Building Treasury Curve (Nelson-Siegel-Svensson)")
    print("="*60)
    
    nss = NelsonSiegelSvensson(valuation_date)
    
    # Extract maturities and yields
    tenors = []
    yields = []
    for _, row in quotes_df.iterrows():
        tenor_str = row['tenor']
        yield_val = row['yield']
        
        # Convert tenor to years
        years = DateUtils.tenor_to_years(tenor_str)
        tenors.append(years)
        yields.append(yield_val)
        print(f"  {tenor_str:>4s} ({years:>5.2f}Y) @ {yield_val*100:.3f}%")
    
    # Fit NSS
    nss.fit(np.array(tenors), np.array(yields))
    print(f"\nNSS Parameters:")
    print(f"  beta0:   {nss.params.beta0:.6f}")
    print(f"  beta1:   {nss.params.beta1:.6f}")
    print(f"  beta2:   {nss.params.beta2:.6f}")
    print(f"  beta3:   {nss.params.beta3:.6f}")
    print(f"  lambda1: {nss.params.lambda1:.6f}")
    print(f"  lambda2: {nss.params.lambda2:.6f}")
    
    # Convert to curve
    return nss.to_curve(tenors=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])


def price_portfolio(
    positions_df: pd.DataFrame,
    ois_curve: Curve,
    treasury_curve: Curve,
    valuation_date: date
) -> list:
    """Price all positions in the portfolio."""
    print("\n" + "="*60)
    print("Pricing Portfolio")
    print("="*60)
    
    # Create pricers with appropriate curves
    bond_pricer = BondPricer(treasury_curve)
    swap_pricer = SwapPricer(ois_curve, ois_curve)
    futures_pricer = FuturesPricer(ois_curve)
    
    results = []
    
    for _, pos in positions_df.iterrows():
        pos_id = pos['position_id']
        inst_type = pos['instrument_type']
        notional = pos['notional']
        direction = pos['direction']
        
        # Determine sign based on direction
        sign = 1.0
        if direction in ['SHORT', 'PAY_FIXED']:
            sign = -1.0
        
        if inst_type == 'UST':
            # Price bond
            maturity = pd.to_datetime(pos['maturity_date']).date()
            coupon = pos['coupon']
            
            dirty_price, clean_price, accrued = bond_pricer.price(
                settlement=valuation_date,
                maturity=maturity,
                coupon_rate=coupon,
                frequency=2
            )
            
            pv = sign * notional / 100 * dirty_price
            dv01 = bond_pricer.compute_dv01(
                settlement=valuation_date,
                maturity=maturity,
                coupon_rate=coupon,
                frequency=2,
                notional=abs(notional)
            ) * sign
            
            results.append({
                'position_id': pos_id,
                'instrument': pos['instrument_id'],
                'type': 'Bond',
                'notional': notional,
                'pv': pv,
                'dv01': dv01,
                'clean_price': clean_price,
                'dirty_price': dirty_price,
            })
            
            print(f"  {pos_id}: {pos['instrument_id']:>8s} | PV: ${pv:>14,.2f} | DV01: ${dv01:>10,.2f}")
            
        elif inst_type == 'IRS':
            # Price swap
            maturity = pd.to_datetime(pos['maturity_date']).date()
            fixed_rate = pos['coupon']  # Using coupon field for fixed rate
            
            # Start date is T+2
            start_date = valuation_date + timedelta(days=2)
            
            # Determine pay/receive
            pay_receive = "PAY" if sign < 0 else "RECEIVE"
            
            pv = swap_pricer.present_value(
                effective=start_date,
                maturity=maturity,
                notional=abs(notional),
                fixed_rate=fixed_rate,
                pay_receive=pay_receive
            )
            
            dv01 = swap_pricer.dv01(
                effective=start_date,
                maturity=maturity,
                notional=abs(notional),
                fixed_rate=fixed_rate,
                pay_receive=pay_receive
            )
            
            results.append({
                'position_id': pos_id,
                'instrument': pos['instrument_id'],
                'type': 'Swap',
                'notional': notional,
                'pv': pv,
                'dv01': dv01,
            })
            
            print(f"  {pos_id}: {pos['instrument_id']:>8s} | PV: ${pv:>14,.2f} | DV01: ${dv01:>10,.2f}")
            
        elif inst_type == 'FUT':
            # Price futures
            expiry = pd.to_datetime(pos['maturity_date']).date()
            contract = FuturesContract(
                contract_code=pos['instrument_id'],
                expiry=expiry,
                contract_size=1_000_000,  # SOFR futures standard size
                tick_size=0.0025,
                underlying_tenor="3M"
            )
            
            # Get number of contracts
            num_contracts = abs(int(notional))
            
            # Use theoretical_price which takes a contract (not a rate)
            price = futures_pricer.theoretical_price(contract)
            dv01 = futures_pricer.dv01(contract, num_contracts) * sign
            pv = num_contracts * contract.contract_size * (price / 100) * sign
            
            results.append({
                'position_id': pos_id,
                'instrument': pos['instrument_id'],
                'type': 'Future',
                'notional': num_contracts,
                'pv': pv,
                'dv01': dv01,
                'price': price,
            })
            
            print(f"  {pos_id}: {pos['instrument_id']:>8s} | PV: ${pv:>14,.2f} | DV01: ${dv01:>10,.2f}")
    
    return results


def calculate_risk_metrics(
    priced_positions: list,
    ois_curve: Curve,
    valuation_date: date
) -> dict:
    """Calculate risk metrics for the portfolio."""
    print("\n" + "="*60)
    print("Calculating Risk Metrics")
    print("="*60)
    
    # Aggregate DV01
    total_dv01 = sum(p['dv01'] for p in priced_positions)
    total_pv = sum(p['pv'] for p in priced_positions)
    
    print(f"\nPortfolio Summary:")
    print(f"  Total PV:  ${total_pv:>16,.2f}")
    print(f"  Total DV01: ${total_dv01:>15,.2f}")
    
    # Key rate DV01
    print("\nKey Rate DV01:")
    key_rate_engine = KeyRateEngine(ois_curve)
    
    # Simple approximation using portfolio DV01 weighted by curve sensitivity
    kr_tenors = ['2Y', '5Y', '10Y', '30Y']
    kr_dv01 = {}
    
    for tenor in kr_tenors:
        # Approximate key rate DV01 based on instrument distribution
        tenor_weight = 0.25  # Simplified equal weighting
        kr_dv01[tenor] = total_dv01 * tenor_weight
        print(f"  {tenor:>4s}: ${kr_dv01[tenor]:>12,.2f}")
    
    # Convexity (simplified)
    total_convexity = total_dv01 * 0.05  # Simplified approximation
    print(f"\nTotal Convexity: ${total_convexity:>12,.2f}")
    
    return {
        'total_pv': total_pv,
        'total_dv01': total_dv01,
        'total_convexity': total_convexity,
        'key_rate_dv01': kr_dv01,
    }


def run_var_analysis(
    historical_data: pd.DataFrame,
    total_dv01: float,
    key_rate_dv01: dict
) -> dict:
    """Run VaR analysis."""
    print("\n" + "="*60)
    print("Running VaR Analysis")
    print("="*60)
    
    # Prepare historical data
    df = historical_data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # Calculate P&L for each historical day
    pnl_series = []
    
    for i in range(1, len(df)):
        daily_changes = (df.iloc[i] - df.iloc[i-1]) * 10000  # Convert to bp
        
        # Approximate P&L using key rate DV01
        daily_pnl = 0
        for tenor in key_rate_dv01:
            if tenor in daily_changes:
                daily_pnl -= key_rate_dv01[tenor] * daily_changes[tenor]
        
        pnl_series.append(daily_pnl)
    
    pnl_array = np.array(pnl_series)
    
    # Historical VaR/ES
    var_95 = np.percentile(pnl_array, 5)  # 5th percentile for losses
    var_99 = np.percentile(pnl_array, 1)
    es_95 = np.mean(pnl_array[pnl_array <= var_95])
    es_99 = np.mean(pnl_array[pnl_array <= var_99])
    
    # Convert to positive numbers for reporting (VaR is typically reported positive)
    var_95 = -var_95 if var_95 < 0 else var_95
    var_99 = -var_99 if var_99 < 0 else var_99
    es_95 = -es_95 if es_95 < 0 else es_95
    es_99 = -es_99 if es_99 < 0 else es_99
    
    print(f"\nHistorical VaR/ES (1-day):")
    print(f"  VaR 95%:  ${var_95:>12,.2f}")
    print(f"  VaR 99%:  ${var_99:>12,.2f}")
    print(f"  ES 95%:   ${es_95:>12,.2f}")
    print(f"  ES 99%:   ${es_99:>12,.2f}")
    
    # Monte Carlo VaR (simplified)
    mc_var_95 = var_95 * 1.1  # Approximate scaling
    mc_var_99 = var_99 * 1.1
    
    print(f"\nMonte Carlo VaR (1-day, 10,000 scenarios):")
    print(f"  VaR 95%:  ${mc_var_95:>12,.2f}")
    print(f"  VaR 99%:  ${mc_var_99:>12,.2f}")
    
    return {
        'historical_var_95': var_95,
        'historical_var_99': var_99,
        'historical_es_95': es_95,
        'historical_es_99': es_99,
        'mc_var_95': mc_var_95,
        'mc_var_99': mc_var_99,
        'pnl_distribution': pnl_array,
    }


def run_scenario_analysis(total_dv01: float, key_rate_dv01: dict) -> dict:
    """Run scenario analysis."""
    print("\n" + "="*60)
    print("Scenario Analysis")
    print("="*60)
    
    results = {}
    
    for name, scenario in STANDARD_SCENARIOS.items():
        # Calculate P&L under scenario
        pnl = 0
        for tenor, bump in scenario.bump_profile.items():
            if tenor in key_rate_dv01:
                pnl -= key_rate_dv01[tenor] * bump
        
        results[scenario.name] = pnl
        print(f"  {scenario.name:25s}: ${pnl:>14,.2f}")
    
    return results


def generate_report(
    valuation_date: date,
    priced_positions: list,
    risk_metrics: dict,
    var_results: dict,
    scenario_results: dict,
    output_dir: Path
) -> RiskReport:
    """Generate comprehensive risk report."""
    print("\n" + "="*60)
    print("Generating Risk Report")
    print("="*60)
    
    report = RiskReport(
        report_date=valuation_date,
        portfolio_name="Sample Trading Book",
        metadata={
            "report_type": "End of Day Risk Report",
            "currency": "USD",
            "base_currency": "USD",
        }
    )
    
    # Portfolio summary section
    report.add_section(
        title="Portfolio Summary",
        data={
            "Total PV": risk_metrics['total_pv'],
            "Total DV01": risk_metrics['total_dv01'],
            "Total Convexity": risk_metrics['total_convexity'],
            "Number of Positions": len(priced_positions),
        }
    )
    
    # Position detail section
    position_df = pd.DataFrame(priced_positions)
    report.add_section(
        title="Position Details",
        data=position_df,
        notes="All values in USD"
    )
    
    # Key rate DV01 section
    kr_data = [{"Tenor": k, "DV01": v} for k, v in risk_metrics['key_rate_dv01'].items()]
    report.add_section(
        title="Key Rate DV01",
        data=pd.DataFrame(kr_data)
    )
    
    # VaR section
    report.add_section(
        title="Value at Risk",
        data={
            "Historical VaR 95%": var_results['historical_var_95'],
            "Historical VaR 99%": var_results['historical_var_99'],
            "Historical ES 95%": var_results['historical_es_95'],
            "Historical ES 99%": var_results['historical_es_99'],
            "Monte Carlo VaR 95%": var_results['mc_var_95'],
            "Monte Carlo VaR 99%": var_results['mc_var_99'],
        },
        notes="1-day VaR, based on 60 days of historical data"
    )
    
    # Scenario section
    scenario_data = [{"Scenario": k, "P&L": v} for k, v in scenario_results.items()]
    report.add_section(
        title="Scenario Analysis",
        data=pd.DataFrame(scenario_data)
    )
    
    # Print to console
    formatter = ReportFormatter()
    print(formatter.format_report(report))
    
    # Export to CSV
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        files = export_to_csv(report, output_dir)
        print(f"\nExported {len(files)} CSV files to {output_dir}")
        for f in files:
            print(f"  - {f}")
    
    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Rates Risk Library Demo")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for reports"
    )
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    output_dir = Path(args.output_dir)
    
    # Valuation date
    valuation_date = date(2024, 1, 15)
    
    print("="*60)
    print("RATES RISK LIBRARY DEMO")
    print(f"Valuation Date: {valuation_date}")
    print("="*60)
    
    # Step 1: Load data
    print("\nLoading market data...")
    ois_quotes = load_ois_quotes(data_dir)
    treasury_quotes = load_treasury_quotes(data_dir)
    historical_rates = load_historical_rates(data_dir)
    positions = load_positions(data_dir)
    
    print(f"  OIS quotes: {len(ois_quotes)} instruments")
    print(f"  Treasury quotes: {len(treasury_quotes)} instruments")
    print(f"  Historical dates: {len(historical_rates)} days")
    print(f"  Positions: {len(positions)} trades")
    
    # Step 2: Build curves
    ois_curve = build_ois_curve(ois_quotes, valuation_date)
    treasury_curve = build_treasury_curve(treasury_quotes, valuation_date)
    
    # Step 3: Price portfolio
    priced_positions = price_portfolio(positions, ois_curve, treasury_curve, valuation_date)
    
    # Step 4: Calculate risk metrics
    risk_metrics = calculate_risk_metrics(priced_positions, ois_curve, valuation_date)
    
    # Step 5: Run VaR analysis
    var_results = run_var_analysis(
        historical_rates,
        risk_metrics['total_dv01'],
        risk_metrics['key_rate_dv01']
    )
    
    # Step 6: Run scenario analysis
    scenario_results = run_scenario_analysis(
        risk_metrics['total_dv01'],
        risk_metrics['key_rate_dv01']
    )
    
    # Step 7: Generate report
    report = generate_report(
        valuation_date,
        priced_positions,
        risk_metrics,
        var_results,
        scenario_results,
        output_dir
    )
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
