#!/usr/bin/env python3
"""
Comprehensive rates plus options demo.

This script exercises the full library stack on the shipped sample data:
1. Load market data and portfolio positions
2. Build an OIS curve and Treasury NSS curve
3. Calibrate a SABR surface
4. Assemble a MarketState
5. Price the sample portfolio with diagnostics
6. Compute curve risk and option-aware limit metrics
7. Run standard curve scenarios
8. Run historical simulation VaR
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

# Add src to path for direct script execution.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rateslib import (
    DEFAULT_LIMITS,
    CurveState,
    HistoricalSimulation,
    MarketState,
    NelsonSiegelSvensson,
    ScenarioEngine,
    build_sabr_surface,
    compute_limit_metrics,
    evaluate_limits,
    normalize_vol_quotes,
    price_portfolio_with_diagnostics,
    price_trade,
)
from rateslib.curves import bootstrap_from_quotes
from rateslib.dates import DateUtils
from rateslib.risk import compute_curve_risk_metrics


def print_section(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    print(f"\n{char * 78}")
    print(f" {title}")
    print(f"{char * 78}")


def normalize_vol_type(series: pd.Series) -> pd.Series:
    """Normalize volatility type labels for demo input files."""
    cleaned = series.fillna("NORMAL").astype(str).str.strip().str.upper()
    return cleaned.replace({"N": "NORMAL", "LN": "LOGNORMAL", "LOGN": "LOGNORMAL"})


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all sample market data needed by the demo."""
    print_section("1. Loading Market Data")

    ois_df = pd.read_csv(data_dir / "sample_quotes" / "ois_quotes.csv", comment="#")
    treasury_df = pd.read_csv(data_dir / "sample_quotes" / "treasury_quotes.csv", comment="#")
    historical_df = pd.read_csv(data_dir / "sample_quotes" / "historical_rates.csv", comment="#")
    vol_df = pd.read_csv(data_dir / "vol_quotes.csv", comment="#")
    positions_df = pd.read_csv(data_dir / "sample_book" / "positions.csv", comment="#")

    if "vol_type" in vol_df.columns:
        vol_df = vol_df.copy()
        vol_df["vol_type"] = normalize_vol_type(vol_df["vol_type"])

    print(f"[OK] Loaded {len(ois_df)} OIS quotes")
    print(f"[OK] Loaded {len(treasury_df)} Treasury quotes")
    print(f"[OK] Loaded {len(historical_df)} historical rate rows")
    print(f"[OK] Loaded {len(vol_df)} volatility quotes")
    print(f"[OK] Loaded {len(positions_df)} portfolio positions")

    print("\nInstrument counts:")
    for inst, count in positions_df["instrument_type"].value_counts().items():
        print(f"  - {inst}: {count}")

    return ois_df, treasury_df, historical_df, vol_df, positions_df


def build_curves(
    ois_df: pd.DataFrame,
    treasury_df: pd.DataFrame,
    valuation_date: date,
) -> tuple:
    """Build the discount curve and Treasury fit."""
    print_section("2. Curve Construction")

    ois_quotes = (
        ois_df[["instrument_type", "tenor", "rate", "day_count"]]
        .rename(columns={"rate": "quote"})
        .to_dict("records")
    )
    ois_curve = bootstrap_from_quotes(anchor_date=valuation_date, quotes=ois_quotes)

    print("[OK] OIS curve built")
    print(f"  Nodes: {len(ois_curve._nodes)}")
    print(f"  1Y discount factor:  {ois_curve.discount_factor(1.0):.6f}")
    print(f"  5Y discount factor:  {ois_curve.discount_factor(5.0):.6f}")
    print(f"  10Y discount factor: {ois_curve.discount_factor(10.0):.6f}")

    tenors = treasury_df["tenor"].map(DateUtils.tenor_to_years).to_numpy()
    yields = treasury_df["yield"].to_numpy()
    nss = NelsonSiegelSvensson(valuation_date)
    nss.fit(tenors, yields)
    treasury_curve = nss.to_curve(tenors=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])

    print("\n[OK] Treasury curve fit with NSS")
    print(f"  beta0:   {nss.params.beta0:.6f}")
    print(f"  beta1:   {nss.params.beta1:.6f}")
    print(f"  beta2:   {nss.params.beta2:.6f}")
    print(f"  lambda1: {nss.params.lambda1:.6f}")

    return ois_curve, treasury_curve, nss


def calibrate_sabr_surface(vol_df: pd.DataFrame, ois_curve, valuation_date: date):
    """Normalize quotes and calibrate the SABR surface."""
    print_section("3. SABR Surface Calibration")

    curve_state = CurveState(discount_curve=ois_curve, projection_curve=ois_curve)
    normalized = normalize_vol_quotes(
        raw_quotes=vol_df,
        curve_state=curve_state,
        instrument_hint="SWAPTION",
    )
    sabr_surface = build_sabr_surface(
        normalized_quotes=normalized,
        curve_state=curve_state,
        beta_policy=0.5,
    )

    print(f"[OK] Normalized {len(normalized)} quotes")
    print(f"[OK] Calibrated {len(sabr_surface.params_by_bucket)} SABR buckets")
    print(f"  Valuation date: {valuation_date}")

    print(f"\n{'Expiry':>8} {'Tenor':>8} {'sigma_atm':>12} {'rho':>10} {'nu':>10} {'RMSE':>12}")
    print("-" * 78)
    for bucket_key, params in list(sabr_surface.params_by_bucket.items())[:5]:
        expiry, tenor = bucket_key
        rmse = float(params.diagnostics.get("rmse", 0.0))
        print(
            f"{expiry:>8} {tenor:>8} "
            f"{params.sigma_atm * 10000:>10.1f}bp "
            f"{params.rho:>10.3f} "
            f"{params.nu:>10.3f} "
            f"{rmse * 10000:>10.2f}bp"
        )
    if len(sabr_surface.params_by_bucket) > 5:
        extra = len(sabr_surface.params_by_bucket) - 5
        print(f"  ... and {extra} more buckets")

    return sabr_surface


def assemble_market_state(ois_curve, treasury_curve, sabr_surface, valuation_date: date) -> MarketState:
    """Create the MarketState used throughout the demo."""
    print_section("4. Market State Assembly")

    curve_state = CurveState(
        discount_curve=ois_curve,
        projection_curve=ois_curve,
        metadata={"treasury_curve": treasury_curve, "valuation_date": valuation_date},
    )
    market_state = MarketState(curve=curve_state, sabr_surface=sabr_surface, asof=valuation_date)

    print("[OK] MarketState assembled")
    print("  Discount curve: OIS")
    print("  Projection curve: OIS")
    print(f"  SABR buckets: {len(sabr_surface.params_by_bucket)}")
    print(f"  As of: {valuation_date}")

    return market_state


def print_pricing_table(position_ids: Iterable[str], instrument_types: Iterable[str], pvs: Iterable[float]) -> None:
    """Print a compact pricing table."""
    print(f"\n{'Position':>10} {'Type':>12} {'PV':>18}")
    print("-" * 46)
    for position_id, inst_type, pv in zip(position_ids, instrument_types, pvs):
        print(f"{position_id:>10} {inst_type:>12} {pv:>18,.2f}")


def price_portfolio(positions_df: pd.DataFrame, market_state: MarketState, valuation_date: date):
    """Price the sample portfolio using the production builders."""
    print_section("5. Portfolio Pricing")

    pricing = price_portfolio_with_diagnostics(
        positions_df=positions_df,
        market_state=market_state,
        valuation_date=valuation_date,
        include_options=True,
        allow_legacy_options=False,
    )

    successful_ids = [trade.get("_position_id", f"TRADE_{i}") for i, trade in enumerate(pricing.successful_trades, start=1)]
    successful_types = [trade.get("instrument_type", "UNKNOWN") for trade in pricing.successful_trades]
    print_pricing_table(successful_ids, successful_types, pricing.successful_pvs)

    print("-" * 46)
    print(f"{'TOTAL':>10} {'':>12} {pricing.total_pv:>18,.2f}")
    print(f"\n[OK] Priced {pricing.successful_count} / {pricing.total_positions} positions")

    if pricing.failed_trades:
        print(f"[WARN] {pricing.failed_count} position(s) failed:")
        for failure in pricing.failed_trades:
            print(
                f"  - {failure.position_id} [{failure.instrument_type}] "
                f"at {failure.stage}: {failure.error_message}"
            )

    return pricing


def run_risk_and_limits(
    positions_df: pd.DataFrame,
    market_state: MarketState,
    valuation_date: date,
) -> tuple:
    """Compute curve risk, option metrics, and limit status."""
    print_section("6. Risk Metrics And Limits")

    curve_risk = compute_curve_risk_metrics(
        market_state=market_state,
        positions_df=positions_df,
        valuation_date=valuation_date,
    )
    limit_metrics, limit_meta = compute_limit_metrics(
        market_state=market_state,
        positions_df=positions_df,
        valuation_date=valuation_date,
    )

    print(f"[OK] Portfolio DV01: {curve_risk.total_dv01:,.2f}")
    print(f"[OK] Worst key-rate DV01: {curve_risk.worst_keyrate_dv01:,.2f}")
    print(f"[OK] Coverage: {curve_risk.instrument_coverage}/{curve_risk.total_instruments}")

    print("\nKey-rate ladder:")
    for tenor, dv01 in curve_risk.keyrate_dv01.items():
        print(f"  - {tenor:>4}: {dv01:>12,.2f}")

    if limit_meta["has_option_positions"]:
        print("\nOption aggregates:")
        print(f"  - delta:        {limit_metrics['option_delta']}")
        print(f"  - gamma:        {limit_metrics['option_gamma']}")
        print(f"  - sabr vega:    {limit_metrics['sabr_vega_atm']}")
        print(f"  - sabr rho:     {limit_metrics['sabr_vega_rho']}")
        print(f"  - sabr nu:      {limit_metrics['sabr_vega_nu']}")

    if curve_risk.warnings:
        print("\nWarnings:")
        for warning in curve_risk.warnings:
            print(f"  - {warning}")

    return curve_risk, limit_metrics, limit_meta


def make_portfolio_pricer(successful_trades: List[dict], market_state: MarketState):
    """Build a callable that reprices the successful trade set under a shocked curve."""

    def portfolio_pricer(curve) -> float:
        scenario_state = MarketState(
            curve=CurveState(
                discount_curve=curve,
                projection_curve=curve,
                metadata=market_state.curve.metadata,
            ),
            sabr_surface=market_state.sabr_surface,
            asof=market_state.asof,
        )
        total_pv = 0.0
        for trade in successful_trades:
            total_pv += price_trade(trade, scenario_state).pv
        return total_pv

    return portfolio_pricer


def run_scenarios(successful_trades: List[dict], market_state: MarketState, curve_risk) -> list:
    """Run the built-in standard scenario set."""
    print_section("7. Scenario Analysis")

    portfolio_pricer = make_portfolio_pricer(successful_trades, market_state)
    engine = ScenarioEngine(
        base_curve=market_state.curve.discount_curve,
        pricer_func=portfolio_pricer,
        key_rate_dv01=curve_risk.keyrate_dv01,
    )
    results = engine.run_standard_scenarios()

    print(f"{'Scenario':>24} {'P&L':>18}")
    print("-" * 44)
    for result in results:
        print(f"{result.scenario.name:>24} {result.pnl:>18,.2f}")

    worst = min(results, key=lambda item: item.pnl)
    print("-" * 44)
    print(f"[OK] Worst scenario: {worst.scenario.name} ({worst.pnl:,.2f})")

    return results


def run_historical_var(
    historical_df: pd.DataFrame,
    successful_trades: List[dict],
    market_state: MarketState,
) -> object:
    """Run historical simulation VaR on the successfully priced trades."""
    print_section("8. Historical Simulation VaR")

    portfolio_pricer = make_portfolio_pricer(successful_trades, market_state)
    hs = HistoricalSimulation(
        base_curve=market_state.curve.discount_curve,
        historical_data=historical_df,
        pricer_func=portfolio_pricer,
    )
    result = hs.run_simulation(lookback_days=250)

    print(f"[OK] Scenarios used: {result.num_scenarios}")
    print(f"[OK] VaR 95: {result.var_95:,.2f}")
    print(f"[OK] VaR 99: {result.var_99:,.2f}")
    print(f"[OK] ES 95:  {result.es_95:,.2f}")
    print(f"[OK] ES 99:  {result.es_99:,.2f}")
    print(f"[OK] Worst loss: {result.worst_loss:,.2f}")

    return result


def apply_limit_checks(limit_metrics: dict, curve_risk, scenario_results: list, var_result) -> list:
    """Populate derived limit metrics and evaluate the default limit set."""
    scenario_worst = max(0.0, max(-result.pnl for result in scenario_results))

    enriched_metrics = dict(limit_metrics)
    enriched_metrics["worst_keyrate_dv01"] = curve_risk.worst_keyrate_dv01
    enriched_metrics["scenario_worst"] = scenario_worst
    enriched_metrics["var_95"] = var_result.var_95
    enriched_metrics["var_99"] = var_result.var_99
    enriched_metrics["es_975"] = var_result.es_95

    results = evaluate_limits(enriched_metrics, DEFAULT_LIMITS)

    print_section("9. Limit Evaluation")
    print(f"{'Metric':>28} {'Value':>16} {'Status':>12}")
    print("-" * 60)
    for result in results:
        if result.value is None:
            continue
        print(f"{result.definition.name:>28} {result.value:>16,.2f} {result.status:>12}")

    breaches = sum(1 for result in results if result.status == "Breach")
    warnings = sum(1 for result in results if result.status == "Warning")
    print("-" * 60)
    print(f"[OK] Limit status summary: {breaches} breach(es), {warnings} warning(s)")

    return results


def main() -> None:
    """Run the full comprehensive demo."""
    print("\n" + "=" * 78)
    print(" COMPREHENSIVE RATES PLUS OPTIONS DEMO")
    print(" End-to-end smoke test of curves, pricing, risk, scenarios, and VaR")
    print("=" * 78)

    data_dir = Path(__file__).parent.parent / "data"
    valuation_date = date(2024, 1, 15)

    ois_df, treasury_df, historical_df, vol_df, positions_df = load_data(data_dir)
    ois_curve, treasury_curve, _ = build_curves(ois_df, treasury_df, valuation_date)
    sabr_surface = calibrate_sabr_surface(vol_df, ois_curve, valuation_date)
    market_state = assemble_market_state(ois_curve, treasury_curve, sabr_surface, valuation_date)

    pricing = price_portfolio(positions_df, market_state, valuation_date)
    curve_risk, limit_metrics, _ = run_risk_and_limits(positions_df, market_state, valuation_date)
    scenario_results = run_scenarios(pricing.successful_trades, market_state, curve_risk)
    var_result = run_historical_var(historical_df, pricing.successful_trades, market_state)
    limit_results = apply_limit_checks(limit_metrics, curve_risk, scenario_results, var_result)

    print_section("SUMMARY")
    print(f"[OK] Successful prices: {pricing.successful_count}/{pricing.total_positions}")
    print(f"[OK] Portfolio PV: {pricing.total_pv:,.2f}")
    print(f"[OK] Portfolio DV01: {curve_risk.total_dv01:,.2f}")
    print(f"[OK] Worst scenario loss: {max(0.0, max(-item.pnl for item in scenario_results)):,.2f}")
    print(f"[OK] Historical VaR 95: {var_result.var_95:,.2f}")
    print(f"[OK] Evaluated {len(limit_results)} default limits")


if __name__ == "__main__":
    main()
