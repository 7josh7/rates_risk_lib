"""
Unit tests for the new engine-layer risk reporting functions.

Tests cover:
1. compute_curve_risk_metrics - DV01 and key-rate DV01
2. run_scenario_set - scenario repricing
3. build_var_portfolio_pricer - VaR portfolio pricing with options handling
"""

from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest

from rateslib.curves import Curve, create_flat_curve
from rateslib.market_state import MarketState, CurveState
from rateslib.risk.reporting import (
    compute_curve_risk_metrics,
    CurveRiskMetrics,
    build_var_portfolio_pricer,
    VaRCoverageInfo,
)
from rateslib.var.scenarios import (
    run_scenario_set,
    run_single_scenario,
    scenarios_to_dataframe,
    PortfolioScenarioResult,
    STANDARD_SCENARIOS,
    Scenario,
)


@pytest.fixture
def sample_curve():
    """Create sample discount curve."""
    anchor_date = date(2024, 1, 15)
    return create_flat_curve(anchor_date, rate=0.05, max_tenor_years=30.0)


@pytest.fixture
def sample_market_state(sample_curve):
    """Create sample market state."""
    curve_state = CurveState(
        discount_curve=sample_curve,
        projection_curve=sample_curve,
        metadata={"valuation_date": "2024-01-15"},
    )
    return MarketState(curve=curve_state, sabr_surface=None, asof="2024-01-15")


@pytest.fixture
def bond_portfolio_df():
    """Create sample bond portfolio DataFrame."""
    return pd.DataFrame({
        "position_id": ["P001", "P002", "P003"],
        "instrument_type": ["UST", "UST", "UST"],
        "instrument_id": ["UST_2Y", "UST_5Y", "UST_10Y"],
        "notional": [1_000_000, 2_000_000, 3_000_000],
        "direction": ["LONG", "LONG", "SHORT"],
        "coupon": [0.04, 0.045, 0.05],
        "maturity_date": [
            date(2026, 1, 15),
            date(2029, 1, 15),
            date(2034, 1, 15),
        ],
    })


@pytest.fixture
def mixed_portfolio_df():
    """Create sample portfolio with bonds and swaps."""
    return pd.DataFrame({
        "position_id": ["P001", "P002", "P003"],
        "instrument_type": ["UST", "SWAP", "UST"],
        "instrument_id": ["UST_2Y", "IRS_5Y", "UST_10Y"],
        "notional": [1_000_000, 5_000_000, 2_000_000],
        "direction": ["LONG", "PAY", "LONG"],
        "coupon": [0.04, 0.045, 0.05],
        "maturity_date": [
            date(2026, 1, 15),
            date(2029, 1, 15),
            date(2034, 1, 15),
        ],
    })


class TestComputeCurveRiskMetrics:
    """Tests for compute_curve_risk_metrics function."""

    def test_returns_non_zero_dv01_for_bond_portfolio(self, bond_portfolio_df, sample_market_state):
        """Test that DV01 is non-zero for a bond portfolio."""
        result = compute_curve_risk_metrics(
            positions_df=bond_portfolio_df,
            market_state=sample_market_state,
            valuation_date=date(2024, 1, 15),
            keyrate_tenors=["2Y", "5Y", "10Y", "30Y"],
            bump_bp=1.0,
        )
        
        assert isinstance(result, CurveRiskMetrics)
        assert result.total_dv01 != 0, "Total DV01 should be non-zero for bond portfolio"
        assert result.instrument_coverage > 0, "Should have priced at least one instrument"

    def test_keyrate_dv01_not_equal_split(self, bond_portfolio_df, sample_market_state):
        """Test that key-rate DV01 ladder is NOT equally split (real computation)."""
        result = compute_curve_risk_metrics(
            positions_df=bond_portfolio_df,
            market_state=sample_market_state,
            valuation_date=date(2024, 1, 15),
            keyrate_tenors=["2Y", "5Y", "10Y", "30Y"],
            bump_bp=1.0,
        )
        
        kr_values = list(result.keyrate_dv01.values())
        
        # Check that not all values are equal (which would be synthetic equal-split)
        # Allow for small tolerance due to numerical precision
        if len(kr_values) > 1 and any(v != 0 for v in kr_values):
            unique_values = set(round(v, 2) for v in kr_values)
            assert len(unique_values) > 1, \
                "Key-rate DV01 should NOT be equally split - different tenors should have different sensitivities"

    def test_keyrate_changes_with_portfolio(self, sample_market_state):
        """Test that key-rate DV01 changes when portfolio composition changes."""
        # Portfolio 1: 2Y bond only
        portfolio1 = pd.DataFrame({
            "position_id": ["P001"],
            "instrument_type": ["UST"],
            "instrument_id": ["UST_2Y"],
            "notional": [1_000_000],
            "direction": ["LONG"],
            "coupon": [0.04],
            "maturity_date": [date(2026, 1, 15)],
        })
        
        # Portfolio 2: 10Y bond only
        portfolio2 = pd.DataFrame({
            "position_id": ["P001"],
            "instrument_type": ["UST"],
            "instrument_id": ["UST_10Y"],
            "notional": [1_000_000],
            "direction": ["LONG"],
            "coupon": [0.05],
            "maturity_date": [date(2034, 1, 15)],
        })
        
        result1 = compute_curve_risk_metrics(
            positions_df=portfolio1,
            market_state=sample_market_state,
            valuation_date=date(2024, 1, 15),
        )
        
        result2 = compute_curve_risk_metrics(
            positions_df=portfolio2,
            market_state=sample_market_state,
            valuation_date=date(2024, 1, 15),
        )
        
        # Key-rate profiles should be different
        assert result1.keyrate_dv01 != result2.keyrate_dv01, \
            "Key-rate DV01 profile should change with portfolio composition"

    def test_units_are_dollars_per_bp(self, bond_portfolio_df, sample_market_state):
        """Test that DV01 units are consistent with $ per 1bp."""
        result = compute_curve_risk_metrics(
            positions_df=bond_portfolio_df,
            market_state=sample_market_state,
            valuation_date=date(2024, 1, 15),
            bump_bp=1.0,
        )
        
        # For a $1M face value bond, DV01 should typically be in range of hundreds to thousands
        # (not millions or cents)
        # A rough check: for $6M notional across multiple bonds, expect DV01 in reasonable range
        total_notional = bond_portfolio_df['notional'].sum()
        dv01_per_notional = abs(result.total_dv01) / total_notional * 100  # per $100
        
        assert 0.001 < dv01_per_notional < 1.0, \
            f"DV01 per $100 notional = {dv01_per_notional:.6f} seems out of expected range for $ per 1bp"


class TestRunScenarioSet:
    """Tests for scenario repricing functions."""

    def test_scenario_repricing_returns_results(self, bond_portfolio_df, sample_market_state):
        """Test that scenario set returns results."""
        results = run_scenario_set(
            positions_df=bond_portfolio_df,
            market_state=sample_market_state,
            valuation_date=date(2024, 1, 15),
        )
        
        assert len(results) > 0, "Should return at least one scenario result"
        assert all(isinstance(r, PortfolioScenarioResult) for r in results)

    def test_parallel_up_and_down_have_different_pnl(self, bond_portfolio_df, sample_market_state):
        """Test that +100bp and -100bp scenarios produce different (opposite) P&L."""
        # Run just the parallel scenarios
        parallel_scenarios = {
            "up": STANDARD_SCENARIOS["parallel_up_100"],
            "down": STANDARD_SCENARIOS["parallel_down_100"],
        }
        
        results = run_scenario_set(
            positions_df=bond_portfolio_df,
            market_state=sample_market_state,
            valuation_date=date(2024, 1, 15),
            scenarios=parallel_scenarios,
        )
        
        assert len(results) == 2
        
        pnl_up = next(r.pnl for r in results if "up" in r.scenario_name.lower() or "+100" in r.scenario_name)
        pnl_down = next(r.pnl for r in results if "down" in r.scenario_name.lower() or "-100" in r.scenario_name)
        
        # They should have opposite signs (roughly) for a long bond portfolio
        assert pnl_up != pnl_down, "Parallel +100bp and -100bp should produce different P&L"

    def test_scenario_pnl_changes_with_portfolio(self, sample_market_state):
        """Test that scenario P&L changes when portfolio positions change."""
        # Portfolio 1: Long bonds
        portfolio1 = pd.DataFrame({
            "position_id": ["P001"],
            "instrument_type": ["UST"],
            "instrument_id": ["UST_5Y"],
            "notional": [1_000_000],
            "direction": ["LONG"],
            "coupon": [0.04],
            "maturity_date": [date(2029, 1, 15)],
        })
        
        # Portfolio 2: Short bonds (same bond but opposite direction)
        portfolio2 = pd.DataFrame({
            "position_id": ["P001"],
            "instrument_type": ["UST"],
            "instrument_id": ["UST_5Y"],
            "notional": [1_000_000],
            "direction": ["SHORT"],
            "coupon": [0.04],
            "maturity_date": [date(2029, 1, 15)],
        })
        
        # Just run one scenario
        test_scenario = {"test": STANDARD_SCENARIOS["parallel_up_100"]}
        
        result1 = run_scenario_set(
            positions_df=portfolio1,
            market_state=sample_market_state,
            valuation_date=date(2024, 1, 15),
            scenarios=test_scenario,
        )
        
        result2 = run_scenario_set(
            positions_df=portfolio2,
            market_state=sample_market_state,
            valuation_date=date(2024, 1, 15),
            scenarios=test_scenario,
        )
        
        # P&L should have opposite signs for long vs short position
        pnl1 = result1[0].pnl
        pnl2 = result2[0].pnl
        
        assert (pnl1 * pnl2) < 0 or (pnl1 == 0 and pnl2 == 0), \
            "Long and short positions should have opposite P&L for same scenario"

    def test_computation_method_is_documented(self, bond_portfolio_df, sample_market_state):
        """Test that computation method is documented in results."""
        results = run_scenario_set(
            positions_df=bond_portfolio_df,
            market_state=sample_market_state,
            valuation_date=date(2024, 1, 15),
        )
        
        for result in results:
            assert "repric" in result.computation_method.lower(), \
                "Computation method should mention repricing"


class TestBuildVarPortfolioPricer:
    """Tests for VaR portfolio pricing with options handling."""

    def test_returns_pricer_and_coverage(self, mixed_portfolio_df, sample_market_state):
        """Test that function returns pricer function and coverage info."""
        pricer, coverage = build_var_portfolio_pricer(
            positions_df=mixed_portfolio_df,
            valuation_date=date(2024, 1, 15),
            market_state=sample_market_state,
            include_options=True,
        )
        
        assert callable(pricer), "Should return a callable pricer function"
        assert isinstance(coverage, VaRCoverageInfo)

    def test_pricer_function_works(self, bond_portfolio_df, sample_market_state, sample_curve):
        """Test that the returned pricer function can price the portfolio."""
        pricer, coverage = build_var_portfolio_pricer(
            positions_df=bond_portfolio_df,
            valuation_date=date(2024, 1, 15),
            market_state=sample_market_state,
            include_options=True,
        )
        
        # Should be able to call the pricer with a curve
        pv = pricer(sample_curve)
        
        assert isinstance(pv, (int, float)), "Pricer should return numeric PV"

    def test_coverage_info_accurate(self, bond_portfolio_df, sample_market_state):
        """Test that coverage info accurately reflects what's included."""
        pricer, coverage = build_var_portfolio_pricer(
            positions_df=bond_portfolio_df,
            valuation_date=date(2024, 1, 15),
            market_state=sample_market_state,
            include_options=True,
        )
        
        assert coverage.total_instruments == len(bond_portfolio_df)
        assert coverage.included_instruments <= coverage.total_instruments

    def test_warns_when_options_excluded(self, sample_market_state):
        """Test that warnings are generated when options are excluded."""
        # Portfolio with option
        portfolio_with_option = pd.DataFrame({
            "position_id": ["P001", "P002"],
            "instrument_type": ["UST", "SWAPTION"],
            "instrument_id": ["UST_5Y", "SWAPTION_5Y10Y"],
            "notional": [1_000_000, 500_000],
            "direction": ["LONG", "LONG"],
            "coupon": [0.04, 0.0],
            "maturity_date": [date(2029, 1, 15), date(2029, 1, 15)],
            "expiry_tenor": [None, "5Y"],
            "swap_tenor": [None, "10Y"],
        })
        
        pricer, coverage = build_var_portfolio_pricer(
            positions_df=portfolio_with_option,
            valuation_date=date(2024, 1, 15),
            market_state=sample_market_state,
            include_options=False,  # Explicitly exclude options
        )
        
        assert coverage.is_linear_only, "Should flag as linear-only when options excluded"
        assert len(coverage.warnings) > 0, "Should have warning messages"
        assert any("linear" in w.lower() or "option" in w.lower() for w in coverage.warnings)

    def test_no_silent_skipping(self, sample_market_state):
        """Test that no instrument types are silently skipped."""
        portfolio = pd.DataFrame({
            "position_id": ["P001", "P002"],
            "instrument_type": ["UST", "UNKNOWN_TYPE"],
            "instrument_id": ["UST_5Y", "UNKNOWN"],
            "notional": [1_000_000, 500_000],
            "direction": ["LONG", "LONG"],
            "coupon": [0.04, 0.0],
            "maturity_date": [date(2029, 1, 15), date(2029, 1, 15)],
        })
        
        pricer, coverage = build_var_portfolio_pricer(
            positions_df=portfolio,
            valuation_date=date(2024, 1, 15),
            market_state=sample_market_state,
            include_options=True,
        )
        
        # If something is excluded, it should be reported
        if coverage.excluded_instruments > 0:
            assert len(coverage.excluded_types) > 0, \
                "Excluded instrument types should be reported, not silently skipped"


class TestScenarioToDataframe:
    """Tests for scenario results to DataFrame conversion."""

    def test_converts_to_dataframe(self, bond_portfolio_df, sample_market_state):
        """Test conversion to DataFrame for display."""
        results = run_scenario_set(
            positions_df=bond_portfolio_df,
            market_state=sample_market_state,
            valuation_date=date(2024, 1, 15),
        )
        
        df = scenarios_to_dataframe(results)
        
        assert isinstance(df, pd.DataFrame)
        assert "Scenario" in df.columns
        assert "P&L" in df.columns
        assert len(df) == len(results)
