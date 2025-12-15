"""
Unit tests for VaR module.
"""

from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest

from rateslib.curves import Curve, create_flat_curve
from rateslib.var import (
    HistoricalSimulation,
    MonteCarloVaR,
    ScenarioEngine,
    Scenario,
    STANDARD_SCENARIOS,
)


@pytest.fixture
def sample_curve():
    """Create sample discount curve."""
    anchor_date = date(2024, 1, 15)
    return create_flat_curve(anchor_date, rate=0.05, max_tenor_years=30.0)


@pytest.fixture
def historical_data():
    """Create sample historical rate data in wide format."""
    dates = pd.date_range('2023-10-15', periods=60, freq='B')
    
    np.random.seed(42)  # For reproducibility
    
    # Create rate data with some correlation - wide format (date as index, tenors as columns)
    base_rates = {'3M': 0.053, '1Y': 0.050, '2Y': 0.048, '5Y': 0.045, '10Y': 0.042}
    
    data = {}
    for tenor, base_rate in base_rates.items():
        changes = np.random.normal(0, 0.001, len(dates))  # ~10bp daily vol
        rates = base_rate + np.cumsum(changes)
        data[tenor] = rates
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'date'
    return df


class TestHistoricalSimulation:
    """Tests for Historical Simulation VaR."""
    
    def test_var_calculation(self, sample_curve, historical_data):
        """Test basic VaR calculation."""
        def simple_pricer(curve):
            anchor = curve.anchor_date
            return curve.discount_factor(anchor + timedelta(days=365)) * 1_000_000
        
        sim = HistoricalSimulation(
            base_curve=sample_curve,
            historical_data=historical_data,
            pricer_func=simple_pricer
        )
        
        result = sim.run_simulation()
        
        # VaR should be positive (reported as loss)
        assert result.var_95 > 0
        assert result.var_99 > 0
        
        # 99% VaR should be higher than 95%
        assert result.var_99 >= result.var_95
    
    def test_es_higher_than_var(self, sample_curve, historical_data):
        """Test ES is higher than VaR."""
        def simple_pricer(curve):
            anchor = curve.anchor_date
            return curve.discount_factor(anchor + timedelta(days=365)) * 1_000_000
        
        sim = HistoricalSimulation(
            base_curve=sample_curve,
            historical_data=historical_data,
            pricer_func=simple_pricer
        )
        
        result = sim.run_simulation()
        
        # ES should be at least as high as VaR
        assert result.es_95 >= result.var_95
        assert result.es_99 >= result.var_99


class TestMonteCarloVaR:
    """Tests for Monte Carlo VaR."""
    
    def test_mc_var(self, sample_curve, historical_data):
        """Test Monte Carlo VaR."""
        def simple_pricer(curve):
            anchor = curve.anchor_date
            return curve.discount_factor(anchor + timedelta(days=365)) * 1_000_000
        
        mc = MonteCarloVaR(
            base_curve=sample_curve,
            historical_data=historical_data,
            pricer_func=simple_pricer
        )
        
        result = mc.run_simulation(num_paths=1000)
        
        assert result.var_95 > 0
        assert result.var_99 > 0
        assert result.num_paths == 1000
    
    def test_delta_normal_var(self, sample_curve, historical_data):
        """Test delta-normal VaR."""
        def simple_pricer(curve):
            anchor = curve.anchor_date
            return curve.discount_factor(anchor + timedelta(days=365)) * 1_000_000
        
        mc = MonteCarloVaR(
            base_curve=sample_curve,
            historical_data=historical_data,
            pricer_func=simple_pricer
        )
        
        # run_delta_normal_var computes key-rate DV01 internally
        var_95, var_99 = mc.run_delta_normal_var()
        
        assert var_95 > 0
        assert var_99 > 0
        assert var_99 > var_95  # 99% VaR should be higher


class TestScenarioEngine:
    """Tests for scenario analysis."""
    
    def test_run_single_scenario(self, sample_curve):
        """Test running a single scenario."""
        def simple_pricer(curve):
            anchor = curve.anchor_date
            # Simple bond-like payoff
            pv = 0
            for i in range(1, 11):
                d = anchor + timedelta(days=182*i)
                pv += curve.discount_factor(d) * 25
            pv += curve.discount_factor(anchor + timedelta(days=1825)) * 1000
            return pv
        
        engine = ScenarioEngine(sample_curve, simple_pricer)
        
        scenario = STANDARD_SCENARIOS['parallel_up_100']
        result = engine.run_scenario(scenario)
        
        # Parallel up should hurt bond value
        assert result.pnl < 0
        assert result.scenario_pv < result.base_pv
    
    def test_parallel_down_increases_value(self, sample_curve):
        """Test that parallel down increases bond value."""
        def bond_pricer(curve):
            anchor = curve.anchor_date
            pv = 0
            for i in range(1, 21):  # 10 year semi-annual
                d = anchor + timedelta(days=182*i)
                pv += curve.discount_factor(d) * 20  # $20 coupon
            pv += curve.discount_factor(anchor + timedelta(days=3650)) * 1000
            return pv
        
        engine = ScenarioEngine(sample_curve, bond_pricer)
        
        scenario = STANDARD_SCENARIOS['parallel_down_100']
        result = engine.run_scenario(scenario)
        
        # Parallel down should help bond value
        assert result.pnl > 0
    
    def test_standard_scenarios_exist(self):
        """Test that standard scenarios are defined."""
        required = [
            'parallel_up_100',
            'parallel_down_100',
            'steepener_2s10s',
            'flattener_2s10s',
        ]
        
        for name in required:
            assert name in STANDARD_SCENARIOS
            assert isinstance(STANDARD_SCENARIOS[name], Scenario)
    
    def test_custom_scenario(self, sample_curve):
        """Test creating and running a custom scenario."""
        custom = Scenario(
            name="Custom Test",
            description="Test scenario",
            bump_profile={'2Y': 50, '5Y': 25, '10Y': 0}
        )
        
        def simple_pricer(curve):
            anchor = curve.anchor_date
            return curve.discount_factor(anchor + timedelta(days=1825)) * 1_000_000
        
        engine = ScenarioEngine(sample_curve, simple_pricer)
        result = engine.run_scenario(custom)
        
        assert result.scenario.name == "Custom Test"
        assert result.pnl != 0
