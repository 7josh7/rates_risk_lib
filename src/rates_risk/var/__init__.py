"""
VaR package - Value at Risk and Expected Shortfall.

Provides:
- Historical Simulation VaR/ES
- Monte Carlo VaR/ES
- Stressed VaR
- Scenario analysis
"""

from .historical import (
    HistoricalSimulation,
    HistoricalVaRResult,
    compute_historical_var,
    compute_historical_es,
)
from .monte_carlo import (
    MonteCarloVaR,
    MonteCarloResult,
    compute_mc_var,
)
from .stress import (
    StressedVaR,
    StressResult,
)
from .scenarios import (
    ScenarioEngine,
    Scenario,
    ScenarioResult,
    STANDARD_SCENARIOS,
    create_custom_scenario,
    apply_market_scenario,
    apply_named_market_regime,
    SabrShock,
    SABR_STRESS_REGIMES,
    PortfolioScenarioResult,
    run_scenario_set,
    run_single_scenario,
    scenarios_to_dataframe,
)

__all__ = [
    "HistoricalSimulation",
    "HistoricalVaRResult",
    "compute_historical_var",
    "compute_historical_es",
    "MonteCarloVaR",
    "MonteCarloResult",
    "compute_mc_var",
    "StressedVaR",
    "StressResult",
    "ScenarioEngine",
    "Scenario",
    "ScenarioResult",
    "STANDARD_SCENARIOS",
    "create_custom_scenario",
    "apply_market_scenario",
    "apply_named_market_regime",
    "SabrShock",
    "SABR_STRESS_REGIMES",
    "PortfolioScenarioResult",
    "run_scenario_set",
    "run_single_scenario",
    "scenarios_to_dataframe",
]
