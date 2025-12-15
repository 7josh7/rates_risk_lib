"""
Scenario analysis for yield curves.

Implements predefined and custom scenarios:
- Parallel shifts (+/- 100bp)
- Twist (short/long opposing moves)
- Steepener/Flattener (2s10s)
- Historical scenarios (specific dates)

Scenarios are defined as rate changes at each key tenor.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..curves.curve import Curve
from ..dates import DateUtils
from ..risk.bumping import BumpEngine


@dataclass
class Scenario:
    """
    Definition of a curve scenario.
    
    Attributes:
        name: Scenario name
        description: Description of the scenario
        bump_profile: Dict of {tenor: bump_in_bp}
    """
    name: str
    description: str
    bump_profile: Dict[str, float]
    
    def get_parallel_equivalent(self) -> float:
        """Get average bump (parallel equivalent)."""
        if not self.bump_profile:
            return 0.0
        return np.mean(list(self.bump_profile.values()))


@dataclass
class ScenarioResult:
    """
    Result of running a scenario.
    
    Attributes:
        scenario: The scenario that was run
        base_pv: PV before scenario
        scenario_pv: PV after scenario
        pnl: P&L from scenario
        contributors: Top contributors to P&L (by key rate)
    """
    scenario: Scenario
    base_pv: float
    scenario_pv: float
    pnl: float
    contributors: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "scenario_name": self.scenario.name,
            "description": self.scenario.description,
            "base_pv": self.base_pv,
            "scenario_pv": self.scenario_pv,
            "pnl": self.pnl,
            "contributors": self.contributors
        }


# Standard scenario definitions
STANDARD_SCENARIOS = {
    "parallel_up_100": Scenario(
        name="Parallel +100bp",
        description="Parallel upward shift of 100 basis points",
        bump_profile={
            "3M": 100, "6M": 100, "1Y": 100, "2Y": 100, "3Y": 100,
            "5Y": 100, "7Y": 100, "10Y": 100, "20Y": 100, "30Y": 100
        }
    ),
    "parallel_down_100": Scenario(
        name="Parallel -100bp",
        description="Parallel downward shift of 100 basis points",
        bump_profile={
            "3M": -100, "6M": -100, "1Y": -100, "2Y": -100, "3Y": -100,
            "5Y": -100, "7Y": -100, "10Y": -100, "20Y": -100, "30Y": -100
        }
    ),
    "steepener_2s10s": Scenario(
        name="2s10s Steepener",
        description="2Y -25bp, 10Y +25bp (50bp steepening)",
        bump_profile={
            "3M": -25, "6M": -25, "1Y": -25, "2Y": -25, "3Y": -12.5,
            "5Y": 0, "7Y": 12.5, "10Y": 25, "20Y": 25, "30Y": 25
        }
    ),
    "flattener_2s10s": Scenario(
        name="2s10s Flattener",
        description="2Y +25bp, 10Y -25bp (50bp flattening)",
        bump_profile={
            "3M": 25, "6M": 25, "1Y": 25, "2Y": 25, "3Y": 12.5,
            "5Y": 0, "7Y": -12.5, "10Y": -25, "20Y": -25, "30Y": -25
        }
    ),
    "twist_5y": Scenario(
        name="Twist around 5Y",
        description="Short end -50bp, long end +50bp, pivot at 5Y",
        bump_profile={
            "3M": -50, "6M": -50, "1Y": -40, "2Y": -30, "3Y": -15,
            "5Y": 0, "7Y": 15, "10Y": 30, "20Y": 40, "30Y": 50
        }
    ),
    "front_end_sell_off": Scenario(
        name="Front-end Sell-off",
        description="Short rates up 75bp, long rates up 25bp",
        bump_profile={
            "3M": 75, "6M": 75, "1Y": 65, "2Y": 50, "3Y": 40,
            "5Y": 35, "7Y": 30, "10Y": 25, "20Y": 25, "30Y": 25
        }
    ),
    "long_end_rally": Scenario(
        name="Long-end Rally",
        description="Long rates down 50bp, short rates down 10bp",
        bump_profile={
            "3M": -10, "6M": -10, "1Y": -15, "2Y": -20, "3Y": -25,
            "5Y": -35, "7Y": -40, "10Y": -50, "20Y": -50, "30Y": -50
        }
    ),
    "bear_flattener": Scenario(
        name="Bear Flattener",
        description="All rates up, short end more (Fed hiking)",
        bump_profile={
            "3M": 100, "6M": 100, "1Y": 90, "2Y": 80, "3Y": 70,
            "5Y": 60, "7Y": 55, "10Y": 50, "20Y": 45, "30Y": 40
        }
    ),
    "bull_steepener": Scenario(
        name="Bull Steepener",
        description="All rates down, short end more (Fed cutting)",
        bump_profile={
            "3M": -100, "6M": -100, "1Y": -90, "2Y": -80, "3Y": -70,
            "5Y": -60, "7Y": -55, "10Y": -50, "20Y": -45, "30Y": -40
        }
    ),
}


class ScenarioEngine:
    """
    Engine for running scenario analysis.
    
    Applies predefined or custom scenarios to a portfolio
    and computes P&L impact.
    """
    
    def __init__(
        self,
        base_curve: Curve,
        pricer_func: Callable[[Curve], float],
        key_rate_dv01: Optional[Dict[str, float]] = None
    ):
        """
        Initialize scenario engine.
        
        Args:
            base_curve: Current yield curve
            pricer_func: Function that prices portfolio
            key_rate_dv01: Optional key-rate DV01s for contribution analysis
        """
        self.base_curve = base_curve
        self.pricer_func = pricer_func
        self.key_rate_dv01 = key_rate_dv01 or {}
        self.bump_engine = BumpEngine(base_curve)
    
    def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        """
        Run a single scenario.
        
        Args:
            scenario: Scenario to run
            
        Returns:
            ScenarioResult
        """
        # Base PV
        base_pv = self.pricer_func(self.base_curve)
        
        # Apply scenario bumps
        scenario_curve = self.bump_engine.custom_bump(scenario.bump_profile)
        
        # Scenario PV
        scenario_pv = self.pricer_func(scenario_curve)
        pnl = scenario_pv - base_pv
        
        # Calculate contributions by key rate
        contributors = {}
        if self.key_rate_dv01:
            for tenor, bump in scenario.bump_profile.items():
                dv01 = self.key_rate_dv01.get(tenor, 0)
                contribution = -dv01 * bump
                contributors[tenor] = contribution
        
        return ScenarioResult(
            scenario=scenario,
            base_pv=base_pv,
            scenario_pv=scenario_pv,
            pnl=pnl,
            contributors=contributors
        )
    
    def run_standard_scenarios(self) -> List[ScenarioResult]:
        """
        Run all standard scenarios.
        
        Returns:
            List of ScenarioResults
        """
        results = []
        for scenario in STANDARD_SCENARIOS.values():
            result = self.run_scenario(scenario)
            results.append(result)
        return results
    
    def run_historical_scenario(
        self,
        historical_data: pd.DataFrame,
        scenario_date: date,
        scenario_name: Optional[str] = None
    ) -> ScenarioResult:
        """
        Run scenario based on actual historical rate changes.
        
        Args:
            historical_data: Historical rate data
            scenario_date: Date of historical scenario
            scenario_name: Optional name for scenario
            
        Returns:
            ScenarioResult
        """
        df = historical_data.copy()
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            pivot = df.pivot_table(index='date', columns='tenor', values='rate')
        else:
            pivot = df
        
        # Get rate changes for scenario date
        changes = pivot.diff() * 10000  # Convert to bp
        
        target_date = pd.Timestamp(scenario_date)
        if target_date not in changes.index:
            # Find closest date
            closest_idx = np.argmin(np.abs(changes.index - target_date))
            target_date = changes.index[closest_idx]
        
        rate_changes = changes.loc[target_date]
        
        # Create scenario
        bump_profile = {}
        for tenor in rate_changes.index:
            if not pd.isna(rate_changes[tenor]):
                bump_profile[tenor] = rate_changes[tenor]
        
        scenario = Scenario(
            name=scenario_name or f"Historical {scenario_date}",
            description=f"Rate changes from {target_date.date()}",
            bump_profile=bump_profile
        )
        
        return self.run_scenario(scenario)
    
    def find_worst_historical_scenarios(
        self,
        historical_data: pd.DataFrame,
        n: int = 5
    ) -> List[ScenarioResult]:
        """
        Find the N worst historical scenarios.
        
        Args:
            historical_data: Historical data
            n: Number of worst scenarios to find
            
        Returns:
            List of ScenarioResults (worst first)
        """
        df = historical_data.copy()
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            pivot = df.pivot_table(index='date', columns='tenor', values='rate')
        else:
            pivot = df
        
        # Compute P&L for each historical day
        changes = pivot.diff() * 10000
        changes = changes.dropna()
        
        pnl_by_date = []
        
        for idx in changes.index:
            bump_profile = {t: changes.loc[idx, t] for t in changes.columns 
                          if not pd.isna(changes.loc[idx, t])}
            
            try:
                scenario_curve = self.bump_engine.custom_bump(bump_profile)
                base_pv = self.pricer_func(self.base_curve)
                scenario_pv = self.pricer_func(scenario_curve)
                pnl = scenario_pv - base_pv
                pnl_by_date.append((idx, pnl, bump_profile))
            except:
                continue
        
        # Sort by P&L (worst = most negative)
        pnl_by_date.sort(key=lambda x: x[1])
        
        # Create results for worst N
        results = []
        for i in range(min(n, len(pnl_by_date))):
            scenario_date, pnl, bump_profile = pnl_by_date[i]
            scenario = Scenario(
                name=f"Worst Day #{i+1}",
                description=f"Historical scenario from {scenario_date.date()}",
                bump_profile=bump_profile
            )
            
            result = ScenarioResult(
                scenario=scenario,
                base_pv=self.pricer_func(self.base_curve),
                scenario_pv=self.pricer_func(self.base_curve) + pnl,
                pnl=pnl
            )
            results.append(result)
        
        return results


def create_custom_scenario(
    name: str,
    description: str,
    **tenor_bumps: float
) -> Scenario:
    """
    Create a custom scenario.
    
    Args:
        name: Scenario name
        description: Description
        **tenor_bumps: Keyword args of tenor=bump_bp
        
    Returns:
        Scenario
        
    Example:
        scenario = create_custom_scenario(
            "My Scenario",
            "Custom steepener",
            **{"2Y": -50, "5Y": 0, "10Y": 50}
        )
    """
    return Scenario(
        name=name,
        description=description,
        bump_profile=tenor_bumps
    )


__all__ = [
    "Scenario",
    "ScenarioResult",
    "ScenarioEngine",
    "STANDARD_SCENARIOS",
    "create_custom_scenario",
]
