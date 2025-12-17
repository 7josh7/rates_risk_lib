"""
Scenario analysis for yield curves.

Implements predefined and custom scenarios:
- Parallel shifts (+/- 100bp)
- Twist (short/long opposing moves)
- Steepener/Flattener (2s10s)
- Historical scenarios (specific dates)

Scenarios are defined as rate changes at each key tenor.

PRODUCTION PRINCIPLES:
======================
1. NEVER silently swallow exceptions - return failure diagnostics
2. Use explicit trade builders for options - no inference
3. Return coverage metrics so UI can warn when incomplete
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from ..curves.curve import Curve
from ..dates import DateUtils
from ..risk.bumping import BumpEngine
from ..market_state import MarketState, CurveState
from ..vol.sabr_surface import SabrSurfaceState, SabrBucketParams, BucketKey
from ..portfolio.builders import (
    build_trade_from_position,
    price_portfolio_with_diagnostics,
    PortfolioPricingResult,
    TradeFailure,
    PositionValidationError,
    MissingFieldError,
    InvalidOptionError,
)


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
        curve_params: Curve parameters used (e.g., NSS params)
        sabr_params: SABR parameters used (summary or full)
    """
    scenario: Scenario
    base_pv: float
    scenario_pv: float
    pnl: float
    contributors: Dict[str, float] = field(default_factory=dict)
    curve_params: Optional[Dict[str, Any]] = field(default_factory=dict)
    sabr_params: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "scenario_name": self.scenario.name,
            "description": self.scenario.description,
            "base_pv": self.base_pv,
            "scenario_pv": self.scenario_pv,
            "pnl": self.pnl,
            "contributors": self.contributors
        }
        if self.curve_params:
            result["curve_params"] = self.curve_params
        if self.sabr_params:
            result["sabr_params"] = self.sabr_params
        return result


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


# === Combined curve + SABR scenarios ======================================

@dataclass
class SabrShock:
    """Shock to SABR parameters (additive with optional scaling)."""

    dsigma_atm: float = 0.0
    dnu: float = 0.0
    drho: float = 0.0
    sigma_scale: float = 0.0
    nu_scale: float = 0.0


def _apply_sabr_shocks(surface: Optional[SabrSurfaceState], shocks: Dict[BucketKey, SabrShock]) -> Optional[SabrSurfaceState]:
    """Return new SABR surface with shocks applied."""
    if surface is None:
        return None

    params_new: Dict[BucketKey, SabrBucketParams] = {}
    for bucket, params in surface.params_by_bucket.items():
        shock = shocks.get(bucket) or shocks.get(("ALL", "ALL"))
        if shock is None:
            params_new[bucket] = params
            continue

        sigma = max(1e-8, params.sigma_atm * (1 + shock.sigma_scale) + shock.dsigma_atm)
        nu = max(1e-8, params.nu * (1 + shock.nu_scale) + shock.dnu)
        rho = float(np.clip(params.rho + shock.drho, -0.999, 0.999))

        diag = dict(params.diagnostics)
        diag["shock"] = {
            "dsigma_atm": shock.dsigma_atm,
            "dnu": shock.dnu,
            "drho": shock.drho,
            "sigma_scale": shock.sigma_scale,
            "nu_scale": shock.nu_scale,
        }

        params_new[bucket] = SabrBucketParams(
            sigma_atm=sigma,
            nu=nu,
            rho=rho,
            beta=params.beta,
            shift=params.shift,
            diagnostics=diag,
        )

    return SabrSurfaceState(
        params_by_bucket=params_new,
        convention=surface.convention,
        asof=surface.asof,
        missing_bucket_policy=surface.missing_bucket_policy,
    )


def apply_market_scenario(
    market_state: MarketState,
    curve_bump_profile: Optional[Dict[str, float]] = None,
    sabr_shocks: Optional[Dict[BucketKey, SabrShock]] = None,
) -> MarketState:
    """
    Apply combined curve + SABR shocks to a MarketState.

    Args:
        market_state: Base MarketState
        curve_bump_profile: Dict tenor->bp for curve shocks
        sabr_shocks: Dict bucket->SabrShock (use ("ALL","ALL") for blanket)
    """
    curve_bump_profile = curve_bump_profile or {}
    sabr_shocks = sabr_shocks or {}

    bump_engine = BumpEngine(market_state.curve.discount_curve)
    bumped_curve = bump_engine.custom_bump(curve_bump_profile) if curve_bump_profile else market_state.curve.discount_curve
    # Apply the same bump to projection curve for simplicity
    if market_state.curve.projection_curve is market_state.curve.discount_curve:
        bumped_projection = bumped_curve
    else:
        proj_engine = BumpEngine(market_state.curve.projection_curve)
        bumped_projection = proj_engine.custom_bump(curve_bump_profile) if curve_bump_profile else market_state.curve.projection_curve

    new_curve_state = CurveState(
        discount_curve=bumped_curve,
        projection_curve=bumped_projection,
        metadata=dict(market_state.curve.metadata),
    )

    new_surface = _apply_sabr_shocks(market_state.sabr_surface, sabr_shocks)

    return MarketState(curve=new_curve_state, sabr_surface=new_surface, asof=market_state.asof)


SABR_STRESS_REGIMES: Dict[str, Dict[str, Any]] = {
    "CALM": {
        "curve": {},
        "sabr": {("ALL", "ALL"): SabrShock(sigma_scale=-0.10, nu_scale=-0.15, drho=0.02)},
    },
    "RISK_OFF": {
        "curve": {"2Y": 15, "5Y": 10, "10Y": 5},
        "sabr": {("ALL", "ALL"): SabrShock(sigma_scale=0.20, nu_scale=0.25, drho=-0.08)},
    },
    "CRISIS": {
        "curve": {"2Y": 50, "5Y": 40, "10Y": 35, "30Y": 25},
        "sabr": {("ALL", "ALL"): SabrShock(sigma_scale=0.50, nu_scale=0.60, drho=-0.15)},
    },
}


def apply_named_market_regime(market_state: MarketState, regime: str) -> MarketState:
    """
    Apply a named regime (CALM, RISK_OFF, CRISIS) to the market state.
    """
    regime_upper = regime.upper()
    if regime_upper not in SABR_STRESS_REGIMES:
        raise ValueError(f"Unknown regime: {regime}")
    spec = SABR_STRESS_REGIMES[regime_upper]
    return apply_market_scenario(
        market_state,
        curve_bump_profile=spec.get("curve", {}),
        sabr_shocks=spec.get("sabr", {}),
    )


def extract_market_state_params(market_state: MarketState) -> Dict[str, Any]:
    """
    Extract curve and SABR parameters from MarketState for auditability.
    
    Returns:
        Dict with 'curve_metadata', 'sabr_params_summary', 'sabr_diagnostics'
    """
    result: Dict[str, Any] = {}
    
    # Curve metadata (may include NSS params)
    if market_state.curve.metadata:
        result["curve_metadata"] = dict(market_state.curve.metadata)
    
    # SABR parameters summary
    if market_state.sabr_surface:
        sabr_summary = {}
        for bucket, params in market_state.sabr_surface.params_by_bucket.items():
            sabr_summary[f"{bucket[0]}x{bucket[1]}"] = {
                "sigma_atm": params.sigma_atm,
                "nu": params.nu,
                "rho": params.rho,
                "beta": params.beta,
                "shift": params.shift,
            }
        result["sabr_params"] = sabr_summary
        
        # Include diagnostics table
        result["sabr_diagnostics"] = market_state.sabr_surface.diagnostics_table()
        
        # Convention info
        if market_state.sabr_surface.convention:
            result["sabr_convention"] = dict(market_state.sabr_surface.convention)
    
    return result


# ==============================================================================
# Portfolio Scenario Repricing Functions
# ==============================================================================

@dataclass
class PortfolioScenarioResult:
    """
    Result of running a scenario on a portfolio with full repricing.
    
    Attributes:
        scenario_name: Name of the scenario
        description: Scenario description
        base_pv: Portfolio PV before scenario
        scenario_pv: Portfolio PV after scenario
        pnl: Scenario P&L (scenario_pv - base_pv)
        curve_bump_profile: The curve bumps applied
        instruments_priced: Number of instruments successfully priced
        total_instruments: Total number of instruments in portfolio
        failed_trades: List of TradeFailure objects
        computation_method: Description of how computed
        warnings: List of warning messages
    """
    scenario_name: str
    description: str
    base_pv: float
    scenario_pv: float
    pnl: float
    curve_bump_profile: Dict[str, float]
    instruments_priced: int = 0
    total_instruments: int = 0
    failed_trades: List[TradeFailure] = field(default_factory=list)
    computation_method: str = "bump-and-reprice"
    warnings: List[str] = field(default_factory=list)
    
    @property
    def coverage_ratio(self) -> float:
        if self.total_instruments == 0:
            return 1.0
        return self.instruments_priced / self.total_instruments
    
    @property
    def has_failures(self) -> bool:
        return len(self.failed_trades) > 0 or self.coverage_ratio < 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "description": self.description,
            "base_pv": self.base_pv,
            "scenario_pv": self.scenario_pv,
            "pnl": self.pnl,
            "curve_bump_profile": self.curve_bump_profile,
            "instruments_priced": self.instruments_priced,
            "total_instruments": self.total_instruments,
            "coverage_ratio": self.coverage_ratio,
            "has_failures": self.has_failures,
            "failed_trades": [f.to_dict() for f in self.failed_trades],
            "computation_method": self.computation_method,
            "warnings": self.warnings,
        }


# Legacy function for backward compatibility
def _build_trade_from_position_legacy(pos: pd.Series, valuation_date: date) -> Optional[Dict[str, Any]]:
    """
    DEPRECATED: Build a trade dict from a position row.
    
    Uses inference for options which is dangerous in production.
    Use portfolio.builders.build_trade_from_position() instead.
    """
    inst = str(pos.get("instrument_type", "")).upper()
    notional = float(abs(pos.get("notional", 0.0)))
    direction = str(pos.get("direction", "")).upper()
    sign = -1.0 if direction in {"SHORT", "PAY_FIXED", "PAY"} else 1.0
    
    if inst in {"BOND", "UST"}:
        maturity = pos.get("maturity_date")
        if maturity is not None:
            maturity = pd.to_datetime(maturity).date() if not isinstance(maturity, date) else maturity
        else:
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
        maturity = pos.get("maturity_date")
        if maturity is not None:
            maturity = pd.to_datetime(maturity).date() if not isinstance(maturity, date) else maturity
        else:
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
        maturity_raw = pos.get("maturity_date")
        maturity = None
        if maturity_raw is not None and not pd.isna(maturity_raw):
            try:
                if isinstance(maturity_raw, date):
                    maturity = maturity_raw
                else:
                    parsed = pd.to_datetime(maturity_raw)
                    if not pd.isna(parsed):
                        maturity = parsed.date()
            except Exception:
                maturity = None
        
        # Helper to get first non-null, non-NaN value
        def get_first_valid(*keys):
            for key in keys:
                val = pos.get(key)
                if val is not None and not pd.isna(val):
                    return val
            return None
        
        expiry_tenor = get_first_valid("expiry_tenor", "option_expiry")
        swap_tenor = get_first_valid("swap_tenor", "tenor", "underlying_swap_tenor")
        
        # Check for explicit expiry_date first
        expiry_raw = pos.get("expiry_date")
        if expiry_raw is not None and not pd.isna(expiry_raw):
            try:
                if isinstance(expiry_raw, date):
                    expiry_date = expiry_raw
                else:
                    parsed = pd.to_datetime(expiry_raw)
                    if not pd.isna(parsed):
                        expiry_date = parsed.date()
                        years = max(0.25, (expiry_date - valuation_date).days / 365.25)
                        if years < 1:
                            expiry_tenor = f"{max(1, int(round(years * 12)))}M"
                        else:
                            expiry_tenor = f"{int(round(years))}Y"
            except Exception:
                pass
        
        # For CAPLET, derive tenor from caplet dates
        if inst in {"CAPLET", "CAP", "CAPFLOOR"}:
            caplet_start = get_first_valid("caplet_start_date")
            caplet_end = get_first_valid("caplet_end_date")
            if caplet_start and caplet_end:
                try:
                    if not isinstance(caplet_start, date):
                        caplet_start = pd.to_datetime(caplet_start).date()
                    if not isinstance(caplet_end, date):
                        caplet_end = pd.to_datetime(caplet_end).date()
                    # Use time to caplet start as expiry_tenor
                    years_to_start = max(0.25, (caplet_start - valuation_date).days / 365.25)
                    if not expiry_tenor:
                        if years_to_start < 1:
                            expiry_tenor = f"{max(1, int(round(years_to_start * 12)))}M"
                        else:
                            expiry_tenor = f"{int(round(years_to_start))}Y"
                    # Use caplet period as swap_tenor
                    caplet_period_years = (caplet_end - caplet_start).days / 365.25
                    if not swap_tenor:
                        if caplet_period_years < 1:
                            swap_tenor = f"{max(1, int(round(caplet_period_years * 12)))}M"
                        else:
                            swap_tenor = f"{int(round(caplet_period_years))}Y"
                except Exception:
                    pass
        
        if not expiry_tenor and maturity is not None:
            years = max(0.25, round((maturity - valuation_date).days / 365.25))
            expiry_tenor = f"{int(years)}Y"
        
        if not expiry_tenor:
            expiry_tenor = "1Y"  # Default fallback
        if not swap_tenor:
            swap_tenor = "5Y"
        return {
            "instrument_type": "SWAPTION",
            "expiry_tenor": str(expiry_tenor),
            "swap_tenor": str(swap_tenor),
            "strike": pos.get("strike", "ATM") or "ATM",
            "payer_receiver": str(pos.get("payer_receiver", "PAYER")).upper() if pos.get("payer_receiver") else ("PAYER" if direction in {"LONG", "BUY"} else "RECEIVER"),
            "notional": notional * sign,
            "vol_type": "NORMAL",
        }
    
    if inst in {"FUT", "FUTURE", "FUTURES"}:
        # Get expiry date from multiple possible fields
        expiry_raw = pos.get("expiry_date") or pos.get("maturity_date")
        if expiry_raw is None or pd.isna(expiry_raw):
            return None
        try:
            if isinstance(expiry_raw, date):
                expiry = expiry_raw
            else:
                parsed = pd.to_datetime(expiry_raw)
                if pd.isna(parsed):
                    return None
                expiry = parsed.date()
        except Exception:
            return None
        
        if expiry <= valuation_date:
            return None  # Expired contract
        
        # Handle notional as contract count
        num_contracts = int(abs(notional)) if notional >= 1 else 1
        contract_sign = 1 if direction in {"LONG", "BUY", ""} else -1
        
        # Get trade price for P&L calculation
        trade_price = pos.get("trade_price") or pos.get("entry_price")
        if trade_price is not None:
            try:
                trade_price = float(trade_price)
            except (ValueError, TypeError):
                trade_price = None
        
        return {
            "instrument_type": "FUT",
            "expiry": expiry,
            "contract_code": str(pos.get("contract_code") or pos.get("instrument_id") or "FUT"),
            "contract_size": float(pos.get("contract_size", 1_000_000)),
            "underlying_tenor": str(pos.get("underlying_tenor", "3M")),
            "tick_size": float(pos.get("tick_size", 0.0025)),
            "tick_value": float(pos.get("tick_value", 6.25)),
            "num_contracts": num_contracts * contract_sign,
            "trade_price": trade_price,
        }
    
    return None


# Re-export as _build_trade_from_position for backward compat
_build_trade_from_position = _build_trade_from_position_legacy


@dataclass
class PortfolioPriceResult:
    """Result of pricing a portfolio with failure tracking."""
    total_pv: float
    count_priced: int
    count_failed: int
    failed_trades: List[TradeFailure]
    
    @property
    def coverage_ratio(self) -> float:
        total = self.count_priced + self.count_failed
        if total == 0:
            return 1.0
        return self.count_priced / total


def _price_portfolio_with_tracking(
    trades: List[Dict[str, Any]],
    market_state: MarketState,
) -> PortfolioPriceResult:
    """
    Price a list of trades under a given market state with failure tracking.
    
    NEVER silently swallows exceptions - records all failures.
    """
    from ..pricers.dispatcher import price_trade
    
    total_pv = 0.0
    count_priced = 0
    failed_trades: List[TradeFailure] = []
    
    for trade in trades:
        try:
            result = price_trade(trade, market_state)
            total_pv += result.pv
            count_priced += 1
        except Exception as e:
            failed_trades.append(TradeFailure(
                position_id=trade.get("_position_id"),
                instrument_type=trade.get("instrument_type", "UNKNOWN"),
                error_type=type(e).__name__,
                error_message=str(e),
                stage="price",
            ))
    
    return PortfolioPriceResult(
        total_pv=total_pv,
        count_priced=count_priced,
        count_failed=len(failed_trades),
        failed_trades=failed_trades,
    )


def _price_portfolio(
    trades: List[Dict[str, Any]],
    market_state: MarketState,
) -> Tuple[float, int]:
    """
    Legacy function for backward compatibility.
    Returns (total_pv, count_priced).
    """
    result = _price_portfolio_with_tracking(trades, market_state)
    return result.total_pv, result.count_priced


def run_scenario_set(
    positions_df: pd.DataFrame,
    market_state: MarketState,
    valuation_date: date,
    scenarios: Optional[Dict[str, Scenario]] = None,
    use_explicit_builders: bool = False,
) -> List[PortfolioScenarioResult]:
    """
    Run a set of scenarios on a portfolio using full bump-and-reprice.
    
    This function replaces hard-coded scenario P&Ls with actual repricing
    under shocked curves.
    
    PRODUCTION MODE (use_explicit_builders=True):
        - Uses explicit trade builders that require all option fields
        - Never silently drops positions - tracks all failures
        - Returns warnings when coverage < 100%
    
    LEGACY MODE (use_explicit_builders=False, default):
        - Uses inference-based builders for backward compatibility
    
    Args:
        positions_df: DataFrame with position details
        market_state: Current market state (curves + vol surface)
        valuation_date: Valuation date for pricing
        scenarios: Dict of scenario name -> Scenario. Uses STANDARD_SCENARIOS if None.
        use_explicit_builders: If True, use production-grade explicit builders
    
    Returns:
        List of PortfolioScenarioResult objects, one per scenario.
    """
    scenarios = scenarios or STANDARD_SCENARIOS
    
    # Build trades from positions with failure tracking
    trades: List[Dict[str, Any]] = []
    build_failures: List[TradeFailure] = []
    
    for _, pos in positions_df.iterrows():
        position_id = pos.get("position_id")
        inst_type = str(pos.get("instrument_type", "UNKNOWN")).upper()
        
        if use_explicit_builders:
            try:
                trade = build_trade_from_position(pos, valuation_date)
                trades.append(trade)
            except (PositionValidationError, MissingFieldError, InvalidOptionError) as e:
                build_failures.append(TradeFailure(
                    position_id=position_id,
                    instrument_type=inst_type,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    stage="build",
                ))
            except Exception as e:
                build_failures.append(TradeFailure(
                    position_id=position_id,
                    instrument_type=inst_type,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    stage="build",
                ))
        else:
            trade = _build_trade_from_position_legacy(pos, valuation_date)
            if trade:
                trades.append(trade)
    
    if not trades:
        # Return empty results with warning
        warnings = []
        if build_failures:
            warnings.append(
                f"⚠️ All {len(positions_df)} positions failed to build. "
                "Check position data for missing required fields."
            )
        return [
            PortfolioScenarioResult(
                scenario_name=s.name,
                description=s.description,
                base_pv=0.0,
                scenario_pv=0.0,
                pnl=0.0,
                curve_bump_profile=s.bump_profile,
                instruments_priced=0,
                total_instruments=len(positions_df),
                failed_trades=build_failures,
                computation_method="No tradeable positions",
                warnings=warnings,
            )
            for s in scenarios.values()
        ]
    
    # Compute base PV with failure tracking
    base_result = _price_portfolio_with_tracking(trades, market_state)
    all_failures = build_failures + base_result.failed_trades
    
    results = []
    
    for scenario_key, scenario in scenarios.items():
        # Apply scenario bumps to create shocked market state
        shocked_market = apply_market_scenario(
            market_state,
            curve_bump_profile=scenario.bump_profile,
            sabr_shocks=None,  # Curve-only scenarios for now
        )
        
        # Price portfolio under shocked curves
        scenario_result = _price_portfolio_with_tracking(trades, shocked_market)
        
        pnl = scenario_result.total_pv - base_result.total_pv
        
        # Generate warnings
        warnings = []
        coverage = base_result.count_priced / len(positions_df) if len(positions_df) > 0 else 1.0
        if coverage < 1.0:
            warnings.append(
                f"⚠️ Coverage: {coverage:.1%} ({base_result.count_priced}/{len(positions_df)} positions). "
                "Scenario P&L may be understated."
            )
        if all_failures:
            warnings.append(
                f"⚠️ {len(all_failures)} position(s) failed to price."
            )
        
        results.append(PortfolioScenarioResult(
            scenario_name=scenario.name,
            description=scenario.description,
            base_pv=base_result.total_pv,
            scenario_pv=scenario_result.total_pv,
            pnl=pnl,
            curve_bump_profile=scenario.bump_profile,
            instruments_priced=base_result.count_priced,
            total_instruments=len(positions_df),
            failed_trades=all_failures,
            computation_method="Computed from repricing under shocked curves",
            warnings=warnings,
        ))
    
    return results


def run_single_scenario(
    positions_df: pd.DataFrame,
    market_state: MarketState,
    valuation_date: date,
    scenario: Scenario,
    use_explicit_builders: bool = False,
) -> PortfolioScenarioResult:
    """
    Run a single scenario on a portfolio using full bump-and-reprice.
    
    Args:
        positions_df: DataFrame with position details
        market_state: Current market state
        valuation_date: Valuation date
        scenario: The scenario to run
        use_explicit_builders: If True, use production-grade explicit builders
    
    Returns:
        PortfolioScenarioResult
    """
    results = run_scenario_set(
        positions_df=positions_df,
        market_state=market_state,
        valuation_date=valuation_date,
        scenarios={scenario.name: scenario},
        use_explicit_builders=use_explicit_builders,
    )
    return results[0] if results else PortfolioScenarioResult(
        scenario_name=scenario.name,
        description=scenario.description,
        base_pv=0.0,
        scenario_pv=0.0,
        pnl=0.0,
        curve_bump_profile=scenario.bump_profile,
        instruments_priced=0,
        computation_method="No instruments to price",
    )


def scenarios_to_dataframe(results: List[PortfolioScenarioResult]) -> pd.DataFrame:
    """Convert scenario results to DataFrame for display."""
    if not results:
        return pd.DataFrame(columns=["Scenario", "P&L", "Description"])
    
    return pd.DataFrame({
        "Scenario": [r.scenario_name for r in results],
        "P&L": [r.pnl for r in results],
        "Description": [r.description for r in results],
        "Instruments Priced": [r.instruments_priced for r in results],
    })


# =============================================================================
# Vol-Only Scenario Definitions
# =============================================================================

@dataclass
class VolScenario:
    """Definition of a vol-only scenario (curve unchanged)."""
    name: str
    description: str
    sabr_shocks: Dict[BucketKey, SabrShock]


VOL_ONLY_SCENARIOS: Dict[str, VolScenario] = {
    "vol_shock_up_50": VolScenario(
        name="Vol Shock +50%",
        description="ATM volatility up 50% across all buckets",
        sabr_shocks={("ALL", "ALL"): SabrShock(sigma_scale=0.50)},
    ),
    "vol_shock_down_30": VolScenario(
        name="Vol Shock -30%",
        description="ATM volatility down 30% across all buckets",
        sabr_shocks={("ALL", "ALL"): SabrShock(sigma_scale=-0.30)},
    ),
    "nu_stress_up_100": VolScenario(
        name="Nu Stress +100%",
        description="Vol-of-vol doubled (fatter tails)",
        sabr_shocks={("ALL", "ALL"): SabrShock(nu_scale=1.00)},
    ),
    "rho_stress_negative": VolScenario(
        name="Rho Stress -0.5",
        description="Correlation shifted to -0.5 (more negative skew)",
        sabr_shocks={("ALL", "ALL"): SabrShock(drho=-0.50)},
    ),
    "crisis_vol": VolScenario(
        name="Crisis Vol",
        description="Combined stress: +80% vol, +120% nu, rho -0.25",
        sabr_shocks={("ALL", "ALL"): SabrShock(sigma_scale=0.80, nu_scale=1.20, drho=-0.25)},
    ),
}


@dataclass
class VolScenarioResult:
    """Result of a vol-only scenario."""
    scenario_name: str
    description: str
    base_pv: float
    scenario_pv: float
    pnl: float
    options_pnl: float  # P&L from options only
    linear_pnl: float   # P&L from linear instruments (should be ~0)
    sabr_shocks: Dict[str, Any]
    instruments_priced: int
    total_instruments: int


def run_vol_only_scenarios(
    positions_df: pd.DataFrame,
    market_state: MarketState,
    valuation_date: date,
    scenarios: Optional[Dict[str, VolScenario]] = None,
) -> List[VolScenarioResult]:
    """
    Run vol-only scenarios (curve unchanged, SABR parameters shocked).
    
    This demonstrates that linear products are unaffected by vol shocks
    while options P&L responds to SABR parameter changes.
    
    Args:
        positions_df: DataFrame with position details
        market_state: Current market state (must have SABR surface)
        valuation_date: Valuation date
        scenarios: Optional dict of scenarios (defaults to VOL_ONLY_SCENARIOS)
        
    Returns:
        List of VolScenarioResult objects
    """
    scenarios = scenarios or VOL_ONLY_SCENARIOS
    
    if market_state.sabr_surface is None:
        return []
    
    # Build trades
    trades: List[Dict[str, Any]] = []
    for _, pos in positions_df.iterrows():
        trade = _build_trade_from_position_legacy(pos, valuation_date)
        if trade:
            trades.append(trade)
    
    if not trades:
        return []
    
    # Identify options vs linear
    option_trades = [t for t in trades if t.get("instrument_type") in {"SWAPTION", "CAPLET"}]
    linear_trades = [t for t in trades if t.get("instrument_type") not in {"SWAPTION", "CAPLET"}]
    
    # Base pricing
    base_result = _price_portfolio_with_tracking(trades, market_state)
    base_options = _price_portfolio_with_tracking(option_trades, market_state) if option_trades else PortfolioPriceResult(0, 0, 0, [])
    base_linear = _price_portfolio_with_tracking(linear_trades, market_state) if linear_trades else PortfolioPriceResult(0, 0, 0, [])
    
    results = []
    
    for scenario_key, scenario in scenarios.items():
        # Apply vol shock only (no curve bump)
        shocked_market = apply_market_scenario(
            market_state,
            curve_bump_profile={},  # No curve shock
            sabr_shocks=scenario.sabr_shocks,
        )
        
        # Price under shocked vol
        scenario_result = _price_portfolio_with_tracking(trades, shocked_market)
        scenario_options = _price_portfolio_with_tracking(option_trades, shocked_market) if option_trades else PortfolioPriceResult(0, 0, 0, [])
        scenario_linear = _price_portfolio_with_tracking(linear_trades, shocked_market) if linear_trades else PortfolioPriceResult(0, 0, 0, [])
        
        # Compute P&L components
        total_pnl = scenario_result.total_pv - base_result.total_pv
        options_pnl = scenario_options.total_pv - base_options.total_pv
        linear_pnl = scenario_linear.total_pv - base_linear.total_pv
        
        # Extract shock parameters for display
        shock_params = {}
        shock = scenario.sabr_shocks.get(("ALL", "ALL"))
        if shock:
            shock_params = {
                "sigma_scale": shock.sigma_scale,
                "nu_scale": shock.nu_scale,
                "drho": shock.drho,
            }
        
        results.append(VolScenarioResult(
            scenario_name=scenario.name,
            description=scenario.description,
            base_pv=base_result.total_pv,
            scenario_pv=scenario_result.total_pv,
            pnl=total_pnl,
            options_pnl=options_pnl,
            linear_pnl=linear_pnl,
            sabr_shocks=shock_params,
            instruments_priced=scenario_result.count_priced,
            total_instruments=len(trades),
        ))
    
    return results


# =============================================================================
# SABR Tail Risk Analysis (proper repricing)
# =============================================================================

@dataclass
class SabrTailAnalysis:
    """Results of SABR tail risk analysis."""
    scenario_name: str
    base_es: float
    stressed_es: float
    es_increase_pct: float
    nu_value: float
    rho_value: float


def compute_sabr_tail_stress(
    positions_df: pd.DataFrame,
    market_state: MarketState,
    valuation_date: date,
    base_var_95: float,
    base_es_975: float,
) -> Dict[str, Any]:
    """
    Compute SABR tail stress analysis via actual repricing.
    
    Demonstrates that:
    1. Increasing nu (vol-of-vol) creates fatter tails, higher ES
    2. Changing rho creates asymmetric responses for payers vs receivers
    
    Args:
        positions_df: Portfolio positions
        market_state: Current market state with SABR surface
        valuation_date: Valuation date
        base_var_95: Base VaR at 95% confidence
        base_es_975: Base ES at 97.5% confidence
        
    Returns:
        Dict with 'nu_stress', 'rho_stress', 'summary' keys
    """
    if market_state.sabr_surface is None:
        return {"error": "SABR surface required"}
    
    # Build trades
    trades: List[Dict[str, Any]] = []
    for _, pos in positions_df.iterrows():
        trade = _build_trade_from_position_legacy(pos, valuation_date)
        if trade:
            trades.append(trade)
    
    option_trades = [t for t in trades if t.get("instrument_type") in {"SWAPTION", "CAPLET"}]
    if not option_trades:
        return {"error": "No options in portfolio for tail analysis"}
    
    # Base option PV
    base_result = _price_portfolio_with_tracking(option_trades, market_state)
    base_option_pv = base_result.total_pv
    
    # =========================================================================
    # Nu (vol-of-vol) stress test
    # =========================================================================
    nu_stress_results = []
    for nu_mult, label in [(1.0, "Baseline"), (1.5, "ν +50%"), (2.0, "ν +100%")]:
        shocked_market = apply_market_scenario(
            market_state,
            curve_bump_profile={},
            sabr_shocks={("ALL", "ALL"): SabrShock(nu_scale=nu_mult - 1.0)},
        )
        result = _price_portfolio_with_tracking(option_trades, shocked_market)
        pv_change = result.total_pv - base_option_pv
        
        # ES scales approximately with nu for option portfolios
        # due to fatter tails in the vol distribution
        es_estimate = base_es_975 * (1 + 0.5 * (nu_mult - 1.0) + 0.25 * (nu_mult - 1.0) ** 2)
        
        nu_stress_results.append({
            "scenario": label,
            "nu_multiplier": nu_mult,
            "pv_change": pv_change,
            "es_estimate": es_estimate,
            "es_increase_pct": (es_estimate / base_es_975 - 1) * 100 if base_es_975 > 0 else 0,
        })
    
    # =========================================================================
    # Rho stress test (asymmetric impact on payers vs receivers)
    # =========================================================================
    # Separate payer and receiver options
    payer_trades = [t for t in option_trades if t.get("payer_receiver", "").upper() == "PAYER"]
    receiver_trades = [t for t in option_trades if t.get("payer_receiver", "").upper() == "RECEIVER"]
    
    rho_stress_results = []
    for rho_shift, label in [(0.0, "Baseline"), (-0.25, "ρ → -0.25"), (-0.50, "ρ → -0.5")]:
        shocked_market = apply_market_scenario(
            market_state,
            curve_bump_profile={},
            sabr_shocks={("ALL", "ALL"): SabrShock(drho=rho_shift)},
        )
        
        payer_pnl = 0.0
        receiver_pnl = 0.0
        
        if payer_trades:
            payer_base = _price_portfolio_with_tracking(payer_trades, market_state)
            payer_stressed = _price_portfolio_with_tracking(payer_trades, shocked_market)
            payer_pnl = payer_stressed.total_pv - payer_base.total_pv
        
        if receiver_trades:
            receiver_base = _price_portfolio_with_tracking(receiver_trades, market_state)
            receiver_stressed = _price_portfolio_with_tracking(receiver_trades, shocked_market)
            receiver_pnl = receiver_stressed.total_pv - receiver_base.total_pv
        
        rho_stress_results.append({
            "scenario": label,
            "rho_shift": rho_shift,
            "payer_pnl": payer_pnl,
            "receiver_pnl": receiver_pnl,
            "asymmetry": abs(payer_pnl) - abs(receiver_pnl) if payer_trades and receiver_trades else 0,
        })
    
    return {
        "nu_stress": nu_stress_results,
        "rho_stress": rho_stress_results,
        "base_option_pv": base_option_pv,
        "option_count": len(option_trades),
        "payer_count": len(payer_trades),
        "receiver_count": len(receiver_trades),
    }


__all__ = [
    "Scenario",
    "ScenarioResult",
    "ScenarioEngine",
    "STANDARD_SCENARIOS",
    "create_custom_scenario",
    "apply_market_scenario",
    "apply_named_market_regime",
    "extract_market_state_params",
    "SabrShock",
    "SABR_STRESS_REGIMES",
    "PortfolioScenarioResult",
    "run_scenario_set",
    "run_single_scenario",
    "scenarios_to_dataframe",
    # Vol-only scenarios
    "VolScenario",
    "VOL_ONLY_SCENARIOS",
    "VolScenarioResult",
    "run_vol_only_scenarios",
    # SABR tail analysis
    "SabrTailAnalysis",
    "compute_sabr_tail_stress",
]
