"""
Limit definitions and evaluation for risk metrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import math


@dataclass
class LimitDefinition:
    name: str
    metric_key: str
    category: str
    warn: float
    breach: float
    unit: str = ""
    aggregation: str = "portfolio"
    direction: str = "abs"  # "abs" or "signed"
    hard: bool = True
    description: str = ""


@dataclass
class LimitResult:
    definition: LimitDefinition
    value: Optional[float]
    status: str  # OK / Warning / Breach / Missing

    @property
    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.definition.name,
            "category": self.definition.category,
            "value": self.value,
            "limit_warn": self.definition.warn,
            "limit_breach": self.definition.breach,
            "unit": self.definition.unit,
            "status": self.status,
            "aggregation": self.definition.aggregation,
            "hard": self.definition.hard,
        }


DEFAULT_LIMITS: List[LimitDefinition] = [
    # Curve Greeks
    LimitDefinition(
        name="Total DV01",
        metric_key="total_dv01",
        category="Curve Greeks",
        warn=150_000,
        breach=200_000,
        unit="$",
        aggregation="portfolio",
        description="Absolute DV01 across book",
    ),
    LimitDefinition(
        name="Worst Key-Rate DV01",
        metric_key="worst_keyrate_dv01",
        category="Curve Greeks",
        warn=100_000,
        breach=150_000,
        unit="$",
        aggregation="portfolio",
        description="Max abs key-rate DV01",
    ),
    # Option Greeks (placeholders, evaluated only if present)
    LimitDefinition(
        name="Option Delta",
        metric_key="option_delta",
        category="Option Greeks",
        warn=5_000_000,
        breach=7_500_000,
        unit="$",
        aggregation="portfolio",
        description="Absolute option delta",
        hard=False,
    ),
    LimitDefinition(
        name="Option Gamma",
        metric_key="option_gamma",
        category="Option Greeks",
        warn=50_000,
        breach=75_000,
        unit="$ per %²",
        aggregation="portfolio",
        description="Absolute option gamma",
        hard=False,
    ),
    # SABR Greeks
    LimitDefinition(
        name="SABR Vega ATM",
        metric_key="sabr_vega_atm",
        category="SABR Greeks",
        warn=250_000,
        breach=400_000,
        unit="$ per vol pt",
        aggregation="portfolio",
    ),
    LimitDefinition(
        name="SABR Vanna (nu)",
        metric_key="sabr_vega_nu",
        category="SABR Greeks",
        warn=150_000,
        breach=250_000,
        unit="$ per nu",
        aggregation="portfolio",
        hard=False,
    ),
    LimitDefinition(
        name="SABR Rho Vega",
        metric_key="sabr_vega_rho",
        category="SABR Greeks",
        warn=100_000,
        breach=150_000,
        unit="$ per rho",
        aggregation="portfolio",
        hard=False,
    ),
    # VaR / ES
    LimitDefinition(
        name="VaR 95%",
        metric_key="var_95",
        category="VaR / ES",
        warn=1_000_000,
        breach=1_500_000,
        unit="$",
        aggregation="portfolio",
    ),
    LimitDefinition(
        name="VaR 99%",
        metric_key="var_99",
        category="VaR / ES",
        warn=1_500_000,
        breach=2_250_000,
        unit="$",
        aggregation="portfolio",
    ),
    LimitDefinition(
        name="ES 97.5%",
        metric_key="es_975",
        category="VaR / ES",
        warn=2_000_000,
        breach=3_000_000,
        unit="$",
        aggregation="portfolio",
    ),
    # Scenario losses
    LimitDefinition(
        name="Worst Scenario Loss",
        metric_key="scenario_worst",
        category="Scenario",
        warn=1_500_000,
        breach=2_500_000,
        unit="$",
        aggregation="worst-of",
    ),
    # Liquidity
    LimitDefinition(
        name="LVaR Uplift",
        metric_key="lvar_uplift",
        category="Liquidity",
        warn=0.2,
        breach=0.35,
        unit="ratio",
        aggregation="portfolio",
        description="(LVaR/Base VaR) - 1",
        hard=False,
    ),
    # Model diagnostics
    LimitDefinition(
        name="SABR RMSE (max)",
        metric_key="sabr_rmse_max",
        category="Model Diagnostics",
        warn=0.0015,
        breach=0.0025,
        unit="vol",
        aggregation="bucket",
        description="Max bucket RMSE",
    ),
    LimitDefinition(
        name="SABR Buckets Calibrated",
        metric_key="sabr_bucket_count",
        category="Model Diagnostics",
        warn=3,
        breach=2,
        unit="count",
        aggregation="portfolio",
        direction="signed",  # compare directly, no abs
        description="Minimum calibrated buckets",
        hard=True,
    ),
]


def evaluate_limit(value: Optional[float], limit_def: LimitDefinition) -> LimitResult:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return LimitResult(definition=limit_def, value=None, status="Missing")

    val = abs(value) if limit_def.direction == "abs" else value
    status = "OK"
    if val >= limit_def.breach:
        status = "Breach"
    elif val >= limit_def.warn:
        status = "Warning"
    return LimitResult(definition=limit_def, value=value, status=status)


def evaluate_limits(metrics: Dict[str, float], limits: List[LimitDefinition] = None) -> List[LimitResult]:
    limits = limits or DEFAULT_LIMITS
    results: List[LimitResult] = []
    for lim in limits:
        value = metrics.get(lim.metric_key)
        results.append(evaluate_limit(value, lim))
    return results


def limits_to_table(results: List[LimitResult]) -> List[Dict[str, Any]]:
    return [r.as_dict for r in results]
