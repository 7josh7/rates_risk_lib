"""
Risk limits framework.

Defines and validates risk limits for:
- Linear sensitivities (DV01, key-rate DV01)
- Option Greeks (delta, gamma, vega)
- SABR Greeks (vanna, volga, rho, nu sensitivities)
- VaR/ES limits
- Scenario loss limits
- Model diagnostics limits (RMSE, fallback thresholds)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class LimitLevel(Enum):
    """Limit breach severity levels."""
    OK = "OK"
    WARNING = "WARNING"
    BREACH = "BREACH"


@dataclass
class LimitValue:
    """
    Single limit value with warning and breach thresholds.
    
    Attributes:
        warning_threshold: Warning level (soft limit)
        breach_threshold: Breach level (hard limit)
        is_absolute: If True, compare absolute values
    """
    warning_threshold: float
    breach_threshold: float
    is_absolute: bool = True
    
    def check(self, value: float) -> LimitLevel:
        """
        Check value against limit.
        
        Args:
            value: Value to check
            
        Returns:
            LimitLevel
        """
        test_value = abs(value) if self.is_absolute else value
        
        if test_value >= self.breach_threshold:
            return LimitLevel.BREACH
        elif test_value >= self.warning_threshold:
            return LimitLevel.WARNING
        else:
            return LimitLevel.OK
    
    def utilization(self, value: float) -> float:
        """
        Compute limit utilization as percentage of breach threshold.
        
        Args:
            value: Current value
            
        Returns:
            Utilization percentage
        """
        test_value = abs(value) if self.is_absolute else value
        if self.breach_threshold == 0:
            return 0.0
        return (test_value / self.breach_threshold) * 100.0


@dataclass
class RiskLimits:
    """
    Complete set of risk limits for a trading book.
    
    All limits are optional. Missing limits are not enforced.
    """
    # Linear sensitivities
    dv01_limit: Optional[LimitValue] = None
    dv01_2y_limit: Optional[LimitValue] = None
    dv01_5y_limit: Optional[LimitValue] = None
    dv01_10y_limit: Optional[LimitValue] = None
    dv01_30y_limit: Optional[LimitValue] = None
    
    # Option Greeks
    delta_limit: Optional[LimitValue] = None
    gamma_limit: Optional[LimitValue] = None
    vega_limit: Optional[LimitValue] = None
    
    # SABR Greeks
    vanna_limit: Optional[LimitValue] = None
    volga_limit: Optional[LimitValue] = None
    
    # SABR parameter sensitivities
    rho_sensitivity_limit: Optional[LimitValue] = None
    nu_sensitivity_limit: Optional[LimitValue] = None
    
    # VaR/ES
    var_95_limit: Optional[LimitValue] = None
    var_99_limit: Optional[LimitValue] = None
    es_95_limit: Optional[LimitValue] = None
    es_99_limit: Optional[LimitValue] = None
    
    # Scenario losses
    parallel_up_100bp_limit: Optional[LimitValue] = None
    parallel_down_100bp_limit: Optional[LimitValue] = None
    steepener_limit: Optional[LimitValue] = None
    flattener_limit: Optional[LimitValue] = None
    
    # Vol scenarios
    vol_up_50pct_limit: Optional[LimitValue] = None
    vol_down_50pct_limit: Optional[LimitValue] = None
    
    # Model diagnostics
    sabr_rmse_limit: Optional[LimitValue] = None
    max_fallback_buckets_limit: Optional[LimitValue] = None
    
    # Liquidity
    lvar_limit: Optional[LimitValue] = None
    
    @classmethod
    def default_limits(cls) -> "RiskLimits":
        """
        Create default risk limits for a typical rates desk.
        
        These are representative limits and should be customized
        based on desk mandate, risk appetite, and regulatory requirements.
        
        Returns:
            RiskLimits with default values
        """
        return cls(
            # Linear DV01 - $1M breach, $750k warning
            dv01_limit=LimitValue(warning_threshold=750_000, breach_threshold=1_000_000),
            
            # Key-rate DV01 - $500k breach, $350k warning per bucket
            dv01_2y_limit=LimitValue(warning_threshold=350_000, breach_threshold=500_000),
            dv01_5y_limit=LimitValue(warning_threshold=350_000, breach_threshold=500_000),
            dv01_10y_limit=LimitValue(warning_threshold=350_000, breach_threshold=500_000),
            dv01_30y_limit=LimitValue(warning_threshold=350_000, breach_threshold=500_000),
            
            # Option Greeks
            delta_limit=LimitValue(warning_threshold=5_000_000, breach_threshold=10_000_000),
            gamma_limit=LimitValue(warning_threshold=100_000, breach_threshold=150_000),
            vega_limit=LimitValue(warning_threshold=500_000, breach_threshold=750_000),
            
            # SABR Greeks
            vanna_limit=LimitValue(warning_threshold=250_000, breach_threshold=500_000),
            volga_limit=LimitValue(warning_threshold=250_000, breach_threshold=500_000),
            
            # SABR parameter sensitivities
            rho_sensitivity_limit=LimitValue(warning_threshold=300_000, breach_threshold=500_000),
            nu_sensitivity_limit=LimitValue(warning_threshold=300_000, breach_threshold=500_000),
            
            # VaR/ES - 1-day 95% VaR at $2M breach
            var_95_limit=LimitValue(warning_threshold=1_500_000, breach_threshold=2_000_000),
            var_99_limit=LimitValue(warning_threshold=2_500_000, breach_threshold=3_000_000),
            es_95_limit=LimitValue(warning_threshold=2_000_000, breach_threshold=2_500_000),
            es_99_limit=LimitValue(warning_threshold=3_000_000, breach_threshold=4_000_000),
            
            # Scenario losses - $5M breach
            parallel_up_100bp_limit=LimitValue(warning_threshold=3_500_000, breach_threshold=5_000_000),
            parallel_down_100bp_limit=LimitValue(warning_threshold=3_500_000, breach_threshold=5_000_000),
            steepener_limit=LimitValue(warning_threshold=2_500_000, breach_threshold=3_500_000),
            flattener_limit=LimitValue(warning_threshold=2_500_000, breach_threshold=3_500_000),
            
            # Vol scenarios
            vol_up_50pct_limit=LimitValue(warning_threshold=1_000_000, breach_threshold=1_500_000),
            vol_down_50pct_limit=LimitValue(warning_threshold=1_000_000, breach_threshold=1_500_000),
            
            # Model diagnostics
            sabr_rmse_limit=LimitValue(warning_threshold=0.0010, breach_threshold=0.0020, is_absolute=False),
            max_fallback_buckets_limit=LimitValue(warning_threshold=3, breach_threshold=5, is_absolute=False),
            
            # Liquidity
            lvar_limit=LimitValue(warning_threshold=2_500_000, breach_threshold=3_500_000),
        )


@dataclass
class LimitCheckResult:
    """Result of a limit check."""
    metric_name: str
    current_value: float
    limit: LimitValue
    level: LimitLevel
    utilization_pct: float
    message: str
    
    def is_ok(self) -> bool:
        """Check if limit is OK (not warning or breach)."""
        return self.level == LimitLevel.OK
    
    def is_warning(self) -> bool:
        """Check if at warning level."""
        return self.level == LimitLevel.WARNING
    
    def is_breach(self) -> bool:
        """Check if breached."""
        return self.level == LimitLevel.BREACH


class RiskLimitChecker:
    """
    Risk limit checker.
    
    Validates current risk metrics against defined limits.
    """
    
    def __init__(self, limits: RiskLimits):
        """
        Initialize checker.
        
        Args:
            limits: Risk limits to enforce
        """
        self.limits = limits
    
    def check_all(self, metrics: Dict[str, float]) -> List[LimitCheckResult]:
        """
        Check all available metrics against limits.
        
        Args:
            metrics: Dictionary of metric_name -> value
            
        Returns:
            List of LimitCheckResult for all checked limits
        """
        results = []
        
        # Map metric names to limit attributes
        limit_mapping = {
            'dv01': 'dv01_limit',
            'dv01_2y': 'dv01_2y_limit',
            'dv01_5y': 'dv01_5y_limit',
            'dv01_10y': 'dv01_10y_limit',
            'dv01_30y': 'dv01_30y_limit',
            'delta': 'delta_limit',
            'gamma': 'gamma_limit',
            'vega': 'vega_limit',
            'vanna': 'vanna_limit',
            'volga': 'volga_limit',
            'rho_sensitivity': 'rho_sensitivity_limit',
            'nu_sensitivity': 'nu_sensitivity_limit',
            'var_95': 'var_95_limit',
            'var_99': 'var_99_limit',
            'es_95': 'es_95_limit',
            'es_99': 'es_99_limit',
            'parallel_up_100bp': 'parallel_up_100bp_limit',
            'parallel_down_100bp': 'parallel_down_100bp_limit',
            'steepener': 'steepener_limit',
            'flattener': 'flattener_limit',
            'vol_up_50pct': 'vol_up_50pct_limit',
            'vol_down_50pct': 'vol_down_50pct_limit',
            'sabr_rmse': 'sabr_rmse_limit',
            'fallback_buckets': 'max_fallback_buckets_limit',
            'lvar': 'lvar_limit',
        }
        
        for metric_name, value in metrics.items():
            limit_attr = limit_mapping.get(metric_name)
            if limit_attr is None:
                continue  # No mapping for this metric
            
            limit = getattr(self.limits, limit_attr, None)
            if limit is None:
                continue  # No limit defined
            
            # Check limit
            level = limit.check(value)
            utilization = limit.utilization(value)
            
            # Create message
            if level == LimitLevel.OK:
                message = f"✅ OK - {utilization:.1f}% of limit"
            elif level == LimitLevel.WARNING:
                message = f"⚠️ WARNING - {utilization:.1f}% of limit (threshold: ${limit.warning_threshold:,.0f})"
            else:
                message = f"🚨 BREACH - {utilization:.1f}% of limit (threshold: ${limit.breach_threshold:,.0f})"
            
            results.append(LimitCheckResult(
                metric_name=metric_name,
                current_value=value,
                limit=limit,
                level=level,
                utilization_pct=utilization,
                message=message
            ))
        
        return results
    
    def get_breaches(self, metrics: Dict[str, float]) -> List[LimitCheckResult]:
        """Get only breached limits."""
        return [r for r in self.check_all(metrics) if r.is_breach()]
    
    def get_warnings(self, metrics: Dict[str, float]) -> List[LimitCheckResult]:
        """Get warning-level limits."""
        return [r for r in self.check_all(metrics) if r.is_warning()]
    
    def has_breaches(self, metrics: Dict[str, float]) -> bool:
        """Check if any limits are breached."""
        return len(self.get_breaches(metrics)) > 0
    
    def summary_stats(self, metrics: Dict[str, float]) -> Dict[str, int]:
        """
        Get summary statistics of limit checks.
        
        Args:
            metrics: Metrics to check
            
        Returns:
            Dict with counts: total_checked, ok, warnings, breaches
        """
        results = self.check_all(metrics)
        
        return {
            'total_checked': len(results),
            'ok': sum(1 for r in results if r.is_ok()),
            'warnings': sum(1 for r in results if r.is_warning()),
            'breaches': sum(1 for r in results if r.is_breach())
        }
