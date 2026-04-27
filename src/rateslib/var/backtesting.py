"""
VaR backtesting utilities.

The functions in this module compare realized P&L against previously
forecast VaR numbers. VaR is assumed to be reported as a positive loss amount,
while realized P&L is positive for gains and negative for losses.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2

from ..curves.curve import Curve
from ..risk.bumping import BumpEngine


@dataclass
class VaRBacktestResult:
    """
    Result from comparing realized P&L to VaR forecasts.
    """

    confidence: float
    observations: int
    exceptions_count: int
    expected_exceptions: float
    exception_rate: float
    kupiec_lr: float
    kupiec_p_value: float
    traffic_light: str
    pnl: pd.Series = field(repr=False)
    var_forecasts: pd.Series = field(repr=False)
    exceptions: pd.Series = field(repr=False)
    christoffersen_lr: Optional[float] = None
    christoffersen_p_value: Optional[float] = None
    transition_counts: Optional[Dict[str, int]] = None

    @property
    def passed_kupiec_95(self) -> bool:
        """Return True when the Kupiec test is not rejected at 5%."""
        return self.kupiec_p_value >= 0.05

    def to_dict(self) -> Dict[str, object]:
        """Return a serializable summary."""
        return {
            "confidence": self.confidence,
            "observations": self.observations,
            "exceptions_count": self.exceptions_count,
            "expected_exceptions": self.expected_exceptions,
            "exception_rate": self.exception_rate,
            "kupiec_lr": self.kupiec_lr,
            "kupiec_p_value": self.kupiec_p_value,
            "christoffersen_lr": self.christoffersen_lr,
            "christoffersen_p_value": self.christoffersen_p_value,
            "traffic_light": self.traffic_light,
            "passed_kupiec_95": self.passed_kupiec_95,
            "transition_counts": self.transition_counts,
        }

    def exceptions_table(self) -> pd.DataFrame:
        """Return aligned P&L, VaR, and exception flags."""
        return pd.DataFrame(
            {
                "pnl": self.pnl,
                "var": self.var_forecasts,
                "exception": self.exceptions,
            }
        )


def backtest_var(
    pnl: Union[Sequence[float], pd.Series],
    var_forecasts: Union[Sequence[float], pd.Series],
    confidence: float = 0.99,
    pnl_is_loss: bool = False,
) -> VaRBacktestResult:
    """
    Backtest VaR forecasts against realized outcomes.

    Args:
        pnl: Realized P&L. Positive values are gains unless ``pnl_is_loss`` is True.
        var_forecasts: Positive VaR forecasts in loss units.
        confidence: VaR confidence level, e.g. 0.99.
        pnl_is_loss: If True, ``pnl`` is interpreted as positive realized loss.

    Returns:
        VaRBacktestResult with exception counts and statistical tests.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    pnl_series, var_series = _align_series(pnl, var_forecasts)
    if (var_series < 0).any():
        raise ValueError("var_forecasts must be non-negative")
    if len(pnl_series) == 0:
        raise ValueError("Need at least one aligned P&L/VaR observation")

    if pnl_is_loss:
        exceptions = pnl_series > var_series
    else:
        exceptions = pnl_series < -var_series

    observations = int(len(exceptions))
    exceptions_count = int(exceptions.sum())
    tail_probability = 1.0 - confidence
    expected = observations * tail_probability
    exception_rate = exceptions_count / observations

    kupiec_lr, kupiec_p_value = kupiec_pof_test(
        exceptions_count,
        observations,
        tail_probability,
    )
    christoffersen_lr, christoffersen_p_value, transition_counts = (
        christoffersen_independence_test(exceptions)
    )

    return VaRBacktestResult(
        confidence=confidence,
        observations=observations,
        exceptions_count=exceptions_count,
        expected_exceptions=expected,
        exception_rate=exception_rate,
        kupiec_lr=kupiec_lr,
        kupiec_p_value=kupiec_p_value,
        christoffersen_lr=christoffersen_lr,
        christoffersen_p_value=christoffersen_p_value,
        transition_counts=transition_counts,
        traffic_light=traffic_light(observations, exceptions_count, tail_probability),
        pnl=pnl_series,
        var_forecasts=var_series,
        exceptions=exceptions,
    )


def kupiec_pof_test(
    exceptions_count: int,
    observations: int,
    tail_probability: float,
) -> Tuple[float, float]:
    """
    Kupiec proportion-of-failures likelihood ratio test.
    """
    if observations <= 0:
        raise ValueError("observations must be positive")
    if not 0.0 < tail_probability < 1.0:
        raise ValueError("tail_probability must be between 0 and 1")
    if exceptions_count < 0 or exceptions_count > observations:
        raise ValueError("exceptions_count must be between 0 and observations")

    observed_probability = exceptions_count / observations
    log_l_null = _binomial_log_likelihood(
        observations,
        exceptions_count,
        tail_probability,
    )
    log_l_observed = _binomial_log_likelihood(
        observations,
        exceptions_count,
        observed_probability,
    )
    lr = max(0.0, -2.0 * (log_l_null - log_l_observed))
    p_value = float(chi2.sf(lr, df=1))
    return float(lr), p_value


def christoffersen_independence_test(
    exceptions: Union[Sequence[bool], pd.Series],
) -> Tuple[Optional[float], Optional[float], Dict[str, int]]:
    """
    Christoffersen independence test for exception clustering.
    """
    ex = pd.Series(exceptions).astype(bool).to_numpy()
    counts = {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
    if len(ex) < 2:
        return None, None, counts

    for prev, curr in zip(ex[:-1], ex[1:]):
        key = f"n{int(prev)}{int(curr)}"
        counts[key] += 1

    n00, n01, n10, n11 = counts["n00"], counts["n01"], counts["n10"], counts["n11"]
    total_transitions = n00 + n01 + n10 + n11
    if total_transitions == 0:
        return None, None, counts

    pi = (n01 + n11) / total_transitions
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0

    log_l_null = _bernoulli_transition_log_likelihood(n00 + n10, n01 + n11, pi)
    log_l_alt = (
        _bernoulli_transition_log_likelihood(n00, n01, pi0)
        + _bernoulli_transition_log_likelihood(n10, n11, pi1)
    )
    lr = max(0.0, -2.0 * (log_l_null - log_l_alt))
    return float(lr), float(chi2.sf(lr, df=1)), counts


def traffic_light(
    observations: int,
    exceptions_count: int,
    tail_probability: float,
) -> str:
    """
    Return a green/yellow/red exception-count band using binomial thresholds.
    """
    green_cutoff = int(binom.ppf(0.95, observations, tail_probability))
    yellow_cutoff = int(binom.ppf(0.9999, observations, tail_probability))
    if exceptions_count <= green_cutoff:
        return "green"
    if exceptions_count <= yellow_cutoff:
        return "yellow"
    return "red"


def rolling_historical_var_backtest(
    base_curve: Curve,
    historical_data: pd.DataFrame,
    pricer_func: Callable[[Curve], float],
    confidence: float = 0.99,
    lookback_days: int = 250,
    tenors: Optional[List[str]] = None,
) -> VaRBacktestResult:
    """
    Build rolling historical VaR forecasts and backtest them.

    This is a hypothetical backtest using one base curve and historical shocks:
    each historical one-day rate move is applied to ``base_curve`` to generate
    a realized P&L series, then VaR is forecast from a rolling lookback window
    of that generated P&L.
    """
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")

    changes = _historical_rate_changes(historical_data, tenors)
    if len(changes) <= lookback_days:
        raise ValueError("Need more historical changes than lookback_days")

    base_pv = pricer_func(base_curve)
    bump_engine = BumpEngine(base_curve)
    realized_pnl = []
    realized_index = []

    for idx, row in changes.iterrows():
        bump_profile = {
            tenor: float(row[tenor])
            for tenor in changes.columns
            if not pd.isna(row[tenor])
        }
        shocked_curve = bump_engine.custom_bump(bump_profile)
        realized_pnl.append(pricer_func(shocked_curve) - base_pv)
        realized_index.append(idx)

    pnl_series = pd.Series(realized_pnl, index=realized_index, dtype=float)
    forecasts = []
    forecast_index = []
    realized_outcomes = []

    percentile = (1.0 - confidence) * 100.0
    for i in range(lookback_days, len(pnl_series)):
        window = pnl_series.iloc[i - lookback_days:i]
        forecasts.append(max(0.0, -float(np.percentile(window, percentile))))
        forecast_index.append(pnl_series.index[i])
        realized_outcomes.append(float(pnl_series.iloc[i]))

    return backtest_var(
        pd.Series(realized_outcomes, index=forecast_index, dtype=float),
        pd.Series(forecasts, index=forecast_index, dtype=float),
        confidence=confidence,
        pnl_is_loss=False,
    )


def _align_series(
    pnl: Union[Sequence[float], pd.Series],
    var_forecasts: Union[Sequence[float], pd.Series],
) -> Tuple[pd.Series, pd.Series]:
    pnl_series = pnl.astype(float) if isinstance(pnl, pd.Series) else pd.Series(pnl, dtype=float)
    var_series = (
        var_forecasts.astype(float)
        if isinstance(var_forecasts, pd.Series)
        else pd.Series(var_forecasts, dtype=float)
    )

    if isinstance(pnl, pd.Series) and isinstance(var_forecasts, pd.Series):
        aligned = pd.concat([pnl_series.rename("pnl"), var_series.rename("var")], axis=1).dropna()
        return aligned["pnl"], aligned["var"]

    if len(pnl_series) != len(var_series):
        raise ValueError("pnl and var_forecasts must have the same length")
    valid = ~(pnl_series.isna() | var_series.isna())
    return pnl_series[valid].reset_index(drop=True), var_series[valid].reset_index(drop=True)


def _binomial_log_likelihood(observations: int, exceptions_count: int, probability: float) -> float:
    non_exceptions = observations - exceptions_count
    return _safe_x_log_p(exceptions_count, probability) + _safe_x_log_p(non_exceptions, 1.0 - probability)


def _bernoulli_transition_log_likelihood(non_exceptions: int, exceptions: int, probability: float) -> float:
    return _safe_x_log_p(exceptions, probability) + _safe_x_log_p(non_exceptions, 1.0 - probability)


def _safe_x_log_p(count: int, probability: float) -> float:
    if count == 0:
        return 0.0
    if probability <= 0.0:
        return -np.inf
    if probability >= 1.0:
        return 0.0 if count > 0 else -np.inf
    return float(count * np.log(probability))


def _historical_rate_changes(
    historical_data: pd.DataFrame,
    tenors: Optional[List[str]],
) -> pd.DataFrame:
    df = historical_data.copy()
    if "rate" not in df.columns and "date" in df.columns:
        value_cols = [c for c in df.columns if c.lower() != "date"]
        df = df.melt(id_vars="date", value_vars=value_cols, var_name="tenor", value_name="rate")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        pivot = df.pivot_table(index="date", columns="tenor", values="rate")
    else:
        pivot = df

    selected_tenors = tenors or list(pivot.columns)
    available_tenors = [t for t in selected_tenors if t in pivot.columns]
    if not available_tenors:
        raise ValueError("No requested risk-factor tenors are present in historical_data.")

    changes = pivot[available_tenors].dropna().diff().dropna() * 10000.0
    if changes.empty:
        raise ValueError("No historical rate changes available after differencing.")
    return changes


__all__ = [
    "VaRBacktestResult",
    "backtest_var",
    "kupiec_pof_test",
    "christoffersen_independence_test",
    "traffic_light",
    "rolling_historical_var_backtest",
]
