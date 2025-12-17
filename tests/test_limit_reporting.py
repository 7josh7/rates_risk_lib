import pandas as pd
from datetime import date
from pathlib import Path

from rateslib.curves.bootstrap import bootstrap_from_quotes
from rateslib.market_state import CurveState, MarketState
from rateslib.vol.quotes import normalize_vol_quotes
from rateslib.vol.calibration import build_sabr_surface
from rateslib.risk.reporting import compute_limit_metrics
from rateslib.risk.limits import evaluate_limits, DEFAULT_LIMITS


def build_market_state():
    base = Path("data/sample_quotes")
    ois = pd.read_csv(base / "ois_quotes.csv", comment="#")
    quotes = [
        {
            "instrument_type": r.instrument_type,
            "tenor": r.tenor,
            "quote": r.rate,
            "day_count": r.day_count,
        }
        for r in ois.itertuples()
    ]
    curve = bootstrap_from_quotes(date(2024, 1, 15), quotes)
    curve_state = CurveState(curve, curve)
    vol_df = pd.read_csv("data/vol_quotes.csv")
    normalized = normalize_vol_quotes(vol_df, curve_state)
    surface = build_sabr_surface(normalized, curve_state, beta_policy=0.5)
    return MarketState(curve=curve_state, sabr_surface=surface, asof=str(date(2024, 1, 15)))


def test_compute_limit_metrics_returns_option_and_sabr_and_dv01():
    ms = build_market_state()
    positions = pd.read_csv("data/sample_book/positions.csv", comment="#")
    metrics, meta = compute_limit_metrics(ms, positions, date(2024, 1, 15))

    assert metrics["total_dv01"] is not None
    assert metrics["sabr_bucket_count"] >= 1
    assert metrics["sabr_rmse_max"] is not None
    # With the sample swaption position, option greeks should be populated
    assert metrics["option_delta"] is not None
    assert metrics["option_gamma"] is not None
    assert meta["computed_option_greeks"] is True


def test_evaluate_limits_with_overrides_no_missing():
    ms = build_market_state()
    positions = pd.read_csv("data/sample_book/positions.csv", comment="#")
    metrics, meta = compute_limit_metrics(ms, positions, date(2024, 1, 15))

    status_overrides = {}
    if metrics.get("worst_keyrate_dv01") is None and not meta.get("has_keyrate_results", False):
        status_overrides["worst_keyrate_dv01"] = "Not Computed"
    for key in ["var_95", "var_99", "es_975", "scenario_worst", "lvar_uplift"]:
        status_overrides[key] = "Not Computed"

    results = evaluate_limits(metrics, DEFAULT_LIMITS, status_overrides=status_overrides)
    statuses = {r.definition.metric_key: r.status for r in results}
    # Ensure key metrics are not marked Missing
    assert statuses["total_dv01"] != "Missing"
    assert statuses["sabr_bucket_count"] != "Missing"
    assert statuses["option_delta"] != "Missing"
    # Metrics without computation should be marked Not Computed
    assert statuses["var_95"] == "Not Computed"
