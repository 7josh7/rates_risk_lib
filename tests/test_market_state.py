"""Regression tests for the combined market-state container."""

from datetime import date, timezone

from rates_risk.curves.curve import Curve
from rates_risk.market_state import CurveState, MarketState


def test_default_asof_is_timezone_aware_utc() -> None:
    curve = Curve(anchor_date=date(2024, 1, 2))
    curve.add_node(time=1.0, discount_factor=0.95)

    market_state = MarketState(curve=CurveState(discount_curve=curve))

    assert market_state.asof.tzinfo is timezone.utc
    assert market_state.asof.utcoffset().total_seconds() == 0.0
