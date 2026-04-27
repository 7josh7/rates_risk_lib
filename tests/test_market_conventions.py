from datetime import date

import pandas as pd
import pytest

from rateslib.curves import create_flat_curve
from rateslib.market_conventions import (
    MarketConventionError,
    available_market_conventions,
    resolve_market_convention_for_trade,
)
from rateslib.market_state import CurveState, MarketState
from rateslib.portfolio.builders import build_bond_trade, build_swap_trade
from rateslib.pricers import price_trade


@pytest.fixture
def sample_market_state():
    curve = create_flat_curve(date(2024, 1, 15), rate=0.05, max_tenor_years=30.0, currency="USD")
    return MarketState(curve=CurveState(discount_curve=curve, projection_curve=curve), asof="2024-01-15")


def test_market_convention_registry_exposes_named_templates():
    templates = available_market_conventions()

    assert "USD_UST" in templates
    assert "USD_VANILLA_SWAP" in templates
    assert templates["USD_UST"].currency == "USD"


def test_resolve_market_convention_from_curve_currency_default():
    resolved = resolve_market_convention_for_trade(
        {"instrument_type": "UST"},
        curve_currency="USD",
    )

    assert resolved.template_id == "USD_UST"
    assert resolved.source == "curve_currency_default"
    assert resolved.bond_conventions is not None
    assert resolved.bond_conventions.settlement_days == 1


def test_resolve_market_convention_rejects_unknown_explicit_template():
    with pytest.raises(MarketConventionError, match="Unknown market convention"):
        resolve_market_convention_for_trade(
            {"instrument_type": "UST", "market_convention": "NO_SUCH_TEMPLATE"},
            curve_currency="USD",
        )


def test_build_bond_trade_uses_template_settlement_lag_from_trade_date():
    pos = pd.Series(
        {
            "position_id": "UST1",
            "instrument_type": "UST",
            "currency": "USD",
            "trade_date": "2024-07-03",
            "notional": 1_000_000,
            "direction": "LONG",
            "maturity_date": "2029-01-15",
            "coupon": 0.04,
        }
    )

    trade = build_bond_trade(pos, valuation_date=date(2024, 7, 3))

    assert trade["settlement"] == date(2024, 7, 5)
    assert trade["currency"] == "USD"


def test_build_swap_trade_uses_template_spot_start_from_trade_date():
    pos = pd.Series(
        {
            "position_id": "IRS1",
            "instrument_type": "IRS",
            "currency": "USD",
            "trade_date": "2024-07-03",
            "notional": 5_000_000,
            "direction": "PAY_FIXED",
            "maturity_date": "2029-07-08",
            "coupon": 0.04,
        }
    )

    trade = build_swap_trade(pos, valuation_date=date(2024, 7, 3))

    assert trade["effective"] == date(2024, 7, 8)
    assert trade["currency"] == "USD"


def test_price_trade_records_resolved_market_convention_in_audit(sample_market_state):
    result = price_trade(
        {
            "instrument_type": "UST",
            "settlement": date(2024, 1, 15),
            "maturity": date(2029, 1, 15),
            "coupon": 0.04,
            "notional": 1_000_000,
            "frequency": 2,
        },
        sample_market_state,
    )

    market_audit = result.audit["market_convention"]
    assert market_audit["template_id"] == "USD_UST"
    assert market_audit["source"] == "curve_currency_default"
    assert market_audit["bond_conventions"]["holiday_calendar"] == "UnitedStatesHolidayCalendar"
