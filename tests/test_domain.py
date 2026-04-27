from datetime import date

import pandas as pd
import pytest

from rateslib.curves import create_flat_curve
from rateslib.domain import BondTrade, PricingPolicy, SwaptionTrade, normalize_trade
from rateslib.market_state import CurveState, MarketState
from rateslib.portfolio import build_typed_trade_from_position
from rateslib.pricers import price_trade
from rateslib.vol import SabrBucketParams, SabrSurfaceState


@pytest.fixture
def sample_curve():
    return create_flat_curve(date(2024, 1, 15), rate=0.05, max_tenor_years=30.0)


@pytest.fixture
def sample_surface():
    return SabrSurfaceState(
        params_by_bucket={
            ("1Y", "5Y"): SabrBucketParams(
                sigma_atm=0.0105,
                nu=0.45,
                rho=-0.10,
                beta=0.5,
            )
        },
        asof="2024-01-15",
    )


def test_normalize_trade_returns_typed_bond_trade(sample_curve):
    trade = normalize_trade(
        {
            "instrument_type": "UST",
            "settlement": date(2024, 1, 15),
            "maturity": date(2029, 1, 15),
            "coupon": 0.04,
            "notional": 1_000_000,
            "frequency": 2,
            "_position_id": "POS001",
        }
    )

    assert isinstance(trade, BondTrade)
    assert trade.instrument_type == "UST"
    assert trade.position_id == "POS001"

    market_state = MarketState(
        curve=CurveState(discount_curve=sample_curve, projection_curve=sample_curve),
        asof="2024-01-15",
    )
    result = price_trade(trade, market_state)

    assert result.pv != 0.0
    assert result.trade_id == "POS001"
    assert result.audit["position_id"] == "POS001"


def test_build_typed_trade_from_position_returns_typed_trade():
    pos = pd.Series(
        {
            "position_id": "POS002",
            "instrument_type": "UST",
            "notional": 10_000_000,
            "direction": "LONG",
            "maturity_date": "2029-01-15",
            "coupon": 0.045,
            "frequency": 2,
        }
    )

    trade = build_typed_trade_from_position(pos, date(2024, 1, 15))

    assert isinstance(trade, BondTrade)
    assert trade.trade_id == "POS002"
    assert trade.notional == 10_000_000


def test_swaption_pricing_warns_on_sabr_bucket_fallback(sample_curve, sample_surface):
    market_state = MarketState(
        curve=CurveState(discount_curve=sample_curve, projection_curve=sample_curve),
        sabr_surface=sample_surface,
        asof="2024-01-15",
        pricing_policy=PricingPolicy(sabr_bucket_fallback="warn"),
    )
    trade = SwaptionTrade(
        instrument_type="SWAPTION",
        trade_id="SWO1",
        expiry_tenor="2Y",
        swap_tenor="7Y",
        strike="ATM",
        payer_receiver="PAYER",
        notional=1_000_000,
        vol_type="NORMAL",
    )

    result = price_trade(trade, market_state)

    assert result.pv != 0.0
    assert result.warnings
    assert result.audit["sabr_lookup"]["used_fallback"] is True
    assert result.details["source_bucket"] == ("1Y", "5Y")


def test_swaption_pricing_can_fail_strictly_when_surface_bucket_is_missing(sample_curve, sample_surface):
    market_state = MarketState(
        curve=CurveState(discount_curve=sample_curve, projection_curve=sample_curve),
        sabr_surface=sample_surface,
        asof="2024-01-15",
        pricing_policy=PricingPolicy(
            sabr_bucket_fallback="error",
            allow_zero_option_vol=False,
        ),
    )
    trade = SwaptionTrade(
        instrument_type="SWAPTION",
        trade_id="SWO2",
        expiry_tenor="2Y",
        swap_tenor="7Y",
        strike="ATM",
        payer_receiver="PAYER",
        notional=1_000_000,
        vol_type="NORMAL",
    )

    with pytest.raises(ValueError, match="No SABR parameters or explicit volatility"):
        price_trade(trade, market_state)


def test_swaption_pricing_uses_explicit_vol_when_policy_blocks_surface_fallback(sample_curve, sample_surface):
    market_state = MarketState(
        curve=CurveState(discount_curve=sample_curve, projection_curve=sample_curve),
        sabr_surface=sample_surface,
        asof="2024-01-15",
        pricing_policy=PricingPolicy(
            sabr_bucket_fallback="error",
            allow_zero_option_vol=False,
        ),
    )

    result = price_trade(
        {
            "instrument_type": "SWAPTION",
            "trade_id": "SWO3",
            "expiry_tenor": "2Y",
            "swap_tenor": "7Y",
            "strike": "ATM",
            "payer_receiver": "PAYER",
            "notional": 1_000_000,
            "vol_type": "NORMAL",
            "vol": 0.0125,
        },
        market_state,
    )

    assert result.pv != 0.0
    assert result.details["implied_vol"] == pytest.approx(0.0125)
    assert result.audit["sabr_lookup"]["used_fallback"] is False
    assert result.audit["sabr_lookup"]["used_bucket"] is None
