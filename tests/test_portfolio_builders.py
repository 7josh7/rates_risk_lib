"""
Tests for production-grade option trade builders.

These tests verify the explicit trade builder system enforces:
1. No inference for options (all fields must be explicit)
2. Correct sign conventions (LONG/SHORT, PAYER/RECEIVER)
3. CAPLET stays CAPLET (no coercion to SWAPTION)
4. Missing required fields raise descriptive errors
5. Portfolio pricing reports failures explicitly
"""

import pytest
import pandas as pd
from datetime import date
from typing import Dict, Any

from rateslib.portfolio.builders import (
    # Trade builders
    build_bond_trade,
    build_swap_trade,
    build_swaption_trade,
    build_caplet_trade,
    build_trade_from_position,
    # Portfolio pricing
    price_portfolio_with_diagnostics,
    PortfolioPricingResult,
    TradeFailure,
    # Sign conventions
    SIGN_LONG,
    SIGN_SHORT,
    get_position_sign,
    get_payer_receiver_sign,
    # Exceptions
    PositionValidationError,
    MissingFieldError,
    InvalidOptionError,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def valuation_date():
    return date(2024, 1, 15)


@pytest.fixture
def valid_bond_position():
    """A valid bond position with all required fields."""
    return pd.Series({
        "position_id": "POS001",
        "instrument_type": "UST",
        "notional": 10_000_000,
        "direction": "LONG",
        "maturity_date": "2029-01-15",
        "coupon": 0.045,
        "frequency": 2,
    })


@pytest.fixture
def valid_swap_position():
    """A valid swap position with all required fields."""
    return pd.Series({
        "position_id": "POS002",
        "instrument_type": "IRS",
        "notional": 25_000_000,
        "direction": "PAY_FIXED",
        "maturity_date": "2034-01-15",
        "coupon": 0.0415,
    })


@pytest.fixture
def valid_swaption_position():
    """A valid swaption position with EXPLICIT fields (no inference)."""
    return pd.Series({
        "position_id": "POS011",
        "instrument_type": "SWAPTION",
        "option_type": "SWAPTION",
        "expiry_tenor": "1Y",
        "underlying_swap_tenor": "5Y",
        "payer_receiver": "PAYER",
        "position": "LONG",
        "strike": "ATM",
        "notional": 10_000_000,
        "vol_type": "NORMAL",
    })


@pytest.fixture
def valid_caplet_position():
    """A valid caplet position with EXPLICIT fields (no inference)."""
    return pd.Series({
        "position_id": "POS014",
        "instrument_type": "CAPLET",
        "option_type": "CAPLET",
        "caplet_start_date": "2024-06-15",
        "caplet_end_date": "2024-09-15",
        "position": "LONG",
        "strike": "ATM",
        "notional": 15_000_000,
        "is_cap": True,
        "vol_type": "NORMAL",
    })


# =============================================================================
# Test Sign Conventions
# =============================================================================

class TestSignConventions:
    """Verify sign conventions are consistently applied."""
    
    def test_sign_long_is_positive(self):
        assert SIGN_LONG == 1.0
    
    def test_sign_short_is_negative(self):
        assert SIGN_SHORT == -1.0
    
    def test_get_position_sign_long(self):
        assert get_position_sign("LONG") == SIGN_LONG
        assert get_position_sign("BUY") == SIGN_LONG
        assert get_position_sign("REC_FIXED") == SIGN_LONG
        assert get_position_sign("RECEIVE") == SIGN_LONG
    
    def test_get_position_sign_short(self):
        assert get_position_sign("SHORT") == SIGN_SHORT
        assert get_position_sign("SELL") == SIGN_SHORT
        assert get_position_sign("PAY_FIXED") == SIGN_SHORT
        assert get_position_sign("PAY") == SIGN_SHORT
    
    def test_get_position_sign_invalid(self):
        with pytest.raises(ValueError, match="Ambiguous direction"):
            get_position_sign("NEUTRAL")
    
    def test_payer_receiver_sign_payer(self):
        """Payer swaption: +delta when rates ↑"""
        assert get_payer_receiver_sign("PAYER") == SIGN_LONG
    
    def test_payer_receiver_sign_receiver(self):
        """Receiver swaption: −delta when rates ↑"""
        assert get_payer_receiver_sign("RECEIVER") == SIGN_SHORT
    
    def test_payer_receiver_sign_invalid(self):
        with pytest.raises(ValueError, match="must be 'PAYER' or 'RECEIVER'"):
            get_payer_receiver_sign("STRADDLE")


# =============================================================================
# Test Bond Builder
# =============================================================================

class TestBondBuilder:
    """Test bond trade builder."""
    
    def test_valid_bond_builds_successfully(self, valid_bond_position, valuation_date):
        trade = build_bond_trade(valid_bond_position, valuation_date)
        
        assert trade["instrument_type"] == "UST"
        assert trade["settlement"] == valuation_date
        assert trade["maturity"] == date(2029, 1, 15)
        assert trade["coupon"] == 0.045
        assert trade["notional"] == 10_000_000  # LONG = positive
        assert trade["frequency"] == 2
    
    def test_short_bond_has_negative_notional(self, valid_bond_position, valuation_date):
        valid_bond_position["direction"] = "SHORT"
        trade = build_bond_trade(valid_bond_position, valuation_date)
        
        assert trade["notional"] == -10_000_000  # SHORT = negative
    
    def test_missing_maturity_raises_error(self, valid_bond_position, valuation_date):
        valid_bond_position["maturity_date"] = None
        
        with pytest.raises(MissingFieldError, match="maturity_date"):
            build_bond_trade(valid_bond_position, valuation_date)
    
    def test_missing_notional_raises_error(self, valuation_date):
        pos = pd.Series({
            "position_id": "TEST",
            "instrument_type": "BOND",
            "direction": "LONG",
            "maturity_date": "2029-01-15",
        })
        
        with pytest.raises(MissingFieldError, match="notional"):
            build_bond_trade(pos, valuation_date)


# =============================================================================
# Test Swap Builder
# =============================================================================

class TestSwapBuilder:
    """Test swap trade builder."""
    
    def test_valid_swap_builds_successfully(self, valid_swap_position, valuation_date):
        trade = build_swap_trade(valid_swap_position, valuation_date)
        
        assert trade["instrument_type"] == "SWAP"
        assert trade["pay_receive"] == "PAY"  # PAY_FIXED → PAY
        assert trade["fixed_rate"] == 0.0415
        assert trade["notional"] == 25_000_000
    
    def test_receive_fixed_swap(self, valid_swap_position, valuation_date):
        valid_swap_position["direction"] = "REC_FIXED"
        trade = build_swap_trade(valid_swap_position, valuation_date)
        
        assert trade["pay_receive"] == "RECEIVE"
    
    def test_ambiguous_direction_raises_error(self, valid_swap_position, valuation_date):
        valid_swap_position["direction"] = "BOTH"  # Ambiguous
        
        with pytest.raises(PositionValidationError, match="ambiguous"):
            build_swap_trade(valid_swap_position, valuation_date)


# =============================================================================
# Test Swaption Builder - CRITICAL: No Inference
# =============================================================================

class TestSwaptionBuilder:
    """Test swaption builder with strict NO INFERENCE policy."""
    
    def test_valid_swaption_builds_successfully(self, valid_swaption_position, valuation_date):
        trade = build_swaption_trade(valid_swaption_position, valuation_date)
        
        assert trade["instrument_type"] == "SWAPTION"
        assert trade["expiry_tenor"] == "1Y"
        assert trade["swap_tenor"] == "5Y"
        assert trade["payer_receiver"] == "PAYER"
        assert trade["notional"] == 10_000_000  # LONG = positive
        assert trade["strike"] == "ATM"
    
    def test_short_swaption_has_negative_notional(self, valid_swaption_position, valuation_date):
        valid_swaption_position["position"] = "SHORT"
        trade = build_swaption_trade(valid_swaption_position, valuation_date)
        
        assert trade["notional"] == -10_000_000  # SHORT = negative
    
    def test_receiver_swaption_is_receiver(self, valid_swaption_position, valuation_date):
        valid_swaption_position["payer_receiver"] = "RECEIVER"
        trade = build_swaption_trade(valid_swaption_position, valuation_date)
        
        assert trade["payer_receiver"] == "RECEIVER"
        assert trade["_payer_receiver_sign"] == SIGN_SHORT  # Receiver = -delta
    
    def test_missing_expiry_raises_error(self, valuation_date):
        """CRITICAL: Cannot infer expiry from maturity."""
        pos = pd.Series({
            "position_id": "TEST",
            "instrument_type": "SWAPTION",
            "maturity_date": "2029-01-15",  # NOT expiry!
            "underlying_swap_tenor": "5Y",
            "payer_receiver": "PAYER",
            "position": "LONG",
            "notional": 1_000_000,
        })
        
        with pytest.raises(InvalidOptionError, match="expiry.*not allowed"):
            build_swaption_trade(pos, valuation_date)
    
    def test_missing_swap_tenor_raises_error(self, valid_swaption_position, valuation_date):
        del valid_swaption_position["underlying_swap_tenor"]
        valid_swaption_position["swap_tenor"] = None
        
        with pytest.raises(MissingFieldError, match="underlying_swap_tenor"):
            build_swaption_trade(valid_swaption_position, valuation_date)
    
    def test_missing_payer_receiver_raises_error(self, valid_swaption_position, valuation_date):
        valid_swaption_position["payer_receiver"] = None
        
        with pytest.raises(MissingFieldError, match="payer_receiver"):
            build_swaption_trade(valid_swaption_position, valuation_date)
    
    def test_missing_position_raises_error(self, valid_swaption_position, valuation_date):
        valid_swaption_position["position"] = None
        valid_swaption_position["direction"] = None
        
        with pytest.raises(MissingFieldError, match="position"):
            build_swaption_trade(valid_swaption_position, valuation_date)
    
    def test_invalid_payer_receiver_raises_error(self, valid_swaption_position, valuation_date):
        valid_swaption_position["payer_receiver"] = "STRADDLE"
        
        with pytest.raises(InvalidOptionError, match="must be 'PAYER' or 'RECEIVER'"):
            build_swaption_trade(valid_swaption_position, valuation_date)


# =============================================================================
# Test Caplet Builder - CRITICAL: No Coercion to Swaption
# =============================================================================

class TestCapletBuilder:
    """Test caplet builder - must NOT coerce to swaption."""
    
    def test_valid_caplet_builds_successfully(self, valid_caplet_position, valuation_date):
        trade = build_caplet_trade(valid_caplet_position, valuation_date)
        
        assert trade["instrument_type"] == "CAPLET"  # NOT SWAPTION!
        assert trade["start_date"] == date(2024, 6, 15)
        assert trade["end_date"] == date(2024, 9, 15)
        assert trade["notional"] == 15_000_000  # LONG = positive
        assert trade["is_cap"] == True
    
    def test_caplet_is_never_swaption(self, valid_caplet_position, valuation_date):
        """CRITICAL: Caplet must never be converted to swaption."""
        trade = build_caplet_trade(valid_caplet_position, valuation_date)
        
        # Verify it's a CAPLET trade, not SWAPTION
        assert trade["instrument_type"] == "CAPLET"
        assert "expiry_tenor" in trade  # For SABR lookup
        assert "index_tenor" in trade  # For SABR lookup
        assert "start_date" in trade  # Caplet-specific
        assert "end_date" in trade    # Caplet-specific
    
    def test_short_caplet_has_negative_notional(self, valid_caplet_position, valuation_date):
        valid_caplet_position["position"] = "SHORT"
        trade = build_caplet_trade(valid_caplet_position, valuation_date)
        
        assert trade["notional"] == -15_000_000  # SHORT = negative
    
    def test_floorlet_is_cap_false(self, valid_caplet_position, valuation_date):
        valid_caplet_position["is_cap"] = False
        trade = build_caplet_trade(valid_caplet_position, valuation_date)
        
        assert trade["is_cap"] == False
    
    def test_missing_start_date_raises_error(self, valid_caplet_position, valuation_date):
        valid_caplet_position["caplet_start_date"] = None
        
        with pytest.raises(MissingFieldError, match="caplet_start_date"):
            build_caplet_trade(valid_caplet_position, valuation_date)
    
    def test_missing_end_date_raises_error(self, valid_caplet_position, valuation_date):
        valid_caplet_position["caplet_end_date"] = None
        
        with pytest.raises(MissingFieldError, match="caplet_end_date"):
            build_caplet_trade(valid_caplet_position, valuation_date)
    
    def test_end_before_start_raises_error(self, valid_caplet_position, valuation_date):
        valid_caplet_position["caplet_start_date"] = "2024-09-15"
        valid_caplet_position["caplet_end_date"] = "2024-06-15"  # Before start!
        
        with pytest.raises(InvalidOptionError, match="must be after"):
            build_caplet_trade(valid_caplet_position, valuation_date)
    
    def test_expired_caplet_raises_error(self, valid_caplet_position, valuation_date):
        valid_caplet_position["caplet_start_date"] = "2023-06-15"  # Before valuation!
        valid_caplet_position["caplet_end_date"] = "2023-09-15"
        
        with pytest.raises(InvalidOptionError, match="must be after valuation_date"):
            build_caplet_trade(valid_caplet_position, valuation_date)


# =============================================================================
# Test Unified Trade Builder Dispatch
# =============================================================================

class TestBuildTradeFromPosition:
    """Test the unified trade builder dispatcher."""
    
    def test_dispatches_to_bond_builder(self, valid_bond_position, valuation_date):
        trade = build_trade_from_position(valid_bond_position, valuation_date)
        assert trade["instrument_type"] == "UST"
    
    def test_dispatches_to_swap_builder(self, valid_swap_position, valuation_date):
        trade = build_trade_from_position(valid_swap_position, valuation_date)
        assert trade["instrument_type"] == "SWAP"
    
    def test_dispatches_to_swaption_builder(self, valid_swaption_position, valuation_date):
        trade = build_trade_from_position(valid_swaption_position, valuation_date)
        assert trade["instrument_type"] == "SWAPTION"
    
    def test_dispatches_to_caplet_builder(self, valid_caplet_position, valuation_date):
        trade = build_trade_from_position(valid_caplet_position, valuation_date)
        assert trade["instrument_type"] == "CAPLET"
    
    def test_option_type_field_takes_priority(self, valuation_date):
        """option_type field should override instrument_type."""
        pos = pd.Series({
            "position_id": "TEST",
            "instrument_type": "SWAPTION",  # Says swaption
            "option_type": "CAPLET",        # But option_type says caplet
            "caplet_start_date": "2024-06-15",
            "caplet_end_date": "2024-09-15",
            "position": "LONG",
            "strike": "ATM",
            "notional": 1_000_000,
        })
        
        trade = build_trade_from_position(pos, valuation_date)
        assert trade["instrument_type"] == "CAPLET"  # option_type wins
    
    def test_unknown_instrument_raises_error(self, valuation_date):
        pos = pd.Series({
            "position_id": "TEST",
            "instrument_type": "EXOTIC_OPTION",
            "notional": 1_000_000,
        })
        
        with pytest.raises(PositionValidationError, match="Unknown instrument_type"):
            build_trade_from_position(pos, valuation_date)


# =============================================================================
# Test Portfolio Pricing with Failure Tracking
# =============================================================================

class TestPortfolioPricingWithDiagnostics:
    """Test that portfolio pricing never silently swallows errors."""
    
    def test_empty_portfolio_returns_empty_result(self, valuation_date):
        empty_df = pd.DataFrame(columns=["position_id", "instrument_type"])
        
        # Create minimal market state (would need actual implementation)
        # For this test, we just verify the interface
        result = PortfolioPricingResult(
            total_pv=0.0,
            successful_trades=[],
            successful_pvs=[],
            failed_trades=[],
            total_positions=0,
        )
        
        assert result.successful_count == 0
        assert result.failed_count == 0
        assert result.coverage_ratio == 1.0
        assert not result.has_failures
    
    def test_trade_failure_captures_error_details(self):
        failure = TradeFailure(
            position_id="POS001",
            instrument_type="SWAPTION",
            error_type="MissingFieldError",
            error_message="SWAPTION requires field 'expiry_tenor'",
            stage="build",
        )
        
        assert failure.position_id == "POS001"
        assert failure.stage == "build"
        assert "expiry_tenor" in failure.error_message
        
        # Test to_dict
        d = failure.to_dict()
        assert d["position_id"] == "POS001"
        assert d["error_type"] == "MissingFieldError"
    
    def test_pricing_result_coverage_calculation(self):
        result = PortfolioPricingResult(
            total_pv=100_000,
            successful_trades=[{"id": 1}, {"id": 2}],
            successful_pvs=[60_000, 40_000],
            failed_trades=[
                TradeFailure("POS003", "SWAPTION", "Error", "msg", "build")
            ],
            total_positions=3,
        )
        
        assert result.successful_count == 2
        assert result.failed_count == 1
        assert result.coverage_ratio == pytest.approx(2/3, rel=0.01)
        assert result.has_failures
    
    def test_pricing_result_generates_warnings(self):
        result = PortfolioPricingResult(
            total_pv=100_000,
            successful_trades=[{"id": 1}],
            successful_pvs=[100_000],
            failed_trades=[
                TradeFailure("POS002", "SWAPTION", "MissingFieldError", "msg", "build"),
                TradeFailure("POS003", "CAPLET", "InvalidOptionError", "msg", "build"),
            ],
            total_positions=3,
        )
        
        warnings = result.get_warnings()
        assert len(warnings) > 0
        assert "2/3" in warnings[0]  # Shows failure count
        assert "build:MissingFieldError" in warnings[1]  # Groups by error type


# =============================================================================
# Test Complete Workflow - Integration Tests
# =============================================================================

class TestCompleteWorkflow:
    """Integration tests for the complete option trade construction workflow."""
    
    def test_long_payer_swaption_signs(self, valid_swaption_position, valuation_date):
        """LONG PAYER swaption: +notional, +delta when rates ↑"""
        trade = build_swaption_trade(valid_swaption_position, valuation_date)
        
        assert trade["notional"] > 0  # LONG
        assert trade["_position_sign"] == SIGN_LONG
        assert trade["_payer_receiver_sign"] == SIGN_LONG  # PAYER
    
    def test_short_receiver_swaption_signs(self, valid_swaption_position, valuation_date):
        """SHORT RECEIVER swaption: -notional, -delta when rates ↑"""
        valid_swaption_position["position"] = "SHORT"
        valid_swaption_position["payer_receiver"] = "RECEIVER"
        trade = build_swaption_trade(valid_swaption_position, valuation_date)
        
        assert trade["notional"] < 0  # SHORT
        assert trade["_position_sign"] == SIGN_SHORT
        assert trade["_payer_receiver_sign"] == SIGN_SHORT  # RECEIVER
    
    def test_long_payer_vs_short_receiver_same_delta_sign(self, valid_swaption_position, valuation_date):
        """Verify delta sign calculation is consistent."""
        # LONG PAYER: notional(+) × payer_sign(+) = + delta
        long_payer = build_swaption_trade(valid_swaption_position, valuation_date)
        long_payer_delta_sign = long_payer["_position_sign"] * long_payer["_payer_receiver_sign"]
        
        # SHORT RECEIVER: notional(-) × receiver_sign(-) = + delta  
        valid_swaption_position["position"] = "SHORT"
        valid_swaption_position["payer_receiver"] = "RECEIVER"
        short_receiver = build_swaption_trade(valid_swaption_position, valuation_date)
        short_receiver_delta_sign = short_receiver["_position_sign"] * short_receiver["_payer_receiver_sign"]
        
        # Both should have same effective delta sign
        assert long_payer_delta_sign == short_receiver_delta_sign == 1.0
    
    def test_long_receiver_vs_short_payer_same_delta_sign(self, valid_swaption_position, valuation_date):
        """Verify opposite delta sign calculation."""
        # LONG RECEIVER: notional(+) × receiver_sign(-) = - delta
        valid_swaption_position["payer_receiver"] = "RECEIVER"
        long_receiver = build_swaption_trade(valid_swaption_position, valuation_date)
        long_receiver_delta_sign = long_receiver["_position_sign"] * long_receiver["_payer_receiver_sign"]
        
        # SHORT PAYER: notional(-) × payer_sign(+) = - delta
        valid_swaption_position["position"] = "SHORT"
        valid_swaption_position["payer_receiver"] = "PAYER"
        short_payer = build_swaption_trade(valid_swaption_position, valuation_date)
        short_payer_delta_sign = short_payer["_position_sign"] * short_payer["_payer_receiver_sign"]
        
        # Both should have same effective delta sign (opposite to above)
        assert long_receiver_delta_sign == short_payer_delta_sign == -1.0
